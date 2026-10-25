"""Three-round, source-disjoint evaluation on original human-labeled frames."""
import argparse,base64,copy,hashlib,html,io,json,math,os,sys,time,zipfile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import cv2,numpy as np,torch,yaml
from ultralytics import YOLO
from ultralytics.engine.results import Results
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tracker import SeparateVehicleTracker,read_report,save_report
from counting import LineCounter
from detection import PersonViews
from evaluation import tracking_metrics,reference_crossings,crossing_metrics,box_iou,ignore_fraction
from scipy.optimize import linear_sum_assignment
DATA=ROOT/'data/human-labeled';OUTPUT=ROOT/'outputs/human-labeled-evaluation';TEMP=Path('/private/tmp/propsight-benchmark');CACHE=TEMP/'raw-cache';CACHE.mkdir(exist_ok=True)
PLAN=json.loads((ROOT/'configs/human-labeled-evaluation.json').read_text())
MODELS={};VIEWS={};ARCHIVES={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def model(name):
 if name not in MODELS:MODELS[name]=YOLO(str(ROOT/name))
 return MODELS[name]
def image(seq,index):
 spec=PLAN['sequences'][seq];archive=spec['archive']
 if archive not in ARCHIVES:ARCHIVES[archive]=zipfile.ZipFile(DATA/archive)
 filename=f'{index+1:06d}.jpg' if seq.startswith('MOT17') else f'img{index+1:05d}.jpg'
 data=ARCHIVES[archive].read(f'{seq}/img1/{filename}');im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
 if im is None:raise ValueError(f'Undecodable original frame {seq}/{filename}')
 return im

def labels(seq):
 spec=PLAN['sequences'][seq];z=zipfile.ZipFile(DATA/spec['archive']);frames=[[] for _ in range(spec['frames'])];regions=[]
 if spec['kind']=='person':
  rows=np.loadtxt(io.BytesIO(z.read(f'{seq}/gt.txt')),delimiter=',')
  for frame,identity,x,y,w,h,valid,cls,visibility in rows:
   frames[int(frame)-1].append([int(identity),x,y,x+w,y+h,valid,int(cls),visibility])
 else:
  xml=ET.fromstring(z.read(f'{seq}/annotations.xml'))
  def box(el):
   a=el.attrib;x,y,w,h=[float(a[k]) for k in ['left','top','width','height']];return [x,y,x+w,y+h]
  regions=[box(el) for el in xml.findall('ignored_region/box')]
  for frame in xml.findall('frame'):
   for target in frame.findall('target_list/target'):
    bounds=box(target.find('box'));a=target.find('attribute').attrib;area=(bounds[2]-bounds[0])*(bounds[3]-bounds[1]);overlap=0
    for region in target.findall('occlusion/region_overlap'):
     if region.attrib.get('occlusion_status') in ['1','-1']:
      overlap+=float(region.attrib['width'])*float(region.attrib['height'])
    reliability=max(0,(1-float(a.get('truncation_ratio',0)))*(1-min(1,overlap/max(area,1))))
    frames[int(frame.attrib['num'])-1].append([int(target.attrib['id']),*bounds,1,1,reliability])
 return frames,regions

def baseline():
 return {'name':'current-retail','model':'yolo11s.pt','imgsz':1280,'conf':.1,'iou':.7,
         'camera':json.loads((ROOT/'configs/retail-security.json').read_text()),'association':{}}

def detection(image_,profile,classes,views=True):
 detector=model(profile['model']);r=detector.predict(image_,classes=classes,imgsz=profile['imgsz'],conf=profile['conf'],iou=profile['iou'],device='cpu',verbose=False)[0]
 settings=profile['camera'].get('person_detection') if views and 0 in classes else None
 if settings:
  key=json.dumps(settings,sort_keys=True)
  if key not in VIEWS:VIEWS[key]=PersonViews(settings,ROOT)
  r=VIEWS[key].predict(r,detector,image_,'cpu',profile['conf'],profile['iou'])
 return r.boxes.data.cpu().tolist()

def raw(seq,profile,fps):
 spec=PLAN['sequences'][seq];step=round(spec['fps']/fps)
 if abs(spec['fps']/step-fps)>1e-8:raise ValueError('Sampling must divide native FPS exactly')
 indices=list(range(0,spec['frames'],step));detector_key={'model':sha(ROOT/profile['model']),'imgsz':profile['imgsz'],'conf':profile['conf'],'iou':profile['iou'],'views':profile['camera'].get('person_detection') if spec['kind']=='person' else None,'code':sha(ROOT/'detection.py'),'source':sha(DATA/spec['archive'])}
 key=hashlib.sha256(json.dumps(detector_key,sort_keys=True).encode()).hexdigest()[:16];path=CACHE/f'{seq}-{fps}-{key}.json'
 if path.exists():return json.loads(path.read_text()),indices
 rows=[];started=time.monotonic()
 for number,index in enumerate(indices):
  rows.append(detection(image(seq,index),profile,[0] if spec['kind']=='person' else [2,5,7]))
  if (number+1)%50==0:print(f'{seq} {profile["name"]} {fps}fps: {number+1}/{len(indices)} frames ({time.monotonic()-started:.0f}s)',flush=True)
 path.write_text(json.dumps(rows));return rows,indices

def eligible_predictions(gt,pred,kind,regions):
 if not pred:return pred
 ious=box_iou([r[1:5] for r in gt],[r[1:5] for r in pred]);matched={};ignored=set()
 if len(gt):
  score=ious.copy();score[score<.5]=0;x,y=linear_sum_assignment(-score)
  matched={int(j):int(i) for i,j in zip(x,y) if score[i,j]>0}
 if kind=='person':ignored={j for j,i in matched.items() if int(gt[i][6]) in [2,7,8,12]}
 else:ignored={j for j,p in enumerate(pred) if j not in matched and ignore_fraction(p[1:5],regions)>=.5}
 return [p for j,p in enumerate(pred) if j not in ignored]

def track(raw_frames,profile,fps,shape,line,kind,gt=None,regions=None):
 names=model(profile['model']).names;tracker=SeparateVehicleTracker(ROOT/'configs/bytetrack-low-fps.yaml',profile['camera']['vehicle_tracking'],profile['camera'].get('person_tracking'))
 for group in tracker.trackers:
  tracker.settings[group].update(profile.get('association',{}))
  for key,value in profile.get('association',{}).items():setattr(tracker.trackers[group].args,key,value)
 height,width=shape;points=[tuple(round(p*(size-1)) for p,size in zip(point,(width,height))) for point in line]
 counter=LineCounter(*points,entry_side='positive',deadband=3 if kind=='vehicle' else 8,confirm_frames=3,initial_confirm_frames=1 if kind=='vehicle' else 3,max_gap_frames=max(1,round(fps*.5)))
 blank=np.zeros((height,width,3),np.uint8);observations=[];events=[];inverse={name:cls for cls,name in names.items()}
 for frame,boxes in enumerate(raw_frames):
  counter.prune(frame);result=Results(blank,path='',names=names,boxes=torch.tensor(np.asarray(boxes,dtype=np.float32).reshape(-1,6)))
  rows=tracker.update(result,blank,names,.1,.7);pred=[]
  for xy,identity,label,confidence,group in rows:
   if group!=kind:continue
   pred.append([int(identity.split('-')[-1]),*xy,confidence,inverse[label]])
  observations.append(pred)
  selected=eligible_predictions(gt[frame],pred,kind,regions or []) if gt is not None else pred
  for identity,x1,y1,x2,y2,confidence,cls in selected:
   point=[(x1+x2)/2,y2];event=counter.update(identity,point,frame)
   if event:events.append({'frame':frame,'video_seconds':frame/fps,'track_id':identity,'direction':event,'point':point,'coco_class':int(cls)})
 return observations,events

def evaluate(seq,profile,fps):
 spec=PLAN['sequences'][seq];kind=spec['kind'];raw_frames,indices=raw(seq,profile,fps);native_gt,regions=labels(seq);gt=[native_gt[i] for i in indices]
 observations,events=track(raw_frames,profile,fps,(spec['height'],spec['width']),spec['line'],kind,gt,regions)
 summary,details=tracking_metrics(gt,observations,kind,regions)
 points=[[round(p*(size-1)) for p,size in zip(point,(spec['width'],spec['height']))] for point in spec['line']]
 reference=reference_crossings(gt,points,fps,kind,indices,spec['width'],spec['height']);crossings=crossing_metrics(reference,events,fps,spec['width'])
 return {'sequence':seq,'kind':kind,'fps':fps,'native_fps':spec['fps'],'frames_processed':len(indices),'source_frame_indices':indices,'profile':copy.deepcopy(profile),'metrics':summary,'metric_details':details,'crossings':crossings,'observations':observations,'observation_columns':['track_id','x1','y1','x2','y2','confidence','coco_class'],'finished_at_utc':stamp()}

def guard(profile):
 results={}
 for tag,source,config in [('meva',Path('/Users/andykim/project/propsight/data/meva/demo-excerpts/forward-5fps-crf18.mp4'),'meva-person.json'),('kathy',ROOT/'data/earthcam-kathy-pub.mp4','kathy-pub.json')]:
  digest=hashlib.sha256(json.dumps({'detector':profile['model'],'imgsz':profile['imgsz'],'views':profile['camera'].get('person_detection'),'source':sha(source)},sort_keys=True).encode()).hexdigest()[:16];path=CACHE/f'guard-{tag}-{digest}.json'
  if path.exists():raw_frames=json.loads(path.read_text())
  else:
   cap=cv2.VideoCapture(str(source));raw_frames=[]
   while True:
    ok,im=cap.read()
    if not ok:break
    raw_frames.append(detection(im,profile,[0,1,2,3,5,7]))
   cap.release();path.write_text(json.dumps(raw_frames))
  cap=cv2.VideoCapture(str(source));ok,im=cap.read();cap.release();assert ok
  line=json.loads((ROOT/'configs'/config).read_text())['line'];observations,events=track(raw_frames,profile,5,im.shape[:2],line,'person' if tag=='meva' else 'vehicle')
  if tag=='meva':
   def region(rect):
    x1,y1,x2,y2=rect;return [(i,p) for i,rows in enumerate(observations) for p in rows if x1<(p[1]+p[3])/2<x2 and y1<p[4]<y2]
   roof=region((1160,805,1205,850));fixture=region((1480,690,1535,750));standing=region((1300,700,1430,820));counts=Counter(e['direction'] for e in events)
   results[tag]={'roof_false_boxes':len(roof),'fixture_false_boxes':len(fixture),'standing_frames':len({i for i,p in standing}),'standing_ids':len({p[0] for i,p in standing}),'counts':dict(counts)}
   results[tag]['passed']=not roof and not fixture and len({i for i,p in standing})>=45 and len({p[0] for i,p in standing})==1 and counts==Counter(entry=3,exit=1)
  else:
   motorcycles=[e for e in events if e['coco_class']==3 and e['direction']=='exit'];van=[(i,p) for i,rows in enumerate(observations) for p in rows if p[6] in [2,7] and 570<(p[1]+p[3])/2<720 and 400<p[4]<450]
   results[tag]={'motorcycle_exits':len(motorcycles),'van_first_frame':min((i for i,p in van),default=None)};results[tag]['passed']=len(motorcycles)==2 and results[tag]['van_first_frame']==0
 return {'passed':all(a['passed'] for a in results.values()),'checks':results}

def render(record,path):
 seq=record['sequence'];spec=PLAN['sequences'][seq];fps=record['fps'];gt,regions=labels(seq);width,height=spec['width'],spec['height'];size=(1280,720) if width>1280 else (width,height);scale=size[0]/width
 writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
 if not writer.isOpened():raise RuntimeError('Video writer unavailable')
 counts=Counter();refs=Counter();events=record['crossings']['model_events'];refevents=record['crossings']['reference_events']
 for frame,index in enumerate(record['source_frame_indices']):
  im=image(seq,index);truth=[g for g in gt[index] if g[5] and g[6]==1];pred=eligible_predictions(gt[index],record['observations'][frame],spec['kind'],regions);ious=box_iou([g[1:5] for g in truth],[p[1:5] for p in pred]);matched_gt=set();matched_pred=set()
  if len(truth) and pred:
   scores=ious.copy();scores[scores<.5]=0;x,y=linear_sum_assignment(-scores);pairs=[(int(i),int(j)) for i,j in zip(x,y) if scores[i,j]>0];matched_gt={i for i,j in pairs};matched_pred={j for i,j in pairs}
  for i,g in enumerate(truth):
   if i in matched_gt:continue
   x1,y1,x2,y2=map(int,g[1:5]);cv2.rectangle(im,(x1,y1),(x2,y2),(0,185,255),2)
  for j,p in enumerate(pred):
   x1,y1,x2,y2=map(int,p[1:5]);color=(40,220,40) if j in matched_pred else (40,40,240);cv2.rectangle(im,(x1,y1),(x2,y2),color,2);cv2.putText(im,f'#{int(p[0])}',(x1,max(20,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
  for x1,y1,x2,y2 in regions:cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(150,150,150),1)
  points=[tuple(round(p*(s-1)) for p,s in zip(point,(width,height))) for point in spec['line']];cv2.line(im,*points,(0,255,255),2)
  for e in events:
   if e['frame']==frame:counts[e['direction']]+=1
  for e in refevents:
   if e['frame']==frame and e['reliable']:refs[e['direction']]+=1
  im=cv2.resize(im,size);cv2.rectangle(im,(0,0),(size[0],48),(20,20,20),-1)
  cv2.putText(im,f'{seq} {fps} FPS | model IN {counts["entry"]} OUT {counts["exit"]} | label-derived IN {refs["entry"]} OUT {refs["exit"]}',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
  cv2.putText(im,'Green: matched track | red: unmatched prediction | amber: missed label | gray: ignored region',(8,39),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);writer.write(im)
 writer.release();cap=cv2.VideoCapture(str(path));n=0
 while cap.read()[0]:n+=1
 cap.release()
 if n!=record['frames_processed']:raise RuntimeError('Incomplete review video')
 record['video']=path.name;record['video_decoded_frames']=n


def gallery(report):
 cards=[];rows=[]
 for roundkey,round_ in report.get('rounds',{}).items():
  winner=round_.get('selected')
  for name,candidate in round_.get('candidates',{}).items():
   for seq,record in candidate['results'].items():
    m=record['metrics'];c=record['crossings'];rows.append(f'<tr><td>{roundkey}</td><td>{html.escape(name)}'+(' ✓' if name==winner else '')+f'</td><td>{seq}</td><td>{m["precision"]:.1f}%</td><td>{m["recall"]:.1f}%</td><td>{m["HOTA"]:.1f}</td><td>{m["IDF1"]:.1f}</td><td>{m["false_detections"]}</td><td>{m["missed_labeled_boxes"]}</td><td>{m["ID_switches"]}</td><td>{m["fragmentations"]}</td><td>{c["matched_events"]}/{sum(c["reference_confirmed"].values())}</td><td>{c["false_events"]}</td></tr>')
  if not winner:continue
  for seq,r in round_['candidates'][winner]['results'].items():
   if 'video' not in r:continue
   cap=cv2.VideoCapture(str(OUTPUT/r['video']));cap.set(cv2.CAP_PROP_POS_FRAMES,min(40,r['frames_processed']-1));ok,im=cap.read();cap.release();assert ok
   im=cv2.resize(im,(640,360));ok,b=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,80]);assert ok
   c=r['crossings'];cards.append(f'<article><h2>Round {roundkey} · {seq}</h2><p>{html.escape(PLAN["sequences"][seq]["title"])} · {html.escape(winner)}</p><img src="data:image/jpeg;base64,{base64.b64encode(b).decode()}" alt="Ground-truth comparison {seq}"><p>Label-derived confirmed crossings: {c["reference_confirmed"]["entry"]} in / {c["reference_confirmed"]["exit"]} out. Model: {c["model_reported"]["entry"]} in / {c["model_reported"]["exit"]} out.</p><a href="{r["video"]}">Watch labeled comparison video</a></article>')
 native=[]
 for seq,r in report.get('native_fps_check',{}).items():
  m=r['metrics'];native.append(f'<li>{seq} at {r["fps"]} FPS: HOTA {m["HOTA"]:.1f}, IDF1 {m["IDF1"]:.1f}, precision {m["precision"]:.1f}%, recall {m["recall"]:.1f}%; crossing matches {r["crossings"]["matched_events"]}/{sum(r["crossings"]["reference_confirmed"].values())}.</li>')
 doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Human-labeled tracker evaluation</title><style>body{font:15px system-ui;background:#eef3f6;color:#17303d;margin:0}main{max-width:1400px;margin:auto;padding:28px}h1{font-size:32px}p,li{line-height:1.6;color:#4a606c}.notice{background:#fff3cd;padding:16px;border-radius:8px}.table{overflow:auto}table{border-collapse:collapse;background:white;font-size:13px;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce4e8;white-space:nowrap}th{background:#163944;color:white}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}article{background:white;padding:18px;border-radius:12px;margin:10px 0}h2{font-size:20px}img{width:100%;border-radius:8px}a{color:#006c7b;font-weight:600}@media(max-width:700px){main{padding:14px}.cards{display:block}}</style><main><h1>Human-labeled tracker evaluation</h1><p>Three source-disjoint rounds: development, further validation and a reserved final holdout. Green boxes match human-labeled objects at IoU ≥0.5; red boxes are unmatched predictions; amber boxes are missed labels. Published ignore rules are applied. Precision/recall concern tracked output boxes, not raw low-confidence detector proposals.</p><p class="notice">'''+html.escape(report.get('status','Evaluation in progress.'))+'''</p><p>Round 1 and round 2 results are development evidence after tuning. Only round 3 is held out from local tuning. Original pretrained-model training overlap is unknown. Vehicle recordings are roads, not retail parking-lot proof. Reference crossings are computed from human trajectories and spot-checked; they are not manually labeled doorway events. Counts exclude ambiguous references as detailed in the shared report.</p><p><a href="report.json">Shared scores, configurations, provenance and observations</a> · <a href="../retail-tuning/review.html">Original reported failures</a> · <a href="../retail-security-test/review.html">Five new retail-area clips</a></p><div class="table"><table><thead><tr><th>Round</th><th>Version</th><th>Recording</th><th>Precision</th><th>Recall</th><th>HOTA</th><th>IDF1</th><th>False boxes</th><th>Missed boxes</th><th>ID switches</th><th>Fragments</th><th>Crossings matched/ref</th><th>False crossings</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div><h2>Native frame-rate check on the reserved recordings</h2><ul>'''+''.join(native)+'''</ul><div class="cards">'''+''.join(cards)+'''</div></main></html>'''
 (OUTPUT/'review.html').write_text(doc)


def candidates_round1():
 base=baseline();single=copy.deepcopy(base);single['name']='single-yolo11s';single['camera'].pop('person_detection',None);medium=copy.deepcopy(single);medium.update(name='single-yolo11m',model='yolo11m.pt');return [base,single,medium]
def candidates_round2(previous):
 current=copy.deepcopy(previous);current['name']='incumbent';strict=copy.deepcopy(current);strict['name']='association-0.8';strict['association']['match_thresh']=.8;fuse=copy.deepcopy(current);fuse['name']='fuse-confidence';fuse['association']['fuse_score']=True;return [current,strict,fuse]

def main():
 OUTPUT.mkdir(exist_ok=True)
 report=read_report(OUTPUT)
 frozen_files=['tracker.py','counting.py','detection.py','evaluation.py','experiments/benchmark.py','configs/retail-security.json','configs/bytetrack-low-fps.yaml','configs/human-labeled-evaluation.json','yolo11s.pt','yolo11n.pt','yolo11m.pt','data/human-labeled/manifest.json']
 if 'protocol' not in report:
  report['protocol']={'started_at_utc':stamp(),'plan':PLAN,'frozen_sha256':{name:sha(ROOT/name) for name in frozen_files},'source_manifest':json.loads((DATA/'manifest.json').read_text()),'metric_implementation':'Official TrackEval commit 12c8791b303e0a0b50f753af204249e622d0281a; NumPy scalar alias compatibility only. MOT17 distractor/zero-marked preprocessing; custom UA-DETRAC grouped vehicle/ignore-region adapter.','older_local_footage':'All previous local webcam, demo and MEVA clips excluded from benchmark scoring; old MEVA/Kathy clips used only for regression guards.'};report['rounds']={};save_report(OUTPUT,report)
 def verify():
  for name,digest in report['protocol']['frozen_sha256'].items():
   if sha(ROOT/name)!=digest:raise RuntimeError(f'Frozen benchmark input changed: {name}')
 previous=None
 for stage in PLAN['rounds']:
  number=stage['round'];key=str(number);sequences=[stage['person'],stage['vehicle']];report=read_report(OUTPUT);verify()
  if report['rounds'].get(key,{}).get('complete'):
   previous=report['rounds'][key]['candidates'][report['rounds'][key]['selected']]['profile'];continue
  if number==1:profiles=candidates_round1()
  elif number==2:profiles=candidates_round2(previous)
  else:
   chosen=copy.deepcopy(previous);chosen['name']='frozen-final';profiles=[chosen,baseline()]
  report['rounds'].setdefault(key,{'split':stage['split'],'opened_at_utc':stamp(),'candidates':{},'profiles_frozen_before_outcomes':profiles});save_report(OUTPUT,report)
  if number==3:
   report['final_profile_frozen_at_utc']=report.get('final_profile_frozen_at_utc',stamp());report['final_profile']=profiles[0];save_report(OUTPUT,report)
  for profile in profiles:
   name=profile['name'];report=read_report(OUTPUT);candidate=report['rounds'][key]['candidates'].setdefault(name,{'profile':profile,'results':{}});save_report(OUTPUT,report)
   if 'regression_guard' not in candidate:
    candidate['regression_guard']=guard(profile);report['rounds'][key]['candidates'][name]=candidate;save_report(OUTPUT,report)
   for seq in sequences:
    if seq in candidate['results']:continue
    verify();record=evaluate(seq,profile,5);candidate['results'][seq]=record;report=read_report(OUTPUT);report['rounds'][key]['candidates'][name]=candidate;save_report(OUTPUT,report);gallery(report)
    print('SCORED',number,name,seq,json.dumps(record['metrics']), 'crossings',record['crossings']['matched_events'],record['crossings']['false_events'],record['crossings']['missed_events'],flush=True)
   candidate['mean_HOTA']=sum(r['metrics']['HOTA'] for r in candidate['results'].values())/2;candidate['mean_IDF1']=sum(r['metrics']['IDF1'] for r in candidate['results'].values())/2
   report=read_report(OUTPUT);report['rounds'][key]['candidates'][name]=candidate;save_report(OUTPUT,report)
  choices=report['rounds'][key]['candidates'];allowed=[name for name,a in choices.items() if a['regression_guard']['passed']]
  if not allowed:raise RuntimeError('All candidates failed known-case regression guards')
  if number==3:winner='frozen-final'
  else:
   winner=max(allowed,key=lambda name:(choices[name]['mean_HOTA'],choices[name]['mean_IDF1']));incumbent=profiles[0]['name']
   if choices[incumbent]['regression_guard']['passed'] and choices[winner]['mean_HOTA']-choices[incumbent]['mean_HOTA']<=.01:winner=incumbent
  report['rounds'][key]['selected']=winner;previous=choices[winner]['profile']
  for seq,record in choices[winner]['results'].items():
   path=OUTPUT/f'round-{number}-{record["kind"]}.mp4'
   if 'video' not in record:render(record,path)
  report['rounds'][key].update(complete=True,finished_at_utc=stamp(),selection_reason='Frozen final version; holdout outcomes never used for tuning.' if number==3 else 'Highest mean HOTA among candidates passing known-case regressions, with predeclared incumbent preference.');save_report(OUTPUT,report);gallery(report);print('ROUND COMPLETE',number,winner,flush=True)
 report=read_report(OUTPUT);profile=report['final_profile'];report.setdefault('native_fps_check',{});save_report(OUTPUT,report)
 for seq in [PLAN['rounds'][2]['person'],PLAN['rounds'][2]['vehicle']]:
  if seq in report['native_fps_check']:continue
  verify();record=evaluate(seq,profile,PLAN['sequences'][seq]['fps']);report=read_report(OUTPUT);report['native_fps_check'][seq]=record;save_report(OUTPUT,report);gallery(report);print('NATIVE SCORED',seq,json.dumps(record['metrics']),flush=True)
 report=read_report(OUTPUT);report['finished_at_utc']=stamp();report['status']='All three rounds and both native-FPS checks completed. Metrics are local human-label results under the stated ignore and crossing rules; this is not a production-readiness certification.';save_report(OUTPUT,report);gallery(report);print('BENCHMARK COMPLETE',flush=True)

if __name__=='__main__':main()
