"""Reproduce retail-camera development reruns and a separate five-location test."""
import argparse,base64,copy,csv,hashlib,html,json,os,sys,tempfile
from argparse import Namespace
from datetime import datetime,timezone
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tracker import run,read_report,save_report

CLASSES=['person','car','truck','bus','motorcycle','bicycle']
CASES={
 'abbey-road-pillar':('abbey-road',4,10,'Weak tiny vehicle proposal on the roadside pillar.'),
 'little-italy-walking-pair':('little-italy',2,56,'Dark walking pair entering from the right; check both early acquisition and later continuity.'),
 'little-italy-person-by-van':('little-italy',3,11,'Partly obscured person walking beside the parked van.'),
 'times-square-crowd':('times-square',5,40,'Visible people in the central group; hidden people cannot be reconstructed.')}
SITES={
 'songridan-gil':{'title':'Songridan-gil storefront crossing, Seoul','line':[[.45,.25],[.45,.95]],'fit':'Daytime, elevated fixed view of storefront approaches, people and a local access street.'},
 'gangnam-storefronts':{'title':'Gangnam convenience-store frontage, Seoul','line':[[.05,.55],[.60,.55]],'fit':'Daytime shop entrances, cones, pavement and nearby vehicles; close retail frontage.'},
 'linkou-old-street':{'title':'Linkou Old Street shopfront forecourt, Taiwan','line':[[.55,.25],[.55,.97]],'fit':'Daytime storefronts, parked cars, scooters and a shared approach area.'},
 'nashville-storefronts':{'title':'Lower Broadway storefront sidewalk, Nashville','line':[[.01,.62],[.25,.62]],'fit':'Night sidewalk and curbside vehicles; glare remains a source-quality limitation.'},
 'duval-storefront-walkway':{'title':'Duval Street entrance walkway, Key West','line':[[.50,.65],[.99,.65]],'fit':'Close fixed view of an entrance-side walkway; dense nighttime festival traffic is a stress case, not typical mall activity.','tiles':True}}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def observations(record):
 table=record['observations'];return [dict(zip(table['columns'],row)) for row in table['rows']]

def execute(source,config,output,name):
 with tempfile.TemporaryDirectory(prefix='propsight-retail-config-') as temp:
  cfg=Path(temp)/'camera.json';cfg.write_text(json.dumps(config))
  key,record=run(Namespace(video=str(source),config=str(cfg),tracker=str(ROOT/'configs/bytetrack-low-fps.yaml'),model=str(ROOT/'yolo11s.pt'),classes=CLASSES,output=str(output),name=name,save_observations=True,preview=None,conf=.1,iou=.7,imgsz=1280,device='cpu',fps=None,max_frames=None))
 csvpath=output/record.pop('observations_file')
 with csvpath.open(newline='') as handle:
  reader=csv.reader(handle);columns=next(reader);rows=list(reader)
 record['observations']={'columns':columns,'rows':rows}
 video=output/record['video'];cap=cv2.VideoCapture(str(video));decoded=0
 while cap.read()[0]:decoded+=1
 cap.release();expected=record['summary']['frames_processed']
 if decoded!=expected:raise RuntimeError(f'Incomplete annotated video: {name}')
 record['validation']={'decoded_frames':decoded,'source_sha256':sha(source),'complete_decode':True,'manual_accuracy':'Not fully labeled; counts are model estimates.'}
 report=read_report(output);report['runs'][key]=record;save_report(output,report);csvpath.unlink()
 return key,record

def make_gallery(output):
 report=read_report(output);cards=[]
 for key,r in report['runs'].items():
  f=r.get('preview_frame',40);pictures=[]
  baseline=r.get('baseline_video');source=Path(r['summary']['source']);views=[('Before',Path(baseline))] if baseline else [('Source',source)]
  views.append(('Adjusted tracker',output/r['video']))
  for label,path in views:
   cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();cap.release()
   if not ok:raise RuntimeError(f'Cannot preview {path}')
   im=cv2.resize(im,(600,round(im.shape[0]*600/im.shape[1])));ok,data=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,82]);assert ok
   pictures.append(f'<figure><figcaption>{label} · {f/5:.1f}s</figcaption><img src="data:image/jpeg;base64,{base64.b64encode(data).decode()}" alt="{html.escape(key)} {label}"></figure>')
  counts=r['summary']['counts'];ct=' · '.join(f'{k}: {v["entry"]} in / {v["exit"]} out' for k,v in counts.items() if v['entry'] or v['exit']) or 'No confirmed line crossings'
  links=f'<a href="{html.escape(os.path.relpath(source,output))}">Source video</a><a href="{r["video"]}">Annotated video</a>'
  if baseline:links+=f'<a href="{html.escape(os.path.relpath(baseline,output))}">Before video</a>'
  notes=r.get('review_note',r.get('scene_fit',''))
  if r.get('source_page'):links+=f'<a href="{html.escape(r["source_page"])}">Camera publisher</a>'
  cards.append(f'<article><h2>{html.escape(r.get("title",key))}</h2><div class="pair">{"".join(pictures)}</div><p>{html.escape(notes)}</p><p class="counts">Model counts · {html.escape(ct)}</p><p>{links}</p></article>')
 is_test=report.get('protocol',{}).get('split')=='test'
 title='Five retail-area camera tests' if is_test else 'Tracker adjustments: before and after'
 intro=('Five new physical locations, each with an 18-second clip. Earlier footage is excluded. These are storefront and walkway analogues; they are not recordings from HMart or Premium Outlets. Three daylight views and two night views. Lines show illustrative image-boundary crossings, not verified property entry/exit.' if is_test else 'The four reported failures are now development cases. Camera-specific changes are saved in the shared report. The older comparison videos remain available. Improvements here do not establish performance on new locations.')
 status=report.get('review_summary','Visual review pending. Model counts and IDs are not ground truth.')
 doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+title+'''</title><style>body{margin:0;background:#edf2f5;color:#142c38;font:16px system-ui}main{max-width:1260px;margin:auto;padding:32px}h1{font-size:34px;margin-bottom:10px}p{line-height:1.6;color:#48606e}article{background:#fff;border:1px solid #d7e0e6;border-radius:14px;padding:20px;margin:24px 0}h2{font-size:22px;margin:0 0 8px}.pair{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}figcaption{font-size:13px;margin:8px 0}img{width:100%;border-radius:8px}.counts{font-weight:600;color:#142c38}a{color:#006c7c;display:inline-block;margin:4px 22px 4px 0;font-weight:600}.notice{padding:16px;background:#fff6d7;border-radius:9px}@media(max-width:700px){main{padding:14px}.pair{display:block}}</style><main><h1>'''+title+'''</h1><p>'''+intro+'''</p><p class="notice">'''+html.escape(status)+'''</p><p><a href="report.json">Shared report and saved settings</a></p>'''+''.join(cards)+'''</main></html>'''
 (output/'review.html').write_text(doc)

def tuning():
 output=ROOT/'outputs/retail-tuning';profile=json.loads((ROOT/'configs/retail-security.json').read_text());oldsettings=json.loads((ROOT/'configs/five-site-test.json').read_text());report=read_report(output)
 report.setdefault('protocol',{'split':'development','model_training':False,'description':'Reported failures used to tune inference and camera-specific settings; not held-out accuracy results.'});save_report(output,report)
 for name,(site,number,preview,note) in CASES.items():
  if name in read_report(output)['runs']:continue
  source=next((ROOT/'data/test-five-sites'/site).glob(f'{number:02d}*.mp4'));config=copy.deepcopy(profile);config.update(camera_id=site,line=oldsettings['sites'][site]['line'])
  if site=='little-italy':config['person_tracking']={'new_track_thresh':.45}
  if site=='times-square':config['person_detection']['tiles']=True
  key,record=execute(source,config,output,name);record.update(title=name.replace('-',' ').title(),preview_frame=preview,review_note=note,baseline_video=str(ROOT/f'outputs/test-five-sites/{site}/{number:02d}.mp4'))
  report=read_report(output);report['runs'][key]=record;save_report(output,report);make_gallery(output)
 # Old standing people and motorcycle crossing cases stay outside the new test set.
 report=read_report(output)
 if 'regression_checks' not in report:
  checks={}
  for name,source,original in [('meva-standing-and-fixtures',Path('/Users/andykim/project/propsight/data/meva/demo-excerpts/forward-5fps-crf18.mp4'),'meva-person.json'),('kathy-van-and-motorcycles',ROOT/'data/earthcam-kathy-pub.mp4','kathy-pub.json')]:
   config=copy.deepcopy(profile);original_config=json.loads((ROOT/'configs'/original).read_text());config.update({k:v for k,v in original_config.items() if k!='vehicle_tracking'})
   with tempfile.TemporaryDirectory(prefix='propsight-retail-regression-') as temp:
    _,record=execute(source,config,Path(temp),name)
    record.pop('video');checks[name]=record
  report=read_report(output);report['regression_checks']=checks;save_report(output,report)
 make_gallery(output)

def testing():
 output=ROOT/'outputs/retail-security-test';dataset=ROOT/'data/retail-security-test';profile=json.loads((ROOT/'configs/retail-security.json').read_text());manifest=json.loads((dataset/'manifest.json').read_text());report=read_report(output)
 if set(manifest['sites'])!=set(SITES):raise ValueError('Exactly the five approved camera views are required')
 configs={}
 for site,spec in SITES.items():
  config=copy.deepcopy(profile);config.update(camera_id=site,line=spec['line']);config['person_detection']['tiles']=spec.get('tiles',False);configs[site]=config
 frozen={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'tracker.py',ROOT/'detection.py',ROOT/'counting.py',ROOT/'configs/retail-security.json',ROOT/'configs/bytetrack-low-fps.yaml',ROOT/'yolo11s.pt',ROOT/'yolo11n.pt']}
 oldclips=list((ROOT/'data/test-five-sites').glob('*/*.mp4'))+list((ROOT/'data').glob('*.mp4'))+list((ROOT/'data/kathy-pub-five-tests').glob('*.mp4'))
 excluded={str(p.relative_to(ROOT)):sha(p) for p in oldclips}
 report.setdefault('protocol',{'split':'test','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'configs':configs,'frozen_file_sha256':frozen,'older_local_footage_excluded':excluded,'source_manifest':manifest,'scope':'Five new locations, one 18-second clip per location. No tuning on these test outcomes. Not different times of day at one location. Pretrained model training sources are not audited.','model_training':False});save_report(output,report)
 for site,spec in SITES.items():
  if site in read_report(output)['runs']:continue
  for name,digest in report['protocol']['frozen_file_sha256'].items():
   if sha(ROOT/name)!=digest:raise RuntimeError(f'Frozen input changed: {name}')
  source=dataset/manifest['sites'][site]['file'];digest=sha(source)
  if digest!=manifest['sites'][site]['sha256'] or digest in excluded.values():raise ValueError('Test source hash failed')
  key,record=execute(source,configs[site],output,site);record.update(title=spec['title'],scene_fit=spec['fit'],source_page=manifest['sites'][site]['source_page'],preview_frame=40)
  report=read_report(output);report['runs'][key]=record;save_report(output,report);make_gallery(output)
 report=read_report(output);report['protocol']['finished_at_utc']=datetime.now(timezone.utc).isoformat();save_report(output,report);make_gallery(output)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['tuning','test','gallery'],required=True);parser.add_argument('--output',type=Path);args=parser.parse_args()
 if args.phase=='gallery':make_gallery(args.output)
 elif args.phase=='tuning':tuning()
 else:testing()
