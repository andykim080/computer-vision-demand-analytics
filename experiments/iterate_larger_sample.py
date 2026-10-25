"""Three native-FPS development rounds and a deferred, historical-data check."""
import argparse
import base64
import copy
import hashlib
import json
import sys
import time
from datetime import datetime,timezone
from itertools import product
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import benchmark as source
from experiments.compare_vehicle_confidence import CASES,assess
from experiments.measure_restored import combine
from evaluation import tracking_metrics
from tracker import SeparateVehicleTracker

OUTPUT=ROOT/'outputs/larger-confidence'
TEMP=Path('/private/tmp/propsight-larger-confidence')
DEV=['MOT17-02','MVI_20011','MOT17-04','MVI_40152']
FINAL=['MOT17-09','MVI_63561']
NAMES={0:'person',1:'bicycle',2:'car',3:'motorcycle',5:'bus',7:'truck'}
INVERSE={v:k for k,v in NAMES.items()}
LABELS={}
INITIAL={'class_confidence':{'car':.5,'truck':.4,'bus':.55,'person':.1,'motorcycle':.1},
         'person_new':.55,'vehicle_new':.2,'match_thresh':.9,'buffer_seconds':None,'fuse_score':False}


def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(r):source.save_report(OUTPUT,r)
def report():return json.loads((OUTPUT/'report.json').read_text())
def key(profile):return json.dumps(profile,sort_keys=True)


def raw(seq,allow_final=False):
    import numpy as np
    import torch
    from ultralytics import YOLO
    if seq in FINAL and not allow_final:raise RuntimeError('Deferred source inference is locked')
    spec=source.PLAN['sequences'][seq];count=round(8*spec['fps']);TEMP.mkdir(parents=True,exist_ok=True)
    path=TEMP/(seq+'-native-8s.json')
    fingerprint={'model':sha(ROOT/'yolo11s.pt'),'archive':sha(source.DATA/spec['archive']),
                 'sequence':seq,'frames':count,'imgsz':1280,'conf':.1,'iou':.7,'device':'cpu',
                 'classes':[0] if spec['kind']=='person' else [2,5,7]}
    if path.exists():
        cached=json.loads(path.read_text())
        if cached['fingerprint']!=fingerprint:raise RuntimeError('Detector cache fingerprint mismatch')
        return cached['rows']
    detector=YOLO(str(ROOT/'yolo11s.pt'));torch.set_num_threads(4);rows=[];start=time.monotonic()
    for first in range(0,count,4):
        images=[source.image(seq,n) for n in range(first,min(count,first+4))]
        predictions=detector.predict(images,classes=fingerprint['classes'],conf=.1,iou=.7,
                                     imgsz=1280,device='cpu',verbose=False)
        rows.extend(p.boxes.data.cpu().numpy().tolist() for p in predictions)
        if len(rows)%40==0 or len(rows)==count:
            print('INFERENCE',seq,len(rows),'/',count,round(time.monotonic()-start,1),'seconds',flush=True)
    path.write_text(json.dumps({'fingerprint':fingerprint,'rows':rows,'completed_at_utc':stamp()}))
    return rows


def tracker(profile,fps):
    t=SeparateVehicleTracker(ROOT/'configs/bytetrack-low-fps.yaml',
        {'track_high_thresh':profile['vehicle_new'],'new_track_thresh':profile['vehicle_new'],'nms_iou':.7},
        {'track_high_thresh':.25,'new_track_thresh':profile['person_new']},
        frame_rate=fps,track_buffer_seconds=profile['buffer_seconds'],class_confidence=profile['class_confidence'])
    for group in t.trackers:
        for field in ['match_thresh','fuse_score']:
            t.settings[group][field]=profile[field]
            setattr(t.trackers[group].args,field,profile[field])
    return t


def track(rows,profile,fps,height,width,kind=None):
    import numpy as np
    from ultralytics.engine.results import Results
    image=np.zeros((height,width,3),np.uint8);t=tracker(profile,fps);predictions=[];observations=[]
    for n,boxes in enumerate(rows):
        result=Results(image,path='',names=NAMES,boxes=np.asarray(boxes,np.float32).reshape(-1,6))
        predicted=[];observed=[]
        for bounds,identity,label,confidence,group in t.update(result,image,NAMES,.1,.7):
            if kind and group!=kind:continue
            predicted.append([int(identity.split('-')[-1]),*bounds,confidence,INVERSE[label]])
            observed.append({'frame':n,'track_id':identity,'class':label,'confidence':confidence,
                             **dict(zip(['x1','y1','x2','y2'],bounds))})
        predictions.append(predicted);observations.extend(observed)
    return predictions,observations


def evaluate(seq,rows,profile,seconds,retain=False):
    spec=source.PLAN['sequences'][seq];n=round(seconds*spec['fps'])
    if seq not in LABELS:LABELS[seq]=source.labels(seq)
    truth,regions=LABELS[seq]
    pred,observations=track(rows[:n],profile,spec['fps'],spec['height'],spec['width'],spec['kind'])
    metrics,details=tracking_metrics(truth[:n],pred,spec['kind'],regions)
    rec={'sequence':seq,'kind':spec['kind'],'frames':n,'seconds':seconds,'fps':spec['fps'],
         'metrics':metrics,'raw':details}
    if retain:rec['observations']=observations
    return rec


def aggregate(records):
    def pooled(items):
        tp=sum(x['metrics']['matched_labeled_boxes'] for x in items)
        fp=sum(x['metrics']['false_detections'] for x in items)
        fn=sum(x['metrics']['missed_labeled_boxes'] for x in items)
        return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0,
                'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
    return {**pooled(list(records.values())),
            'by_group':{group:pooled([x for x in records.values() if x['kind']==group]) for group in ['person','vehicle']}}


def guard(profile):
    cached=json.loads(Path('/private/tmp/propsight-vehicle-review/cache.json').read_text())
    result={};errors=0;controls=[]
    for site,info in cached['sequences'].items():
        _,observations=track(info['rows'],profile,5,720,1280)
        checked=assess(CASES[site],{'observations':observations});result[site]=checked
        errors+=sum(x['frames_with_vehicle_false_box'] for x in checked['false_object_frames'].values())
        wanted=['passing-white-car'] if site=='seoul-songridan-gil' else ['parked-white-car','scooter-beside-white-car','foreground-left-scooter']
        controls.extend(checked['real_object_spot_checks'][name]['correct_class_and_box'] for name in wanted)
    return {'passed':errors==0 and all(controls),'reported_error_box_frames':errors,
            'real_controls_passed':sum(controls),'real_controls_total':len(controls),'details':result}


def candidates(round_number,incumbent):
    choices={'incumbent':copy.deepcopy(incumbent)}
    if round_number==1:
        for car,truck,bus in product([.1,.3,.5,.55],[.1,.3,.4],[.1,.55]):
            p=copy.deepcopy(incumbent);p['class_confidence'].update(car=car,truck=truck,bus=bus)
            choices[f'car-{car}-truck-{truck}-bus-{bus}']=p
    elif round_number==2:
        for person,vehicle in product([.25,.35,.45,.55],[.1,.15,.2]):
            p=copy.deepcopy(incumbent);p.update(person_new=person,vehicle_new=vehicle)
            choices[f'person-new-{person}-vehicle-new-{vehicle}']=p
    else:
        for match,buffer in product([.7,.8,.9],[.6,1.2,2.4]):
            p=copy.deepcopy(incumbent);p.update(match_thresh=match,buffer_seconds=buffer)
            choices[f'match-{match}-buffer-seconds-{buffer}']=p
        p=copy.deepcopy(incumbent);p['fuse_score']=True;choices['fuse-confidence']=p
    unique={};seen=set()
    for name,p in choices.items():
        if key(p) not in seen:unique[name]=p;seen.add(key(p))
    return unique


def initialize():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'report.json').exists():return
    r={'format_version':1,'started_at_utc':stamp(),'runs':{},'iterations':{},'initial_profile':INITIAL,
       'protocol':{'model':'yolo11s.pt','imgsz':1280,'base_confidence':.1,'iou':.7,'device':'cpu',
          'sources':source.PLAN['sequences'],'development':DEV,'deferred_check':FINAL,'native_frame_rates':True,
          'rounds':{'1':{'sequences':DEV[:2],'seconds_each':4,'adjust':'class-specific detector cutoffs'},
                    '2':{'sequences':DEV,'seconds_each':4,'adjust':'track initiation thresholds'},
                    '3':{'sequences':DEV,'seconds_each':8,'adjust':'association match threshold, retention and confidence fusion'}},
          'selection':'Within versions passing the predeclared old error/scooter guard, maximize pooled labeled-box recall, then F1, then precision. Exact ties prefer incumbent, then fewer changed profile fields. Also disclose the unrestricted recall-first winner and its guard result.',
          'guard':'Zero matched car/truck/bus boxes on the five earlier reported fixtures/cargo objects across the original two 90-frame webcam clips; retain the passing Seoul car, parked Taiwan car and both checked Taiwan scooters.',
          'metrics':'Official pinned TrackEval: IoU 0.5 CLEAR box matches, MOT17 distractor/zero-marked preprocessing and published UA-DETRAC ignored-region handling. Pedestrians are person; UA-DETRAC car/bus/van/others are one vehicle group. This does not measure car-versus-truck type accuracy or motorcycle recall.',
          'scope':'First eight seconds per original source, all 25/30 FPS frames retained. Round 1 uses two four-second prefixes, round 2 adds two sources, round 3 extends all four to eight seconds. Final two sources are opened for current-profile evaluation only after freeze.',
          'history':'All six parent recordings appeared in earlier project benchmarks. Deferred means not used for current tuning, not completely unseen historical or pretrained data.',
          'limitations':['Short segments and repeated adjacent-frame object appearances are not unique people/vehicles.',
             'These box/tracking scores are not the earlier property doorway-crossing metrics.',
             'Road vehicles and pedestrian benchmarks do not validate property parking, new-camera generalization, motorcycle accuracy or hourly counts.',
             'Known error guards constrain selection; unrestricted recall-first winners are disclosed separately.']},
       'source_archive_sha256':{name:sha(source.DATA/name) for name in ['mot17-selected.zip','detrac-selected.zip']},
       'history_sha256':{p:sha(ROOT/p) for p in ['outputs/class-confidence/report.json','outputs/vehicle-confidence/report.json','docs/screenshots/seoul-songridan-gil-after.jpg','docs/screenshots/taiwan-linkou-after.jpg']}}
    write(r)


def develop():
    initialize();r=report()
    if r.get('deferred_opened_at_utc'):raise RuntimeError('Cannot tune after deferred outcomes are opened')
    incumbent=copy.deepcopy(r['initial_profile']);guard_cache={};evaluation_cache={}
    for number in [1,2,3]:
        spec=r['protocol']['rounds'][str(number)]
        if str(number) in r['iterations'] and r['iterations'][str(number)].get('complete'):
            incumbent=r['iterations'][str(number)]['selected_profile'];continue
        data={seq:raw(seq) for seq in spec['sequences']}
        configs=candidates(number,incumbent)
        iteration={'started_at_utc':stamp(),'sample':spec,'candidates':configs,'results':{}}
        r['iterations'][str(number)]=iteration;write(r)
        for name,p in configs.items():
            if key(p) not in guard_cache:guard_cache[key(p)]=guard(p)
            records={}
            for seq,rows in data.items():
                cache_key=(seq,spec['seconds_each'],key(p))
                if cache_key not in evaluation_cache:evaluation_cache[cache_key]=evaluate(seq,rows,p,spec['seconds_each'])
                records[seq]=evaluation_cache[cache_key]
            iteration['results'][name]={'totals':aggregate(records),'guard':guard_cache[key(p)],'sequences':records}
            write(r)
            print('ROUND',number,name,iteration['results'][name]['totals'],'guard',guard_cache[key(p)]['passed'],flush=True)
        def rank(name):
            s=iteration['results'][name]['totals'];p=configs[name]
            return s['recall'],s['f1'],s['precision'],name=='incumbent',-sum(p[k]!=incumbent[k] for k in p)
        unrestricted=max(sorted(configs),key=rank)
        valid=[name for name in configs if iteration['results'][name]['guard']['passed']]
        if not valid:raise RuntimeError('No version passes the previously working-case guard')
        selected=max(sorted(valid),key=rank)
        iteration.update(selected=selected,selected_profile=configs[selected],unrestricted_winner=unrestricted,
                         complete=True,finished_at_utc=stamp())
        incumbent=copy.deepcopy(configs[selected]);write(r)
        print('ROUND_SELECTED',number,selected,'UNRESTRICTED',unrestricted,flush=True)
    r['selected_profile']=incumbent;r['frozen_at_utc']=stamp()
    r['frozen_sha256']={p:sha(ROOT/p) for p in ['tracker.py','detection.py','evaluation.py','experiments/iterate_larger_sample.py','yolo11s.pt','configs/bytetrack-low-fps.yaml']}
    write(r)
    print('FROZEN',incumbent,flush=True)


def final():
    r=report()
    if not r.get('frozen_at_utc'):raise RuntimeError('Freeze development first')
    for p,digest in r['frozen_sha256'].items():
        if sha(ROOT/p)!=digest:raise RuntimeError('Frozen input changed: '+p)
    r['deferred_opened_at_utc']=r.get('deferred_opened_at_utc',stamp());write(r)
    for seq in DEV+FINAL:
        rows=raw(seq,allow_final=seq in FINAL)
        for mode,p in [('initial',r['initial_profile']),('selected',r['selected_profile'])]:
            r['runs'].setdefault(mode,{})
            if seq not in r['runs'][mode]:
                r['runs'][mode][seq]=evaluate(seq,rows,p,8,retain=True);write(r)
                print('FINAL_SCORED',seq,mode,r['runs'][mode][seq]['metrics'],flush=True)
    r['totals']={scope:{mode:aggregate({s:r['runs'][mode][s] for s in sequences}) for mode in ['initial','selected']}
                 for scope,sequences in [('development',DEV),('deferred',FINAL),('all',DEV+FINAL)]}
    r['tracking_totals']={mode:combine({s:r['runs'][mode][s] for s in FINAL}) for mode in ['initial','selected']}
    r['final_guard']=guard(r['selected_profile'])
    r['unique_sample_frames']=sum(x['frames'] for x in r['runs']['selected'].values())
    r['unique_sample_labeled_box_appearances']=sum(x['metrics']['labeled_boxes'] for x in r['runs']['selected'].values())
    for p,digest in r['history_sha256'].items():assert sha(ROOT/p)==digest,p
    r['finished_at_utc']=stamp();write(r)
    (ROOT/'configs/larger-confidence-candidate.json').write_text(json.dumps(r['selected_profile'],indent=2)+'\n')
    print('TOTALS',json.dumps(r['totals']),flush=True)


def render():
    import cv2
    import numpy as np
    r=report();cards=[]
    for seq in DEV+FINAL:
        spec=source.PLAN['sequences'][seq];before=r['runs']['initial'][seq];after=r['runs']['selected'][seq]
        target=OUTPUT/(seq.lower()+'.mp4');writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),spec['fps'],(1536,490))
        assert writer.isOpened();maps=[]
        for rec in [before,after]:
            mapping={}
            for row in rec['observations']:mapping.setdefault(row['frame'],[]).append(row)
            maps.append(mapping)
        thumb=None
        for n in range(after['frames']):
            image=source.image(seq,n);panes=[]
            for side,rec in enumerate([before,after]):
                im=image.copy();color=(255,150,30) if side==0 else (40,220,40)
                for row in maps[side].get(n,[]):
                    bounds=[round(row[k]) for k in ['x1','y1','x2','y2']]
                    cv2.rectangle(im,tuple(bounds[:2]),tuple(bounds[2:]),color,2)
                header=np.zeros((58,768,3),np.uint8);m=rec['metrics']
                cv2.putText(header,('Initial class-specific profile' if side==0 else 'After three tuning rounds'),(10,22),0,.58,color,1)
                cv2.putText(header,f'{seq} | {n/spec["fps"]:.2f}s | Box recall {m["recall"]:.1f}% / precision {m["precision"]:.1f}%',(10,46),0,.44,(255,255,255),1)
                panes.append(cv2.vconcat([header,cv2.resize(im,(768,432))]))
            combined=cv2.hconcat(panes);writer.write(combined)
            if n==after['frames']//2:thumb=base64.b64encode(cv2.imencode('.jpg',cv2.resize(combined,(960,306)))[1]).decode()
        writer.release();cap=cv2.VideoCapture(str(target));count=0
        while cap.read()[0]:count+=1
        cap.release();assert count==after['frames']
        after['export']={'comparison_video':target.name,'decoded_frames':count};write(r)
        a=after['metrics'];tag='Deferred check' if seq in FINAL else 'Development'
        cards.append(f'<article><h2>{seq} · {tag}</h2><p>{spec["title"]} · Eight seconds at native {spec["fps"]} FPS. After: recall {a["recall"]:.1f}%, precision {a["precision"]:.1f}%, HOTA {a["HOTA"]:.1f}.</p><video controls preload="none" poster="data:image/jpeg;base64,{thumb}" src="{target.name}"></video></article>')
        print('EXPORTED',seq,count,flush=True)
    iteration_rows=[]
    for number,iteration in r['iterations'].items():
        for mode,title in [('incumbent','Before'),(iteration['selected'],'Selected')]:
            a=iteration['results'][mode]['totals']
            iteration_rows.append(f'<tr><td>{number}</td><td>{title}: {mode}</td><td>{a["recall"]:.1%}</td><td>{a["f1"]:.1%}</td><td>{a["precision"]:.1%}</td></tr>')
    rows=[]
    for scope in ['development','deferred','all']:
        for mode in ['initial','selected']:
            a=r['totals'][scope][mode]
            rows.append(f'<tr><td>{scope}</td><td>{mode}</td><td>{a["recall"]:.1%}</td><td>{a["f1"]:.1%}</td><td>{a["precision"]:.1%}</td><td>{a["tp"]} / {a["fp"]} / {a["fn"]}</td></tr>')
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Larger sample — three tuning rounds</title><style>body{margin:0;background:#eef3f5;color:#17303d;font:16px system-ui}main{max-width:1250px;margin:auto;padding:28px}p{line-height:1.6}table{border-collapse:collapse;background:white;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #cbd8de}th{background:#173944;color:white}.notice{background:#fff1cb;padding:18px;border-radius:12px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px}article{background:white;border-radius:12px;padding:18px;margin:18px 0}video{width:100%;border-radius:8px}a{color:#006d77}.scroll{overflow:auto}@media(max-width:750px){.cards{display:block}}</style><main><h1>Larger sample — three tuning rounds</h1><p>Six publisher-labeled source recordings, eight seconds each, all 25/30 FPS frames retained. '''+str(r['unique_sample_frames'])+''' distinct source frames and '''+str(r['unique_sample_labeled_box_appearances'])+''' labeled object appearances (not unique people/vehicles). No training, location exclusions, prediction relabeling or frame skipping.</p><p class="notice">These source recordings were used in earlier project benchmarks. The last two were deferred from this run’s tuning, but this is a broader historical-data regression check, not a completely new independent test. Scores measure tracked person/vehicle boxes at IoU 0.5; they are not the earlier property-crossing metrics. Vehicle types are grouped because UA-DETRAC van/others labels do not map cleanly to YOLO car/truck classes. Motorcycle accuracy is checked only through the earlier small webcam controls.</p><h2>Adjustments after each development round</h2><p>Round 1: class cutoffs on two four-second prefixes. Round 2: add two source views and tune track initiation. Round 3: extend all four to eight seconds and tune association/retention. Rank recall, F1, then precision among profiles retaining the previously working error/scooter checks. Unrestricted recall-first winners and their failed/passed checks are disclosed in the shared report.</p><div class="scroll"><table><tr><th>Round</th><th>Version</th><th>Recall</th><th>F1</th><th>Precision</th></tr>'''+''.join(iteration_rows)+'''</table></div><h2>Frozen final comparison</h2><div class="scroll"><table><tr><th>Scope</th><th>Version</th><th>Recall</th><th>F1</th><th>Precision</th><th>TP / FP / FN</th></tr>'''+''.join(rows)+'''</table></div><p>Selected profile: '''+str(r['selected_profile'])+'''. General defaults remain unchanged. Complete settings, per-group metrics, HOTA/IDF1, all candidates and regression checks are in the report.</p><div class="cards">'''+''.join(cards)+'''</div><p><a href="report.json">One shared report</a> · <a href="../../configs/larger-confidence-candidate.json">Selected experimental profile</a> · <a href="../class-confidence/review.html">Earlier four-frame development sample</a></p><p>Sources: <a href="https://motchallenge.net/data/MOT17/">MOT17</a> and <a href="https://sites.google.com/view/daweidu/projects/ua-detrac">UA-DETRAC</a>. Published labels and ignore regions are evaluated with pinned TrackEval. The recordings contain pedestrian streets and roads, not operational property occupancy or hourly demand truth.</p></main></html>'''
    (OUTPUT/'review.html').write_text(doc)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['all','develop','final','render'],nargs='?',default='all');args=parser.parse_args()
    if args.phase in ['all','develop']:develop()
    if args.phase in ['all','final']:final()
    if args.phase in ['all','render']:render()
