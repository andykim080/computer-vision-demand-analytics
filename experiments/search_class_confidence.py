"""Class cutoff development search; all reference geometry is scoring-only."""
import base64
import copy
import csv
import hashlib
import json
import sys
import tempfile
from argparse import Namespace
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tracker import SeparateVehicleTracker,run,load_config
from evaluation import box_iou
from counting import LineCounter,confirmation_frames
from experiments.compare_vehicle_confidence import CASES,assess

OUTPUT=ROOT/'outputs/class-confidence'
CACHE=Path('/private/tmp/propsight-vehicle-review/cache.json')
# Approximate agent-reviewed visible boxes. The forecourt scope is evaluation
# only; it is never passed to the detector, tracking or event counter.
REFERENCES={
 'seoul-songridan-gil':{'minimum_bottom_y':210,'frames':{
  '40':[['person',377,277,410,366],['person',397,268,428,364],
        ['person',427,269,458,363],['person',457,273,499,363],
        ['person',754,246,785,308],['person',870,244,901,335],
        ['person',907,258,942,341],['person',1034,220,1061,291],
        ['person',1095,232,1134,301]],
  '80':[['person',349,210,377,273],['person',558,214,586,264],
        ['person',752,285,812,382],['person',844,271,882,348],
        ['person',913,271,958,369],['person',1064,234,1094,304],
        ['person',1088,235,1116,306],['person',1200,338,1280,591]]}},
 'taiwan-linkou':{'minimum_bottom_y':400,'frames':{
  '40':[['car',222,373,452,534],['motorcycle',435,378,520,479],
        ['motorcycle',187,371,254,465],['motorcycle',893,567,1027,720],
        ['motorcycle',1009,482,1174,720],['motorcycle',1201,574,1280,720]],
  '80':[['car',222,373,452,534],['motorcycle',435,378,520,479],
        ['motorcycle',187,371,254,465],['motorcycle',893,567,1027,720],
        ['motorcycle',1009,482,1174,720],['motorcycle',1201,574,1280,720],
        ['motorcycle',306,591,613,720],['motorcycle',1082,356,1178,470],
        ['person',374,535,529,720],['person',1087,326,1146,444],
        ['person',1119,316,1154,418]]}}
}


def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(r):(OUTPUT/'report.json').write_text(json.dumps(r,indent=2)+'\n')


def score(site,observations):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    tp=fp=fn=0;frames={}
    refs=REFERENCES[site]
    for key,truth in refs['frames'].items():
        predicted=[p for p in observations if p['frame']==int(key) and p['y2']>=refs['minimum_bottom_y']]
        matrix=box_iou([b[1:] for b in truth],[[p[k] for k in ['x1','y1','x2','y2']] for p in predicted])
        for i,g in enumerate(truth):
            for j,p in enumerate(predicted):
                if p['class']!=g[0] or matrix[i,j]<.5:matrix[i,j]=0
        ii,jj=linear_sum_assignment(-matrix)
        matches=sum(matrix[i,j]>0 for i,j in zip(ii,jj))
        counts={'tp':int(matches),'fp':len(predicted)-int(matches),'fn':len(truth)-int(matches)}
        frames[key]=counts;tp+=counts['tp'];fp+=counts['fp'];fn+=counts['fn']
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0,
            'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,'frames':frames}


def replay(site,info,camera,thresholds,names):
    import numpy as np
    from ultralytics.engine.results import Results
    image=np.zeros((720,1280,3),np.uint8)
    tracker=SeparateVehicleTracker(ROOT/'configs/bytetrack-low-fps.yaml',camera['vehicle_tracking'],
        camera.get('person_tracking'),person_suppression=camera.get('person_suppression'),class_confidence=thresholds)
    points=[tuple(round(v*s) for v,s in zip(p,(1279,719))) for p in camera['line']]
    common=dict(entry_side=camera['entry_side'],deadband=camera['deadband_pixels'],
                confirm_frames=confirmation_frames(camera,5),max_gap_frames=round(5*camera['max_gap_seconds']))
    people=LineCounter(*points,**common)
    vehicle=LineCounter(*points,**{**common,'deadband':camera['vehicle_tracking'].get('deadband_pixels',common['deadband']),
                    'initial_confirm_frames':camera['vehicle_tracking'].get('initial_confirm_frames',common['confirm_frames'])})
    observations=[];events=[]
    for n,raw in enumerate(info['rows']):
        people.prune(n);vehicle.prune(n)
        result=Results(image,path='',names=names,boxes=np.asarray(raw,np.float32).reshape(-1,6))
        for bounds,identity,label,conf,group in tracker.update(result,image,names,.1,.7):
            row={'frame':n,'track_id':identity,'class':label,'confidence':conf,
                 **dict(zip(['x1','y1','x2','y2'],bounds))}
            observations.append(row)
            direction=(vehicle if group=='vehicle' else people).update(identity,((bounds[0]+bounds[2])/2,bounds[3]),n)
            if direction:events.append({'frame':n,'class':label,'direction':direction})
    return {'observations':observations,'events':events,'score':score(site,observations),
            'spot_checks':assess(CASES[site],{'observations':observations}),
            'class_boxes_filtered':dict(tracker.class_boxes_filtered)}


def totals(records):
    tp=sum(r['score']['tp'] for r in records.values());fp=sum(r['score']['fp'] for r in records.values());fn=sum(r['score']['fn'] for r in records.values())
    errors=sum(v['frames_with_vehicle_false_box'] for r in records.values() for v in r['spot_checks']['false_object_frames'].values())
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,'reported_false_object_box_frames':errors}


def search():
    import torch
    torch.set_num_threads(4)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'report.json').exists():raise RuntimeError('Preserve the completed search')
    cache=json.loads(CACHE.read_text());names={int(k):v for k,v in cache['names'].items()}
    old=json.loads((ROOT/'outputs/readme-location-replays/report.json').read_text())
    assert cache['model_sha256']==sha(ROOT/'yolo11s.pt')
    for info in cache['sequences'].values():assert info['source_sha256']==sha(ROOT/info['source'])
    candidates={'baseline':{},'global-055':{name:.55 for name in names.values()},
                'requested':{'car':.55,'truck':.55,'person':.1,'motorcycle':.1}}
    for car,truck,bus in product([.1,.4,.5,.55,.6],[.1,.4,.5,.55,.65],[.1,.55]):
        candidates[f'car-{car}-truck-{truck}-bus-{bus}']={'car':car,'truck':truck,'bus':bus,'person':.1,'motorcycle':.1}
    r={'registered_at_utc':stamp(),'status':'Development search on previously inspected clips; not independent accuracy',
       'source_cache':str(CACHE),'cache_sha256':sha(CACHE),'model_sha256':cache['model_sha256'],
       'references':REFERENCES,'reference_protocol':'Approximate source-image boxes reviewed by the agent in frames 40 and 80 of each clip. Score all output boxes whose bottom edge reaches y>=210 (Seoul) or y>=400 (Taiwan); all declared reference objects in those foreground areas are retained. Other image areas are unscored. Matching requires correct class and IoU>=0.5 with one-to-one assignment. Reference boxes and scoring areas are never used to filter predictions. No human-annotated track IDs or crossing labels were added.',
       'reference_limitations':['Only four foreground frames and one agent review; uncertain visible extents, partial occlusion and package-bearing motorcycles affect IoU.',
            'Same known clips, no independent held-out validation; scores concern frame boxes, not the earlier 23 crossing events.',
            'No real bus or truck reference target is present in this tiny sample; their recall/generalization is unmeasured.'],
       'selection':'Lexicographic maximum recall, then F1, then precision. Exact ties: fewer reported false-object box-frames over the full clips, then lower sum of changed class cutoffs, then candidate name.',
       'candidate_profiles':candidates,'settings':{'model':'yolo11s.pt','imgsz':1280,'base_confidence':.1,'iou':.7,'native_fps':5},
       'camera_profiles':{s:old['runs'][s]['summary']['config'] for s in CASES},'scores':{},'runs':{},
       'historical_report_sha256':{str(p):sha(ROOT/p) for p in [Path('outputs/vehicle-confidence/report.json'),Path('outputs/readme-location-replays/report.json')]}}
    write(r)
    for key,thresholds in candidates.items():
        records={site:replay(site,info,r['camera_profiles'][site],thresholds,names) for site,info in cache['sequences'].items()}
        r['scores'][key]=totals(records)
        if key in ['baseline','global-055','requested']:r['runs'][key]=records
        write(r)
        print('SCORE',key,r['scores'][key],flush=True)
    def rank(key):
        s=r['scores'][key]
        return s['recall'],s['f1'],s['precision'],-s['reported_false_object_box_frames'],-sum(r['candidate_profiles'][key].get(c,.1) for c in ['car','truck','bus'])
    selected=max(sorted(candidates),key=rank)
    r['selected']=selected;r['selected_profile']=candidates[selected]
    r['runs'][selected]={s:replay(s,info,r['camera_profiles'][s],candidates[selected],names) for s,info in cache['sequences'].items()}
    r['finished_at_utc']=stamp()
    for p,digest in r['historical_report_sha256'].items():assert sha(ROOT/p)==digest
    write(r)
    print('SELECTED',selected,r['scores'][selected],flush=True)


def fresh_and_render():
    import cv2
    import numpy as np
    r=json.loads((OUTPUT/'report.json').read_text());selected=r['selected'];cards=[]
    r['fresh_runs']={}
    with tempfile.TemporaryDirectory(prefix='propsight-class-cutoffs-',dir='/private/tmp') as temp:
        temp=Path(temp)
        for site,case in CASES.items():
            profile=copy.deepcopy(r['camera_profiles'][site]);profile['class_confidence']=r['selected_profile']
            config=ROOT/'configs'/('class-confidence-'+site+'.json');config.write_text(json.dumps(profile,indent=2)+'\n');load_config(config)
            _,record=run(Namespace(video=str(ROOT/case['source']),config=str(config),tracker=str(ROOT/'configs/bytetrack-low-fps.yaml'),
                model=str(ROOT/'yolo11s.pt'),classes=['person','car','truck','bus','motorcycle','bicycle'],
                output=str(temp),name=site,save_observations=True,preview=None,conf=.1,iou=.7,imgsz=1280,device='cpu',
                fps=None,max_frames=None,realtime=False,save_timing=False))
            with (temp/record.pop('observations_file')).open(newline='') as handle:observations=list(csv.DictReader(handle))
            for row in observations:
                row['frame']=int(row['frame'])
                for k in ['x1','y1','x2','y2','confidence','video_seconds']:row[k]=float(row[k])
            r['fresh_runs'][site]={'summary':record['summary'],'events':record['events'],'observations':observations,
                'score':score(site,observations),'spot_checks':assess(case,{'observations':observations})}
            # Re-render requested and selected from the same source frames.
            cap=cv2.VideoCapture(str(ROOT/case['source']));video=OUTPUT/(site+'.mp4')
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),5,(1536,470));assert writer.isOpened()
            groups=[r['runs']['requested'][site]['observations'],observations]
            by_frame=[{} for _ in groups]
            for side,rows in enumerate(groups):
                for row in rows:by_frame[side].setdefault(row['frame'],[]).append(row)
            n=0;preview=None;labels=[]
            while True:
                ok,source=cap.read()
                if not ok:break
                if str(n) in REFERENCES[site]['frames']:
                    im=source.copy()
                    for label,x1,y1,x2,y2 in REFERENCES[site]['frames'][str(n)]:
                        cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,255),2);cv2.putText(im,label,(x1,max(15,y1-3)),0,.45,(0,255,255),1)
                    cv2.line(im,(0,REFERENCES[site]['minimum_bottom_y']),(1279,REFERENCES[site]['minimum_bottom_y']),(255,255,0),1)
                    pixels=base64.b64encode(cv2.imencode('.jpg',cv2.resize(im,(960,540)))[1]).decode()
                    labels.append(f'<img alt="Evaluation-only reference boxes frame {n}" src="data:image/jpeg;base64,{pixels}">')
                panes=[]
                for side,title in enumerate(['Requested: car/truck .55; others .10','Selected: '+selected]):
                    im=source.copy()
                    for row in by_frame[side].get(n,[]):
                        color=(40,220,40) if row['class']=='person' else (255,150,30)
                        x1,y1,x2,y2=[round(row[k]) for k in ['x1','y1','x2','y2']]
                        cv2.rectangle(im,(x1,y1),(x2,y2),color,2);cv2.putText(im,row['class'],(x1,max(18,y1-4)),0,.55,color,1)
                    points=[tuple(round(v*s) for v,s in zip(p,(1279,719))) for p in profile['line']];cv2.line(im,*points,(0,255,255),2)
                    header=np.zeros((38,768,3),np.uint8);cv2.putText(header,title,(10,25),0,.48,(255,255,255),1)
                    panes.append(cv2.vconcat([header,cv2.resize(im,(768,432))]))
                combined=cv2.hconcat(panes);writer.write(combined)
                if n==40:preview=base64.b64encode(cv2.imencode('.jpg',combined)[1]).decode()
                n+=1
            cap.release();writer.release();cap=cv2.VideoCapture(str(video));decoded=0
            while cap.read()[0]:decoded+=1
            cap.release();assert decoded==n==90
            r['fresh_runs'][site]['comparison_video']=video.name;r['fresh_runs'][site]['decoded_comparison_frames']=decoded
            cards.append(f'<article><h2>{site}</h2><video controls preload="metadata" poster="data:image/jpeg;base64,{preview}" src="{video.name}"></video><details><summary>Inspect evaluation-only references (frames 40 and 80)</summary>{"".join(labels)}</details></article>')
            write(r)
    r['fresh_totals']=totals(r['fresh_runs']);r['fresh_finished_at_utc']=stamp();write(r)
    rows=[]
    for key in ['baseline','global-055','requested',selected]:
        a=r['fresh_totals'] if key==selected else r['scores'][key]
        rows.append(f'<tr><td>{key}</td><td>{a["recall"]:.1%}</td><td>{a["f1"]:.1%}</td><td>{a["precision"]:.1%}</td><td>{a["tp"]} / {a["fp"]} / {a["fn"]}</td><td>{a["reported_false_object_box_frames"]}</td></tr>')
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Class-specific confidence</title><style>body{margin:0;background:#eef3f5;color:#17303d;font:16px system-ui}main{max-width:1250px;margin:auto;padding:28px}p{line-height:1.6}article{padding:20px;background:white;border-radius:12px;margin:24px 0}video,img{width:100%;border-radius:8px}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccd8de}th{background:#173944;color:white}.notice{background:#fff1cb;padding:16px;border-radius:10px}a{color:#006d77}.scroll{overflow:auto}</style><main><h1>Class-specific confidence search</h1><p>No location exclusions or manual prediction relabeling. The requested split keeps people/motorcycles at 0.10, raises cars/trucks to 0.55 and leaves bus/bicycle unchanged. It preserves the checked scooters, but a lamp can switch to the unchanged bus class.</p><p class="notice">Development scores on four reviewed foreground frames from the same two known clips. Approximate agent-reviewed reference boxes; not a complete scene benchmark or independent validation. These are class-correct frame-box scores, not the earlier crossing precision/recall. No real bus or truck target was labeled, so their recall is unmeasured.</p><p>Ranking: recall first, F1 second, precision third. Exact ties use fewer reported fixture/cargo false-box frames over the full clips, then lower cutoffs. Fifty grid combinations plus baseline/global/requested controls are retained in the shared report. People and motorcycles remain at 0.10 throughout the grid.</p><div class="scroll"><table><tr><th>Version</th><th>Recall</th><th>F1</th><th>Precision</th><th>TP / FP / FN</th><th>Reported error box-frames</th></tr>'''+''.join(rows)+'''</table></div><p>Selected thresholds: '''+str(r['selected_profile'])+'''. The selected profile was rerun with fresh inference over both complete 18-second clips; other candidates replay identical cached detector proposals through ByteTrack. Both selected profiles are saved under configs/class-confidence-*.json. General application defaults remain unchanged.</p>'''+''.join(cards)+'''<p>Reference boxes and the horizontal blue scoring boundary shown under each reference image are evaluation-only. They are never used to filter detections. All displayed video detections cover the full scene. One-to-one scoring requires the correct class and IoU ≥ 0.5. Use these small development results to choose a next candidate, not claim property-wide accuracy.</p><p><a href="report.json">All candidates, references, settings, tracks and scores</a> · <a href="../vehicle-confidence/review.html">Previous global-confidence comparison</a> · <a href="../../docs/screenshots/README.md">Clip attribution</a></p></main></html>'''
    (OUTPUT/'review.html').write_text(doc)
    print('FRESH_TOTALS',r['fresh_totals'],flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['render']:fresh_and_render()
    else:search();fresh_and_render()
