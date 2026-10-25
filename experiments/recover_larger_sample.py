"""Repair an observed legacy-person regression without retuning on deferred scores."""
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
from experiments import iterate_larger_sample as previous
from evaluation import tracking_metrics
from counting import LineCounter,confirmation_frames

OUTPUT=ROOT/'outputs/larger-confidence-recovery'
MEVA=Path('/Users/andykim/project/propsight/data/meva/demo-excerpts/forward-5fps-crf18.mp4')


def stamp():return datetime.now(timezone.utc).isoformat()
def write(r):previous.source.save_report(OUTPUT,r)
def load():return json.loads((OUTPUT/'report.json').read_text())


def legacy(profile):
    import cv2
    digest=hashlib.sha256(json.dumps({'detector':'yolo11s.pt','imgsz':1280,'views':None,'source':previous.sha(MEVA)},sort_keys=True).encode()).hexdigest()[:16]
    cache=Path('/private/tmp/propsight-benchmark/raw-cache')/('guard-meva-'+digest+'.json')
    cap=cv2.VideoCapture(str(MEVA));ok,image=cap.read();cap.release();assert ok
    _,obs=previous.track(json.loads(cache.read_text()),profile,5,*image.shape[:2],kind='person')
    def region(bounds):
        x1,y1,x2,y2=bounds;return [a for a in obs if x1<(a['x1']+a['x2'])/2<x2 and y1<a['y2']<y2]
    roof=region((1160,805,1205,850));fixture=region((1480,690,1535,750));standing=region((1300,700,1430,820))
    result={'roof_false_boxes':len(roof),'fixture_false_boxes':len(fixture),
            'standing_frames':len({a['frame'] for a in standing}),'standing_ids':len({a['track_id'] for a in standing}),
            'source_sha256':previous.sha(MEVA),'cache_sha256':previous.sha(cache)}
    result['passed']=not roof and not fixture and result['standing_frames']>=45 and result['standing_ids']==1
    return result


def later_raw(seq):
    import torch
    from ultralytics import YOLO
    spec=previous.source.PLAN['sequences'][seq];start=round(8*spec['fps']);count=start
    path=previous.TEMP/(seq+'-native-8-to-16s.json')
    fingerprint={'model':previous.sha(ROOT/'yolo11s.pt'),'archive':previous.sha(previous.source.DATA/spec['archive']),
                 'start_frame':start,'frames':count,'imgsz':1280,'conf':.1,'iou':.7,'device':'cpu',
                 'classes':[0] if spec['kind']=='person' else [2,5,7]}
    if path.exists():
        cached=json.loads(path.read_text());assert cached['fingerprint']==fingerprint;return cached['rows']
    model=YOLO(str(ROOT/'yolo11s.pt'));torch.set_num_threads(4);rows=[];began=time.monotonic()
    for first in range(start,start+count,4):
        images=[previous.source.image(seq,n) for n in range(first,min(first+4,start+count))]
        pred=model.predict(images,classes=fingerprint['classes'],conf=.1,iou=.7,imgsz=1280,device='cpu',verbose=False)
        rows.extend(p.boxes.data.cpu().numpy().tolist() for p in pred)
        if len(rows)%40==0:print('LATER_INFERENCE',seq,len(rows),'/',count,round(time.monotonic()-began,1),flush=True)
    path.write_text(json.dumps({'fingerprint':fingerprint,'rows':rows,'finished_at_utc':stamp()}))
    return rows


def evaluate_later(seq,rows,profile):
    spec=previous.source.PLAN['sequences'][seq];first=round(8*spec['fps'])
    gt,regions=previous.source.labels(seq)
    pred,observations=previous.track(rows,profile,spec['fps'],spec['height'],spec['width'],spec['kind'])
    metrics,details=tracking_metrics(gt[first:first+len(rows)],pred,spec['kind'],regions)
    return {'sequence':seq,'kind':spec['kind'],'source_start_frame':first,'frames':len(rows),'fps':spec['fps'],
            'metrics':metrics,'raw':details,'observations':observations}


def develop():
    import torch
    torch.set_num_threads(4);OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'report.json').exists():raise RuntimeError('Preserve the completed correction')
    old=previous.report();base=copy.deepcopy(old['initial_profile']);candidates={'initial':base}
    for birth,buffer in product([.45,.475,.5,.525,.55],[None,1.2,2.4]):
        p=copy.deepcopy(base);p.update(person_new=birth,buffer_seconds=buffer)
        if p!=base:candidates[f'person-new-{birth}-buffer-{buffer}']=p
    r={'format_version':1,'runs':{},'registered_at_utc':stamp(),'status':'Fourth development round after a discovered legacy regression',
       'initial_profile':base,'candidates':candidates,'results':{},
       'prior_report_sha256':previous.sha(previous.OUTPUT/'report.json'),
       'prior_frozen_profile':old['selected_profile'],'prior_deferred_results':old['totals']['deferred'],
       'prior_legacy_check':legacy(old['selected_profile']),
       'protocol':{'development':previous.DEV,'development_seconds_each':8,
            'later_check':previous.FINAL,'later_source_seconds':[8,16],
            'selection':'Development labeled-box recall, then F1, then precision, restricted to versions passing both the existing webcam vehicle/scooter guard and the newly restored legacy-person guard. Exact ties prefer initial, then fewer changed fields.',
            'new_guard':'Legacy MEVA roof/fixture false person boxes must remain zero; the standing person must remain visible in >=45 frames under one ID. Reference regions are evaluation-only.',
            'why':'Earlier recall-first person initiation 0.25 resurrected old false person boxes. Preserve that failed result; restore the guard and select only using the four development sources plus old regression clips.',
            'later_check_limitations':'Later time windows of the same historically used cameras; correlated actors and viewpoints. Unopened in this experiment until the corrected profile is frozen, but not independent new-camera validation.',
            'metric':'Same pinned TrackEval grouped person/vehicle labeled-box metrics as the prior larger run. No property crossing or vehicle-type accuracy claim.'}}
    write(r);data={s:previous.raw(s) for s in previous.DEV}
    for name,p in candidates.items():
        old_guard=previous.guard(p);person=legacy(p)
        # Avoid evaluating known-failing proposals; the rejection is retained.
        result={'webcam_guard':old_guard,'legacy_person_guard':person,'passed':old_guard['passed'] and person['passed']}
        if result['passed']:
            records={s:previous.evaluate(s,rows,p,8) for s,rows in data.items()}
            result['sequences']=records;result['totals']=previous.aggregate(records)
        r['results'][name]=result;write(r)
        print('RECOVERY',name,'PASS',result['passed'],result.get('totals',{}),'LEGACY',person['roof_false_boxes'],person['fixture_false_boxes'],flush=True)
    def rank(name):
        a=r['results'][name]['totals'];p=candidates[name]
        return a['recall'],a['f1'],a['precision'],name=='initial',-sum(p[k]!=base[k] for k in p)
    valid=[name for name in candidates if r['results'][name]['passed']]
    assert valid
    selected=max(sorted(valid),key=rank);r.update(selected=selected,selected_profile=candidates[selected],frozen_at_utc=stamp())
    r['frozen_sha256']={p:previous.sha(ROOT/p) for p in ['tracker.py','detection.py','evaluation.py','experiments/recover_larger_sample.py','experiments/iterate_larger_sample.py','yolo11s.pt','configs/bytetrack-low-fps.yaml']}
    write(r);print('RECOVERY_FROZEN',selected,r['selected_profile'],flush=True)


def final():
    r=load()
    for p,digest in r['frozen_sha256'].items():assert previous.sha(ROOT/p)==digest,p
    r['later_opened_at_utc']=stamp();write(r)
    for seq in previous.FINAL:
        rows=later_raw(seq)
        for name,profile in [('initial',r['initial_profile']),('selected',r['selected_profile'])]:
            r['runs'].setdefault(name,{})[seq]=evaluate_later(seq,rows,profile);write(r)
    r['later_totals']={name:previous.aggregate(records) for name,records in r['runs'].items()}
    r['later_tracking']={name:previous.combine(records) for name,records in r['runs'].items()}
    r['finished_at_utc']=stamp();assert previous.sha(previous.OUTPUT/'report.json')==r['prior_report_sha256'];write(r)
    (ROOT/'configs/larger-confidence-recovered-profile.json').write_text(json.dumps(r['selected_profile'],indent=2)+'\n')
    print('RECOVERY_FINAL',r['later_totals'],flush=True)


def render():
    import cv2
    import numpy as np
    r=load();cards=[]
    for seq in previous.FINAL:
        spec=previous.source.PLAN['sequences'][seq];maps=[]
        for mode in ['initial','selected']:
            mapping={}
            for row in r['runs'][mode][seq]['observations']:mapping.setdefault(row['frame'],[]).append(row)
            maps.append(mapping)
        target=OUTPUT/(seq.lower()+'-later.mp4');writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),spec['fps'],(1536,490));assert writer.isOpened()
        frames=r['runs']['selected'][seq]['frames'];first=r['runs']['selected'][seq]['source_start_frame'];poster=None
        for n in range(frames):
            image=previous.source.image(seq,first+n);panes=[]
            for side,mode in enumerate(['initial','selected']):
                im=image.copy();color=(255,150,30) if side==0 else (40,220,40)
                for row in maps[side].get(n,[]):
                    b=[round(row[k]) for k in ['x1','y1','x2','y2']];cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),color,2)
                header=np.zeros((58,768,3),np.uint8);m=r['runs'][mode][seq]['metrics']
                cv2.putText(header,'Original profile' if side==0 else 'Corrected profile after round 4',(10,23),0,.58,color,1)
                cv2.putText(header,f'{seq} | source {(first+n)/spec["fps"]:.2f}s | Recall {m["recall"]:.1f}% / precision {m["precision"]:.1f}%',(10,46),0,.43,(255,255,255),1)
                panes.append(cv2.vconcat([header,cv2.resize(im,(768,432))]))
            combined=cv2.hconcat(panes);writer.write(combined)
            if n==frames//2:poster=base64.b64encode(cv2.imencode('.jpg',cv2.resize(combined,(960,306)))[1]).decode()
        writer.release();cap=cv2.VideoCapture(str(target));decoded=0
        while cap.read()[0]:decoded+=1
        cap.release();assert decoded==frames
        r['runs']['selected'][seq]['export']={'video':target.name,'decoded_frames':decoded};write(r)
        cards.append(f'<article><h2>{seq} · later time window</h2><video controls preload="none" poster="data:image/jpeg;base64,{poster}" src="{target.name}"></video></article>')
    rows=[]
    for scope,values in [('Development',{'initial':r['results']['initial']['totals'],'selected':r['results'][r['selected']]['totals']}),('Later check',r['later_totals'])]:
        for mode,a in values.items():rows.append(f'<tr><td>{scope}</td><td>{mode}</td><td>{a["recall"]:.1%}</td><td>{a["f1"]:.1%}</td><td>{a["precision"]:.1%}</td></tr>')
    rejected=r['prior_legacy_check'];current=r['results'][r['selected']]['legacy_person_guard']
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Larger sample — corrected round 4</title><style>body{margin:0;background:#eef3f5;color:#17303d;font:16px system-ui}main{max-width:1250px;margin:auto;padding:28px}p{line-height:1.6}table{width:100%;border-collapse:collapse;background:white}td,th{padding:12px;text-align:left;border-bottom:1px solid #cbd8de}th{background:#173944;color:white}.notice{background:#fff1cb;padding:18px;border-radius:12px}article{background:white;padding:18px;border-radius:12px;margin:20px 0}video{width:100%;border-radius:8px}a{color:#006d77}</style><main><h1>Larger sample: corrected fourth round</h1><p>Three development rounds increased recall by lowering track initiation, but brought back an older false-person failure. That result remains intact. A fourth round restores the legacy-person check, ranks passing settings by recall, F1, precision, and checks the frozen result on later native-FPS source windows.</p><p class="notice">All six source cameras appeared in earlier project work. The later windows are excluded from this correction’s tuning, but share cameras/actors with earlier footage. This is a larger regression/development study, not independent new-property validation. Metrics concern labeled person/vehicle boxes, not doorway crossings or hourly traffic accuracy.</p><table><tr><th>Scope</th><th>Version</th><th>Recall</th><th>F1</th><th>Precision</th></tr>'''+''.join(rows)+'''</table><p>Previously rejected high-recall version: '''+str(rejected['roof_false_boxes'])+''' roof and '''+str(rejected['fixture_false_boxes'])+''' fixture false-person boxes. Corrected version: '''+str(current['roof_false_boxes'])+''' roof / '''+str(current['fixture_false_boxes'])+''' fixture boxes; standing person '''+str(current['standing_frames'])+''' frames under '''+str(current['standing_ids'])+''' ID. Both old checked scooters and all reported vehicle-fixture/cargo checks remain required.</p><p>Corrected profile: '''+str(r['selected_profile'])+'''. No location-based prediction filters, manual relabeling or detector training. General application defaults remain unchanged. This saved JSON is an experiment profile, not a camera-line configuration for --config.</p>'''+''.join(cards)+'''<p><a href="report.json">All fourth-round candidates, guard rejections and results</a> · <a href="../larger-confidence/review.html">Preserved first three rounds and their failed frozen result</a> · <a href="../../configs/larger-confidence-recovered-profile.json">Corrected experiment profile</a></p><p>Labels: MOT17 and UA-DETRAC; pinned official TrackEval with published ignore rules. Vehicle types are grouped; motorcycle recall remains unmeasured beyond the small old spot checks.</p></main></html>'''
    (OUTPUT/'review.html').write_text(doc)


if __name__=='__main__':develop();final();render()
