"""Compare only the detector cutoff on the two reported development clips."""
import base64
import copy
import csv
import hashlib
import html
import json
import sys
import tempfile
from argparse import Namespace
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tracker import run, load_config
from evaluation import box_iou

OUTPUT = ROOT/'outputs/vehicle-confidence'
CASES = {
    'seoul-songridan-gil': {
        'source':'data/retail-security-test/songridan-gil.mp4',
        'false_objects':{'left-lamp':[335,645,505,720],
                         'middle-lamp':[830,490,970,562],
                         'right-lamp':[1065,410,1190,475],
                         'manhole':[681,309,720,323]},
        'real_objects':{'passing-white-car':{'frame':20,'box':[639,367,959,530],'classes':['car']}}},
    'taiwan-linkou': {
        'source':'data/retail-security-test/linkou-old-street.mp4',
        'false_objects':{'scooter-cargo-as-truck':[1010,475,1165,640]},
        'real_objects':{'parked-white-car':{'frame':40,'box':[223,373,450,531],'classes':['car']},
                        'moving-dark-car':{'frame':40,'box':[1164,289,1280,380],'classes':['car']},
                        'scooter-beside-white-car':{'frame':40,'box':[438,380,516,477],'classes':['motorcycle']},
                        'foreground-left-scooter':{'frame':40,'box':[898,592,1010,717],'classes':['motorcycle']}}},
}


def stamp():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(report):(OUTPUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')


def assess(case, record):
    observations = record['observations']
    result = {'false_object_frames':{},'real_object_spot_checks':{},
              'frame40_class_boxes':dict(Counter(row['class'] for row in observations if row['frame']==40))}
    for name,bounds in case['false_objects'].items():
        rows = [row for row in observations if row['class'] in ['car','truck','bus']
                and float(box_iou([bounds],[[row[k] for k in ['x1','y1','x2','y2']]])[0,0])>=.3]
        frames = sorted({row['frame'] for row in rows})
        result['false_object_frames'][name] = {'frames_with_vehicle_false_box':len(frames),'frames':frames,
                                               'frame40_present':40 in frames}
    for name,spec in case['real_objects'].items():
        rows = [row for row in observations if row['frame']==spec['frame'] and row['class'] in spec['classes']
                and float(box_iou([spec['box']],[[row[k] for k in ['x1','y1','x2','y2']]])[0,0])>=.5]
        result['real_object_spot_checks'][name] = {'frame':spec['frame'],'correct_class_and_box':bool(rows)}
    return result


def compare():
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(4)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'report.json').exists():
        raise RuntimeError('Preserve the completed comparison; choose a different output for a new experiment')
    old = json.loads((ROOT/'outputs/readme-location-replays/report.json').read_text())
    report = {'registered_at_utc':stamp(),'status':'Two previously inspected clips; development comparison only',
              'cutoffs':[.1,.55], 'model':'yolo11s.pt','imgsz':1280,'iou':.7,'device':'cpu',
              'classes':['person','car','truck','bus','motorcycle','bicycle'],
              'changed':'Detector confidence only; tracker gates, camera line and other settings unchanged.',
              'no_location_filters':True,
              'spot_check_protocol':'Reported false objects and selected real vehicles checked with evaluation-only reference boxes. These boxes never enter inference, filtering, tracking or counting. False objects: car/truck/bus box IoU>=0.3; real controls: correct class and box IoU>=0.5 at the named frame. These selected checks are not full-scene accuracy.',
              'cases':copy.deepcopy(CASES),'runs':{},
              'model_sha256':sha(ROOT/'yolo11s.pt'),
              'historical_replay_report_sha256':sha(ROOT/'outputs/readme-location-replays/report.json')}
    write(report)
    cards=[]
    with tempfile.TemporaryDirectory(prefix='propsight-confidence-',dir='/private/tmp') as temp:
        temp=Path(temp)
        for site,case in CASES.items():
            camera=copy.deepcopy(old['runs'][site]['summary']['config'])
            cfg=temp/'camera.json';cfg.write_text(json.dumps(camera));load_config(cfg)
            assert 'excluded_boxes' not in camera.get('vehicle_tracking',{})
            paths=[]
            for cutoff in [.1,.55]:
                name=f'{site}-{str(cutoff).replace(".","")}'
                _,record=run(Namespace(video=str(ROOT/case['source']),config=str(cfg),
                    tracker=str(ROOT/'configs/bytetrack-low-fps.yaml'),model=str(ROOT/'yolo11s.pt'),
                    classes=report['classes'],output=str(temp),name=name,save_observations=True,
                    preview=None,conf=cutoff,iou=.7,imgsz=1280,device='cpu',fps=None,max_frames=None,
                    realtime=False,save_timing=False))
                observation_file=temp/record.pop('observations_file')
                with observation_file.open(newline='') as handle:
                    rows=list(csv.DictReader(handle))
                for row in rows:
                    row['frame']=int(row['frame'])
                    for key in ['video_seconds','confidence','x1','y1','x2','y2']:row[key]=float(row[key])
                record['observations']=rows
                record['source_sha256']=sha(ROOT/case['source'])
                record['spot_checks']=assess(case,record)
                report['runs'].setdefault(site,{})[str(cutoff)]=record
                write(report)
                paths.append(temp/record['video'])
                print('SCORED',site,cutoff,json.dumps(record['spot_checks']),flush=True)
            # Only two side-by-side videos, one report and one page are retained.
            captures=[cv2.VideoCapture(str(p)) for p in paths]
            target=OUTPUT/(site+'.mp4')
            writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),5,(1536,470))
            if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
            frames=0;preview=None
            try:
                while True:
                    items=[cap.read() for cap in captures]
                    if not any(ok for ok,_ in items):break
                    if not all(ok for ok,_ in items):raise RuntimeError('Comparison sources have different lengths')
                    panes=[]
                    for side,(_,image) in enumerate(items):
                        header=np.zeros((38,768,3),np.uint8)
                        cv2.putText(header,f'Confidence {report["cutoffs"][side]} | {frames/5:.1f}s',(12,25),0,.65,(255,255,255),1)
                        panes.append(cv2.vconcat([header,cv2.resize(image,(768,432))]))
                    combined=cv2.hconcat(panes);writer.write(combined)
                    if frames==40:preview=base64.b64encode(cv2.imencode('.jpg',combined)[1]).decode()
                    frames+=1
            finally:
                writer.release()
                for cap in captures:cap.release()
            cap=cv2.VideoCapture(str(target));decoded=0
            while cap.read()[0]:decoded+=1
            cap.release()
            assert decoded==frames==report['runs'][site]['0.55']['summary']['frames_processed']
            for record in report['runs'][site].values():
                record.pop('video')
                record['comparison_video']=target.name
                record['comparison_decoded_frames']=decoded
            rows=[]
            for name in case['false_objects']:
                before=report['runs'][site]['0.1']['spot_checks']['false_object_frames'][name]
                after=report['runs'][site]['0.55']['spot_checks']['false_object_frames'][name]
                rows.append(f'<tr><td>{html.escape(name)}</td><td>{before["frames_with_vehicle_false_box"]}</td><td>{after["frames_with_vehicle_false_box"]}</td></tr>')
            controls=[]
            for name in case['real_objects']:
                a=report['runs'][site]['0.1']['spot_checks']['real_object_spot_checks'][name]
                b=report['runs'][site]['0.55']['spot_checks']['real_object_spot_checks'][name]
                controls.append(f'<tr><td>{html.escape(name)} · frame {a["frame"]}</td><td>{"Present" if a["correct_class_and_box"] else "Not matched"}</td><td>{"Present" if b["correct_class_and_box"] else "Not matched"}</td></tr>')
            cards.append(f'<article><h2>{html.escape(site)}</h2><video controls preload="metadata" poster="data:image/jpeg;base64,{preview}" src="{target.name}"></video><h3>Frames containing each reported false vehicle box</h3><table><tr><th>Reported error</th><th>0.1</th><th>0.55</th></tr>{"".join(rows)}</table><h3>Selected real-vehicle checks</h3><table><tr><th>Real object</th><th>0.1</th><th>0.55</th></tr>{"".join(controls)}</table></article>')
            write(report)
    assert sha(ROOT/'outputs/readme-location-replays/report.json')==report['historical_replay_report_sha256']
    report['finished_at_utc']=stamp()
    report['historical_report_unchanged']=True
    write(report)
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vehicle confidence comparison</title><style>body{margin:0;background:#eef3f5;color:#17303d;font:16px system-ui}main{max-width:1250px;margin:auto;padding:28px}p{line-height:1.6}article{background:white;padding:20px;border-radius:12px;margin:24px 0}video{width:100%;border-radius:8px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #d4e0e6}th{background:#173944;color:white}.notice{background:#fff1cb;padding:16px;border-radius:10px}a{color:#006d77}</style><main><h1>Confidence 0.1 versus 0.55</h1><p>Fresh YOLO11s/1280 inference on both complete 18-second, 5 FPS source clips. Only the detector confidence cutoff changes; camera configuration, ByteTrack, counting and classes are identical within each pair. No camera exclusions, manual class reassignment or model training.</p><p class="notice">Known development clips. Selected false-object and real-vehicle checks are evaluation only, not a complete label set or an independent accuracy benchmark. Results do not establish new crossing precision/recall. Higher confidence also removes low-score evidence ByteTrack could use to recover tracks. Default settings and README screenshots remain unchanged.</p>'''+''.join(cards)+'''<p><a href="report.json">Shared settings, tracks, events and checks</a> · <a href="../../docs/screenshots/README.md">Source attribution</a></p><p>Public clip attribution: EarthCam / Songridan-gil, Seoul and Linkou, New Taipei City. Existing private testing scope; no new footage acquired.</p></main></html>'''
    (OUTPUT/'review.html').write_text(doc)
    print('COMPLETE',OUTPUT/'review.html',flush=True)


if __name__=='__main__':compare()
