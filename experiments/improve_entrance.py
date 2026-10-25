"""Second entrance experiment: geometry/duplicate fixes and newly reserved source takes."""
import argparse
import copy
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import retail_entrance as previous

PLAN_PATH = ROOT / 'configs/entrance-improvement.json'
PLAN = json.loads(PLAN_PATH.read_text())
DATA = ROOT / 'data/entrance-improvement'
OUTPUT = ROOT / 'outputs/entrance-improvement'
TEMP = Path('/private/tmp/propsight-entrance-improvement')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def manifest():
    return json.loads((DATA / 'manifest.json').read_text())


def report():
    path = OUTPUT / 'report.json'
    return json.loads(path.read_text()) if path.exists() else {'started_at_utc': stamp(), 'runs': {}}


previous.DATA = DATA
previous.OUTPUT = OUTPUT
previous.PLAN = {**PLAN, 'native_fps': PLAN['fps']}
previous.manifest = manifest
previous.report = report
previous.write_json = write


def profiles():
    old = json.loads((ROOT / 'outputs/retail-entrance/report.json').read_text())
    baseline = copy.deepcopy(old['protocol']['candidates']['full-frame'])
    camera = copy.deepcopy(baseline['camera'])
    camera['line'] = [[205/383, 145/287], [335/383, 145/287]]
    camera.update(deadband_pixels=3, confirm_seconds=.12, max_gap_seconds=.6,
                  track_buffer_seconds=1.2, person_tracking={'track_high_thresh': .25, 'new_track_thresh': .4})
    result = {'full-frame': baseline}
    for name, settings in [('coverage', None), ('coverage-nms', {'iou': .6}),
                           ('coverage-nested', {'nested_height_ratio': .7}),
                           ('coverage-combined', {'iou': .6, 'nested_height_ratio': .7})]:
        config = copy.deepcopy(camera)
        if settings:
            config['person_suppression'] = settings
        result[name] = {'model': PLAN['model'], 'imgsz': PLAN['imgsz'], 'camera': config}
    return result


def references(split):
    import cv2
    import numpy as np
    import xml.etree.ElementTree as ET
    current = manifest()
    if len({info['video_sha256'] for info in current['sequences'].values()}) != len(current['sequences']):
        raise RuntimeError('Duplicate source recordings across splits')
    r = report()
    if split == 'holdout':
        verify(r)
        if r.get('holdout_opened_at_utc'):
            raise RuntimeError('Cannot alter references after reading test results')
        r.setdefault('holdout_reference_review_opened_at_utc', stamp())
        write(OUTPUT / 'report.json', r)
    directory = TEMP / 'reference-review'
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA / 'annotations.zip') as archive:
        for seq in PLAN[split]:
            if seq in current['entrance_references']['sequences']:
                continue
            frames = ET.fromstring(archive.read(seq+'.xml')).findall('frame')
            segments = {}
            boxes = {}
            for n, frame in enumerate(frames):
                if int(frame.attrib['number']) != n:
                    raise RuntimeError(f'Noncontiguous annotations: {seq}')
                for obj in frame.findall('objectlist/object'):
                    identity = int(obj.attrib['id'])
                    b = obj.find('box').attrib
                    x, y, w, h = [float(b[k]) for k in ['xc','yc','w','h']]
                    boxes[n, identity] = [x-w/2, y-h/2, x+w/2, y+h/2]
                    situation = obj.findtext('hypothesislist/hypothesis/situation')
                    if situation not in ['shop enter', 'shop exit']:
                        continue
                    rows = segments.setdefault(identity, [])
                    if rows and rows[-1][2] == situation and rows[-1][1] == n-1:
                        rows[-1][1] = n
                    else:
                        rows.append([n, n, situation])
            events = []
            for identity, rows in segments.items():
                for first, last, situation in rows:
                    _, frame, x, bottom = min((abs(boxes[n,identity][3]-145), n,
                                               (boxes[n,identity][0]+boxes[n,identity][2])/2,
                                               boxes[n,identity][3]) for n in range(first,last+1))
                    events.append({'frame': frame, 'track_id': identity,
                                   'direction': 'entry' if situation == 'shop enter' else 'exit',
                                   'point': [x,bottom], 'activity_interval': [first,last]})
            events.sort(key=lambda e:e['frame'])
            times = sorted({max(0,min(len(frames)-1,e['frame']+offset)) for e in events for offset in [-12,0,12]})
            times = times or [0,len(frames)//2,len(frames)-1]
            tiles = []
            decoded = 0
            for n, image in enumerate(previous.decode(seq)):
                decoded += 1
                if n not in times:
                    continue
                for obj in frames[n].findall('objectlist/object'):
                    identity = int(obj.attrib['id'])
                    x1,y1,x2,y2 = boxes[n,identity]
                    cv2.rectangle(image,(round(x1),round(y1)),(round(x2),round(y2)),(0,255,0),1)
                    cv2.putText(image,str(identity),(round(x1),round(y1)),0,.3,(0,0,255),1)
                cv2.line(image,(205,145),(335,145),(0,255,255),1)
                image = cv2.resize(image[45:210,170:355],(370,330))
                cv2.putText(image,f'{seq} f{n}',(3,320),0,.4,(255,255,255),1)
                tiles.append(image)
            if decoded != len(frames):
                raise RuntimeError(f'Frame alignment mismatch: {seq} {decoded}/{len(frames)}')
            while len(tiles)%3:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(directory/(seq+'.jpg')),cv2.vconcat([cv2.hconcat(tiles[n:n+3]) for n in range(0,len(tiles),3)]))
            current['entrance_references']['sequences'][seq] = {
                'split':split,'frames':decoded,'events':events,'review_status':'pending visual review'}
            write(DATA / 'manifest.json', current)
            print(seq, decoded, events, flush=True)


def approve_references(split):
    current = manifest()
    for seq in PLAN[split]:
        entry = current['entrance_references']['sequences'][seq]
        entry['review_status'] = 'source frames and published activities visually checked before detector outcomes'
        entry['reviewed_at_utc'] = stamp()
    write(DATA / 'manifest.json', current)


def evaluate(seq, profile):
    import numpy as np
    import torch
    from ultralytics.engine.results import Results
    from counting import LineCounter, confirmation_frames
    from tracker import SeparateVehicleTracker
    camera = profile['camera']
    tracker = SeparateVehicleTracker(ROOT / 'configs/bytetrack-low-fps.yaml',camera['vehicle_tracking'],camera['person_tracking'],
                                     frame_rate=25,track_buffer_seconds=camera.get('track_buffer_seconds'),
                                     person_suppression=camera.get('person_suppression'))
    points = [tuple(round(v*s) for v,s in zip(p,(383,287))) for p in camera['line']]
    counter = LineCounter(*points,entry_side=camera['entry_side'],deadband=camera['deadband_pixels'],
                          confirm_frames=confirmation_frames(camera,25),max_gap_frames=round(25*camera['max_gap_seconds']))
    blank = np.zeros((288,384,3),np.uint8)
    observations, events = [], []
    for n, detections in enumerate(previous.raw(seq,profile)):
        counter.prune(n)
        result = Results(blank,path='',names={0:'person'},boxes=torch.tensor(np.asarray(detections,np.float32).reshape(-1,6)))
        rows = []
        for bounds,identity,label,confidence,group in tracker.update(result,blank,{0:'person'},.1,.7):
            if group != 'person':
                continue
            rows.append([identity,*bounds,confidence])
            point = [(bounds[0]+bounds[2])/2,bounds[3]]
            direction = counter.update(identity,point,n)
            if direction:
                events.append({'frame':n,'video_seconds':n/25,'track_id':identity,'point':point,'direction':direction})
        observations.append(rows)
    reference = manifest()['entrance_references']['sequences'][seq]['events']
    record = {'profile':profile,'frames':len(observations),'events':events,'observations':observations,
              'scores':previous.score_events(reference,events),'person_boxes_suppressed':tracker.person_boxes_suppressed,
              'finished_at_utc':stamp()}
    previous.audit_events(seq,record)
    print(seq,record['count_audit'],flush=True)
    return record


def verify(r):
    if 'frozen_sha256' not in r:
        raise RuntimeError('Freeze a validation-selected setup first')
    for filename,digest in r['frozen_sha256'].items():
        if sha(ROOT/filename) != digest:
            raise RuntimeError(f'Frozen input changed: {filename}')
    current = manifest()
    for seq,info in r['frozen_sources'].items():
        if sha(Path(current['sequences'][seq]['video'])) != info:
            raise RuntimeError(f'Source changed: {seq}')
    frozen_refs = r['frozen_development_validation_references']
    for seq,reference in frozen_refs.items():
        if current['entrance_references']['sequences'][seq] != reference:
            raise RuntimeError(f'Development/validation reference changed: {seq}')


def validate():
    r = report()
    if r.get('holdout_reference_review_opened_at_utc') or r.get('holdout_opened_at_utc'):
        raise RuntimeError('A new holdout is required for further tuning')
    current = manifest()
    for seq in PLAN['validation']:
        if current['entrance_references']['sequences'][seq]['review_status'] == 'pending visual review':
            raise RuntimeError('Review validation reference frames before detector outcomes')
    r.setdefault('protocol',{'plan':PLAN,'candidates':profiles(),'reference_notes':current['entrance_references'],
                             'validation_opened_at_utc':stamp()})
    development = [seq for seq,info in current['sequences'].items() if info['split']=='development']
    r['development_sequences'] = development
    write(OUTPUT/'report.json',r)
    for name,profile in r['protocol']['candidates'].items():
        r['runs'].setdefault(name,{})
        for seq in development+list(PLAN['validation']):
            if seq in r['runs'][name]:
                continue
            record = evaluate(seq,profile)
            record.pop('observations')
            record['observations_omitted_after_scoring'] = 'Development/validation only; original detector cache and frozen inputs permit replay. Test observations are retained.'
            r['runs'][name][seq] = record
            write(OUTPUT/'report.json',r)
        r.setdefault('development_count_audit',{})[name] = previous.count_audit_totals([r['runs'][name][s] for s in development])
        r.setdefault('validation_count_audit',{})[name] = previous.count_audit_totals([r['runs'][name][s] for s in PLAN['validation']])
        write(OUTPUT/'report.json',r)
    names = list(r['protocol']['candidates'])
    selected = max(names,key=lambda n:(r['validation_count_audit'][n]['f1'],r['development_count_audit'][n]['f1'],
                                      -r['validation_count_audit'][n]['unverified_predictions'],-names.index(n)))
    r['selected'] = selected
    r['frozen_profile'] = r['protocol']['candidates'][selected]
    r['frozen_at_utc'] = stamp()
    files = ['tracker.py','counting.py','detection.py','experiments/retail_entrance.py','experiments/improve_entrance.py',
             'configs/entrance-improvement.json','configs/bytetrack-low-fps.yaml','yolo11s.pt',
             'data/entrance-improvement/annotations.zip']
    r['frozen_sha256'] = {p:sha(ROOT/p) for p in files}
    r['frozen_sources'] = {s:i['video_sha256'] for s,i in current['sequences'].items()}
    r['frozen_development_validation_references'] = copy.deepcopy(current['entrance_references']['sequences'])
    write(OUTPUT/'report.json',r)
    write(ROOT/'configs/entrance-improvement-selected.json',r['frozen_profile']['camera'])
    print('FROZEN',selected,r['validation_count_audit'],flush=True)


def holdout():
    r = report()
    verify(r)
    current = manifest()
    for seq in PLAN['holdout']:
        if current['entrance_references']['sequences'][seq]['review_status'] == 'pending visual review':
            raise RuntimeError('Review reserved references before model outcomes')
    r.setdefault('holdout_opened_at_utc',stamp())
    r['protocol']['reference_notes'] = current['entrance_references']
    r['holdout_reference_manifest_sha256'] = sha(DATA/'manifest.json')
    write(OUTPUT/'report.json',r)
    for name in dict.fromkeys(['full-frame',r['selected']]):
        r['runs'].setdefault(name,{})
        for seq in PLAN['holdout']:
            if seq not in r['runs'][name]:
                r['runs'][name][seq] = evaluate(seq,r['protocol']['candidates'][name])
                write(OUTPUT/'report.json',r)
        r.setdefault('holdout_count_audit',{})[name] = previous.count_audit_totals([r['runs'][name][s] for s in PLAN['holdout']])
        r.setdefault('holdout_totals',{})[name] = previous.totals([r['runs'][name][s] for s in PLAN['holdout']])
        write(OUTPUT/'report.json',r)
    verify(r)
    r['finished_at_utc'] = stamp()
    write(OUTPUT/'report.json',r)
    print('HOLDOUT_COMPLETE',r['holdout_count_audit'],flush=True)


def acquire():
    DATA.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    if (DATA / 'manifest.json').exists():
        raise RuntimeError('Sources already acquired; preserve the existing reviewed manifest')
    old_data = ROOT / 'data/retail-entrance'
    old = json.loads((old_data / 'manifest.json').read_text())
    current = {'acquired_at_utc': stamp(), 'plan': PLAN, 'sequences': {},
               'entrance_references': {'method': PLAN['reference_method'], 'sequences': {}}}
    labels = {}
    with zipfile.ZipFile(old_data / 'annotations.zip') as archive:
        for seq, info in old['sequences'].items():
            current['sequences'][seq] = {**info, 'split': 'development', 'video': str((old_data / info['video']).resolve())}
            labels[seq] = archive.read(seq + '.xml')
            current['entrance_references']['sequences'][seq] = copy.deepcopy(old['entrance_references']['sequences'][seq])
            current['entrance_references']['sequences'][seq]['split'] = 'development'

    def get(item):
        split, seq, label = item
        if seq in current['sequences']:
            raise RuntimeError('Source take overlaps previous experiment')
        directory = DATA / split
        directory.mkdir(exist_ok=True)
        video = directory / (seq + '.mpg')
        annotation = TEMP / label
        urls = {}
        for filename, path in [(seq + '.mpg', video), (label, annotation)]:
            url = f'https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA2/{seq}/{filename}'
            if not path.exists():
                with urllib.request.urlopen(url, timeout=90) as response:
                    path.write_bytes(response.read())
            urls[filename] = url
        print(f'Acquired {split}: {seq}', flush=True)
        xml = annotation.read_bytes()
        return seq, xml, {'split': split, 'video': str(video.resolve()), 'video_sha256': sha(video), 'annotation': seq+'.xml', 'annotation_sha256': hashlib.sha256(xml).hexdigest(), 'urls': urls}

    items = [(split, seq, label) for split in ['validation', 'holdout'] for seq, label in PLAN[split].items()]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for seq, xml, info in pool.map(get, items):
            current['sequences'][seq] = info
            labels[seq] = xml
    with zipfile.ZipFile(DATA / 'annotations.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for seq, xml in labels.items():
            archive.writestr(seq+'.xml', xml)
    current['annotation_archive_sha256'] = sha(DATA / 'annotations.zip')
    write(DATA / 'manifest.json', current)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['acquire','references','approve-references','validate','holdout','gallery'])
    parser.add_argument('split',nargs='?',choices=['validation','holdout'])
    args = parser.parse_args()
    if args.phase == 'acquire':
        acquire()
    elif args.phase == 'references':
        references(args.split)
    elif args.phase == 'approve-references':
        approve_references(args.split)
    elif args.phase == 'validate':
        validate()
    elif args.phase == 'holdout':
        holdout()
    elif args.phase == 'gallery':
        r = report()
        r['validation_totals'] = {n:previous.totals([r['runs'][n][s] for s in PLAN['validation']]) for n in r['protocol']['candidates']}
        write(OUTPUT/'report.json',r)
        previous.gallery()
