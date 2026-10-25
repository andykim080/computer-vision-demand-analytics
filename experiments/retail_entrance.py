"""Reproducible same-camera entrance pilot; separate source takes and frozen holdout."""
import argparse
import base64
import copy
import hashlib
import html
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
PLAN_PATH = ROOT / 'configs/retail-entrance-evaluation.json'
PLAN = json.loads(PLAN_PATH.read_text())
DATA = ROOT / 'data/retail-entrance'
OUTPUT = ROOT / 'outputs/retail-entrance'
TEMP = Path('/private/tmp/propsight-entrance')


def manifest():
    return json.loads((DATA / 'manifest.json').read_text())


def report():
    path = OUTPUT / 'report.json'
    return json.loads(path.read_text()) if path.exists() else {'runs': {}, 'started_at_utc': stamp()}


def profiles():
    camera = copy.deepcopy(PLAN['camera'])
    baseline = copy.deepcopy(camera)
    for key in ['detection_crop', 'confirm_seconds', 'track_buffer_seconds']:
        baseline.pop(key, None)
    baseline.update(deadband_pixels=8, confirm_frames=3,
                    person_tracking={'track_high_thresh': .25, 'new_track_thresh': .55})
    return {
        'full-frame': {'model': 'yolo11s.pt', 'imgsz': 640, 'camera': baseline},
        'doorway-pretrained': {'model': 'yolo11s.pt', 'imgsz': 640, 'camera': camera},
        'doorway-trained': {'model': 'models/retail-entrance-person.pt', 'imgsz': 640, 'camera': camera},
    }


def decode(seq):
    import cv2
    info = manifest()['sequences'][seq]
    capture = cv2.VideoCapture(str(DATA / info['video']))
    # These MPEG2 streams have unreliable metadata/seeking; always decode sequentially.
    while True:
        ok, image = capture.read()
        if not ok:
            break
        yield image
    capture.release()


def raw(seq, profile):
    import numpy as np
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    source = manifest()['sequences'][seq]
    key = hashlib.sha256(json.dumps({'source': source['video_sha256'], 'model': sha(ROOT / profile['model']), 'crop': profile['camera'].get('detection_crop'), 'imgsz': profile['imgsz'], 'conf': .1, 'iou': .7}, sort_keys=True).encode()).hexdigest()[:20]
    cache = TEMP / 'raw-cache' / f'{seq}-{key}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    model = YOLO(str(ROOT / profile['model']))
    crop = profile['camera'].get('detection_crop')
    batch, rows = [], []
    start = time.monotonic()

    def flush():
        results = model.predict(batch, classes=[0], conf=.1, iou=.7,
                                imgsz=profile['imgsz'], device='cpu', verbose=False)
        for result in results:
            boxes = result.boxes.data.cpu().numpy().copy()
            if crop and len(boxes):
                boxes[:, [0, 2]] += crop[0]
                boxes[:, [1, 3]] += crop[1]
            rows.append(boxes.tolist())
        batch.clear()

    for image in decode(seq):
        batch.append(np.ascontiguousarray(image[crop[1]:crop[3], crop[0]:crop[2]]) if crop else image)
        if len(batch) == 8:
            flush()
            if len(rows) % 200 == 0:
                print(f'{seq} {profile["model"]}: {len(rows)} frames, {time.monotonic()-start:.0f}s', flush=True)
    if batch:
        flush()
    expected = manifest()['entrance_references']['sequences'][seq]['frames']
    if len(rows) != expected:
        raise RuntimeError(f'Video/annotation frame mismatch: {seq} {len(rows)} != {expected}')
    write_json(cache, rows)
    return rows


def score_events(reference, predicted):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    costs = np.full((len(reference), len(predicted)), 1e6)
    for i, truth in enumerate(reference):
        for j, event in enumerate(predicted):
            dt = abs(truth['frame'] - event['frame']) / PLAN['native_fps']
            dx = abs(truth['point'][0] - event['point'][0])
            if truth['direction'] == event['direction'] and dt <= 1 and dx <= 38.4:
                costs[i, j] = dt + dx / 1000
    pairs = []
    if len(reference) and len(predicted):
        ii, jj = linear_sum_assignment(costs)
        pairs = [[int(i), int(j)] for i, j in zip(ii, jj) if costs[i, j] < 1e6]
    tp, fp, fn = len(pairs), len(predicted) - len(pairs), len(reference) - len(pairs)
    return {'matched': tp, 'extra': fp, 'missed': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'reference_counts': dict(Counter(e['direction'] for e in reference)),
            'model_counts': dict(Counter(e['direction'] for e in predicted)),
            'matches': pairs, 'unmatched_reference': [i for i in range(len(reference)) if all(p[0] != i for p in pairs)],
            'unmatched_prediction': [j for j in range(len(predicted)) if all(p[1] != j for p in pairs)]}


def evaluate(seq, profile):
    import numpy as np
    import torch
    from ultralytics.engine.results import Results
    from counting import LineCounter, confirmation_frames
    from tracker import SeparateVehicleTracker
    fps = PLAN['native_fps']
    camera = profile['camera']
    tracker = SeparateVehicleTracker(ROOT / 'configs/bytetrack-low-fps.yaml', camera['vehicle_tracking'], camera['person_tracking'],
                                     frame_rate=fps if camera.get('track_buffer_seconds') else 30,
                                     track_buffer_seconds=camera.get('track_buffer_seconds'))
    points = [tuple(round(p * size) for p, size in zip(point, (383, 287))) for point in camera['line']]
    counter = LineCounter(*points, entry_side=camera['entry_side'], deadband=camera['deadband_pixels'],
                          confirm_frames=confirmation_frames(camera, fps), max_gap_frames=round(camera['max_gap_seconds'] * fps))
    blank = np.zeros((288, 384, 3), np.uint8)
    observations, events = [], []
    for n, boxes in enumerate(raw(seq, profile)):
        counter.prune(n)
        result = Results(blank, path='', names={0: 'person'}, boxes=torch.tensor(np.asarray(boxes, np.float32).reshape(-1, 6)))
        tracked = tracker.update(result, blank, {0: 'person'}, .1, .7)
        rows = []
        for bounds, identity, label, confidence, group in tracked:
            if group != 'person':
                continue
            rows.append([identity, *bounds, confidence])
            point = [(bounds[0] + bounds[2]) / 2, bounds[3]]
            direction = counter.update(identity, point, n)
            if direction:
                events.append({'frame': n, 'video_seconds': n / fps, 'track_id': identity, 'direction': direction, 'point': point})
        observations.append(rows)
    reference = manifest()['entrance_references']['sequences'][seq]['events']
    scores = score_events(reference, events)
    print(seq, scores, flush=True)
    return {'profile': profile, 'events': events, 'scores': scores, 'observations': observations, 'frames': len(observations), 'finished_at_utc': stamp()}


def totals(records):
    tp = sum(r['scores']['matched'] for r in records)
    fp = sum(r['scores']['extra'] for r in records)
    fn = sum(r['scores']['missed'] for r in records)
    directional = {}
    for direction in ['entry', 'exit']:
        truth = sum(r['scores']['reference_counts'].get(direction, 0) for r in records)
        model = sum(r['scores']['model_counts'].get(direction, 0) for r in records)
        directional[direction] = {'reference': truth, 'model': model,
                                  'sum_absolute_clip_error': sum(abs(r['scores']['reference_counts'].get(direction, 0) - r['scores']['model_counts'].get(direction, 0)) for r in records)}
    return {'matched': tp, 'extra': fp, 'missed': fn, 'precision': tp / (tp + fp) if tp + fp else 0,
            'recall': tp / (tp + fn) if tp + fn else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0, 'directions': directional}


def audit_events(seq, record):
    """Verify crossing identity while reporting timing separately; evaluation IDs only."""
    import numpy as np
    import xml.etree.ElementTree as ET
    from scipy.optimize import linear_sum_assignment
    from evaluation import box_iou
    with zipfile.ZipFile(DATA / 'annotations.zip') as archive:
        frames = ET.fromstring(archive.read(seq + '.xml')).findall('frame')
    associations = {}
    for n, (frame, predicted) in enumerate(zip(frames, record['observations'])):
        truth = []
        for obj in frame.findall('objectlist/object'):
            b = obj.find('box').attrib
            x, y, w, h = [float(b[k]) for k in ['xc', 'yc', 'w', 'h']]
            truth.append([int(obj.attrib['id']), x-w/2, y-h/2, x+w/2, y+h/2])
        if not truth or not predicted:
            continue
        scores = box_iou([g[1:] for g in truth], [p[1:5] for p in predicted])
        scores[scores < .3] = 0
        ii, jj = linear_sum_assignment(-scores)
        for i, j in zip(ii, jj):
            if scores[i, j] > 0:
                associations.setdefault((predicted[j][0], truth[i][0]), []).append(n)
    reference = manifest()['entrance_references']['sequences'][seq]['events']
    predicted = record['events']
    costs = np.full((len(reference), len(predicted)), 1e6)
    evidence = {}
    for i, truth in enumerate(reference):
        for j, event in enumerate(predicted):
            delay = (event['frame'] - truth['frame']) / 25
            if event['direction'] != truth['direction'] or abs(delay) > 3:
                continue
            window = (min(truth['frame'], event['frame']) - 25, max(truth['frame'], event['frame']) + 25)
            votes = sum(window[0] <= n <= window[1] for n in associations.get((event['track_id'], truth['track_id']), []))
            if votes >= 3:
                costs[i, j] = abs(delay) + 1 / (votes + 1)
                evidence[i, j] = {'reference_index': i, 'prediction_index': j, 'delay_seconds': delay, 'box_association_frames': votes}
    matched = []
    if reference and predicted:
        ii, jj = linear_sum_assignment(costs)
        matched = [evidence[i, j] for i, j in zip(ii, jj) if costs[i, j] < 1e6]
    count = len(matched)
    record['count_audit'] = {'method': 'Same direction, <=3 seconds difference, >=3 one-to-one box associations at IoU>=0.3 to the published human track ID around the event. Timing errors >1 second are listed separately. No appearance recognition.',
                             'identity_verified_matches': count,
                             'unverified_predictions': len(predicted)-count,
                             'unmatched_references': len(reference)-count,
                             'delayed_or_early_over_one_second': [e for e in matched if abs(e['delay_seconds']) > 1],
                             'matches': matched,
                             'precision': count / len(predicted) if predicted else None,
                             'recall': count / len(reference) if reference else None}


def count_audit_totals(records):
    tp = sum(r['count_audit']['identity_verified_matches'] for r in records)
    extra = sum(r['count_audit']['unverified_predictions'] for r in records)
    missed = sum(r['count_audit']['unmatched_references'] for r in records)
    return {'identity_verified_matches': tp, 'unverified_predictions': extra, 'unmatched_references': missed,
            'precision': tp / (tp + extra) if tp + extra else None,
            'recall': tp / (tp + missed) if tp + missed else None,
            'f1': 2 * tp / (2 * tp + extra + missed) if 2 * tp + extra + missed else 0,
            'timing_errors_over_one_second': sum(len(r['count_audit']['delayed_or_early_over_one_second']) for r in records)}


def validate():
    r = report()
    if r.get('holdout_opened_at_utc'):
        raise RuntimeError('Cannot select or tune after opening this holdout')
    sources = manifest()
    for seq, info in sources['sequences'].items():
        if sha(DATA / info['video']) != info['video_sha256']:
            raise RuntimeError(f'Source changed: {seq}')
    if sha(DATA / 'annotations.zip') != sources['annotation_archive_sha256']:
        raise RuntimeError('Published annotations changed')
    r.setdefault('protocol', {'plan': PLAN, 'source_manifest': manifest(), 'validation_opened_at_utc': stamp(),
                              'reference_notes': manifest()['entrance_references'], 'candidates': profiles(),
                              'training': {'selection': json.loads((TEMP / 'training-selection.json').read_text()),
                                           'results_csv': (TEMP / 'training-run/results.csv').read_text()}})
    write_json(OUTPUT / 'report.json', r)
    for name, profile in r['protocol']['candidates'].items():
        r['runs'].setdefault(name, {})
        for seq in PLAN['validation']:
            if seq not in r['runs'][name]:
                r['runs'][name][seq] = evaluate(seq, profile)
                write_json(OUTPUT / 'report.json', r)
            if 'count_audit' not in r['runs'][name][seq]:
                audit_events(seq, r['runs'][name][seq])
                write_json(OUTPUT / 'report.json', r)
        r.setdefault('validation_totals', {})[name] = totals([r['runs'][name][s] for s in PLAN['validation']])
        write_json(OUTPUT / 'report.json', r)
    r['count_audit_protocol_frozen_at_utc'] = stamp()
    r['validation_count_audit'] = {name: count_audit_totals([r['runs'][name][s] for s in PLAN['validation']]) for name in r['protocol']['candidates']}
    preference = {'doorway-pretrained': 2, 'full-frame': 1, 'doorway-trained': 0}
    strict_selected = max(r['validation_totals'], key=lambda n: (r['validation_totals'][n]['f1'], -r['validation_totals'][n]['extra'], preference.get(n, 0)))
    r['selection_revision_before_holdout'] = {
        'recorded_at_utc': r.get('selection_revision_before_holdout', {}).get('recorded_at_utc', stamp()),
        'initial_rule': 'One-second same-direction/location event F1',
        'initial_winner': strict_selected,
        'revised_rule': 'Identity-verified directional crossing F1 with <=3 seconds timing difference; fewer unverified events breaks ties, then prefer pretrained within a tie. One-second score retained as response-timing diagnostic.',
        'reason': 'Validation inspection found two real entries confirmed about two seconds late. The original rule can match a late crossing to a different nearby person and penalizes valid counts as extra/missed. Identity verification better measures the requested counting task. Revision and all validation outcomes are recorded before opening any holdout model outcomes.'}
    selected = max(r['validation_count_audit'], key=lambda n: (r['validation_count_audit'][n]['f1'], -r['validation_count_audit'][n]['unverified_predictions'], preference.get(n, 0)))
    r['selected'] = selected
    r['frozen_at_utc'] = stamp()
    r['frozen_profile'] = r['protocol']['candidates'][selected]
    r['frozen_sha256'] = {str(path.relative_to(ROOT)): sha(path) for path in [PLAN_PATH, Path(__file__), ROOT / 'experiments/train_retail_entrance.py', ROOT / 'tracker.py', ROOT / 'counting.py', ROOT / 'configs/bytetrack-low-fps.yaml', DATA / 'manifest.json', DATA / 'annotations.zip', ROOT / 'models/retail-entrance-person.pt', ROOT / 'yolo11s.pt']}
    write_json(OUTPUT / 'report.json', r)
    write_json(ROOT / 'configs/retail-entrance-selected.json', r['frozen_profile']['camera'])
    print('VALIDATION_COMPLETE', selected, r['validation_totals'], flush=True)


def refine():
    """Small validation-only calibration sweep; preserve every initial outcome."""
    r = report()
    if r.get('holdout_opened_at_utc'):
        raise RuntimeError('Cannot tune after opening this holdout')
    if 'validation_refinement' not in r:
        candidates = {}
        for base, gate in [('full-frame', .4), ('full-frame', .55), ('doorway-pretrained', .4), ('doorway-pretrained', .25), ('doorway-trained', .4)]:
            profile = copy.deepcopy(r['protocol']['candidates'][base])
            camera = profile['camera']
            camera['line'] = [[205/383, 151/287], [300/383, 151/287]]
            camera.update(deadband_pixels=3, confirm_seconds=.12, track_buffer_seconds=1.2)
            camera['person_tracking']['new_track_thresh'] = gate
            candidates[f'{base}-calibrated-{gate}'] = profile
        r['validation_refinement'] = {'registered_at_utc': stamp(), 'reason': 'Validation shows overlapping people lost by crop and delayed entries from predicted footpoint/band geometry. Test a six-pixel lower counting line with a narrower three-pixel band and two start gates; source reference events are unchanged. Original three candidates and outcomes remain. No holdout detector outcomes have been opened.', 'candidates': candidates}
        r['protocol']['candidates'].update(candidates)
        write_json(OUTPUT / 'report.json', r)
    validate()


def holdout():
    r = report()
    if 'frozen_profile' not in r:
        raise RuntimeError('Select and freeze on validation first')
    for name, digest in r['frozen_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'Frozen input changed: {name}')
    for seq, info in manifest()['sequences'].items():
        if sha(DATA / info['video']) != info['video_sha256']:
            raise RuntimeError(f'Source changed: {seq}')
    r.setdefault('holdout_opened_at_utc', stamp())
    write_json(OUTPUT / 'report.json', r)
    for name in dict.fromkeys(['full-frame', r['selected']]):
        r['runs'].setdefault(name, {})
        for seq in PLAN['holdout']:
            if seq not in r['runs'][name]:
                r['runs'][name][seq] = evaluate(seq, r['protocol']['candidates'][name])
                write_json(OUTPUT / 'report.json', r)
            if 'count_audit' not in r['runs'][name][seq]:
                audit_events(seq, r['runs'][name][seq])
                write_json(OUTPUT / 'report.json', r)
        r.setdefault('holdout_totals', {})[name] = totals([r['runs'][name][s] for s in PLAN['holdout']])
        write_json(OUTPUT / 'report.json', r)
    r['holdout_count_audit'] = {name: count_audit_totals([r['runs'][name][s] for s in PLAN['holdout']]) for name in dict.fromkeys(['full-frame', r['selected']])}
    r['finished_at_utc'] = stamp()
    write_json(OUTPUT / 'report.json', r)
    print('HOLDOUT_COMPLETE', r['holdout_totals'], flush=True)


def gallery():
    import cv2
    import xml.etree.ElementTree as ET
    r = report()
    if not r.get('finished_at_utc'):
        raise RuntimeError('Complete the reserved evaluation before exporting')
    annotations = zipfile.ZipFile(DATA / 'annotations.zip')
    cards = []
    for seq in PLAN['holdout']:
        selected = r['runs'][r['selected']][seq]
        baseline = r['runs']['full-frame'][seq]
        reference = manifest()['entrance_references']['sequences'][seq]['events']
        name = ''.join(('-' + c.lower()) if c.isupper() else c for c in seq).lstrip('-').replace('front', '') + '.mp4'
        target = OUTPUT / name
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), 25, (1536, 636))
        if not writer.isOpened():
            raise RuntimeError('Video encoder unavailable')
        truth_counts, counts = Counter(), [Counter(), Counter()]
        thumb = None
        n = 0
        for n, image in enumerate(decode(seq)):
            for event in reference:
                if event['frame'] == n:
                    truth_counts[event['direction']] += 1
            panes = []
            for side, record in enumerate([baseline, selected]):
                im = image.copy()
                color = (255, 150, 30) if side == 0 else (40, 220, 40)
                for identity, x1, y1, x2, y2, confidence in record['observations'][n]:
                    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                    cv2.putText(im, str(identity).replace('person-', '#'), (int(x1), max(22, int(y1)-2)), 0, .28, color, 1)
                camera = record['profile']['camera']
                points = [tuple(round(p * size) for p, size in zip(point, (383, 287))) for point in camera['line']]
                cv2.line(im, *points, (0, 255, 255), 1)
                if camera.get('detection_crop'):
                    x1, y1, x2, y2 = camera['detection_crop']
                    cv2.rectangle(im, (x1, y1), (x2, y2), (180, 180, 180), 1)
                for event in record['events']:
                    if event['frame'] == n:
                        counts[side][event['direction']] += 1
                im = cv2.resize(im, (768, 576), interpolation=cv2.INTER_LINEAR)
                top = __import__('numpy').zeros((60, 768, 3), dtype='uint8')
                title = 'Full-frame baseline' if side == 0 else 'Selected: ' + r['selected']
                cv2.putText(top, title, (10, 23), 0, .6, color, 1)
                cv2.putText(top, f'Model IN {counts[side]["entry"]} OUT {counts[side]["exit"]} | Reference IN {truth_counts["entry"]} OUT {truth_counts["exit"]} | {n/25:.1f}s', (10, 47), 0, .45, (255, 255, 255), 1)
                panes.append(cv2.vconcat([top, im]))
            combined = cv2.hconcat(panes)
            writer.write(combined)
            if n == (reference[0]['frame'] if reference else selected['frames'] // 2):
                ok, pixels = cv2.imencode('.jpg', cv2.resize(combined, (960, 398)), [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    thumb = base64.b64encode(pixels).decode()
        writer.release()
        capture = cv2.VideoCapture(str(target))
        frames = 0
        while capture.read()[0]:
            frames += 1
        capture.release()
        if frames != selected['frames']:
            raise RuntimeError(f'Incomplete comparison video {seq}: {frames}')
        selected['comparison_video'] = name
        selected['export_decoded_frames'] = frames
        score = selected['scores']
        refs = score['reference_counts']
        pred = score['model_counts']
        audit = selected['count_audit']
        cards.append(f'<article><h2>{html.escape(seq.replace("front", ""))}</h2><p>{frames/25:.1f}s · Reference: {refs.get("entry",0)} in / {refs.get("exit",0)} out. Selected: {pred.get("entry",0)} in / {pred.get("exit",0)} out.</p><p>Strict one-second check: {score["matched"]} matches · {score["missed"]} unmatched references · {score["extra"]} unmatched predictions.</p><p>Identity audit: {audit["identity_verified_matches"]} verified counts · {audit["unmatched_references"]} unmatched references · {audit["unverified_predictions"]} unverified predictions. {len(audit["delayed_or_early_over_one_second"])} verified counts have timing error over one second.</p><a href="{name}"><img alt="Baseline and selected entrance tracking" src="data:image/jpeg;base64,{thumb}"></a><p><a href="{name}">Watch comparison</a> · <a href="../../data/retail-entrance/holdout/{seq}.mpg">Original test footage</a></p></article>')
    write_json(OUTPUT / 'report.json', r)
    rows = []
    for split in ['validation', 'holdout']:
        for name, scores in r[split + '_totals'].items():
            rows.append(f'<tr><td>{split}</td><td>{name}</td><td>{scores["precision"]:.1%}</td><td>{scores["recall"]:.1%}</td><td>{scores["f1"]:.1%}</td><td>{scores["matched"]}</td><td>{scores["missed"]}</td><td>{scores["extra"]}</td></tr>')
    audit_rows = []
    for name, scores in r['holdout_count_audit'].items():
        precision = f'{scores["precision"]:.1%}' if scores['precision'] is not None else 'n/a'
        recall = f'{scores["recall"]:.1%}' if scores['recall'] is not None else 'n/a'
        audit_rows.append(f'<tr><td>{name}</td><td>{precision}</td><td>{recall}</td><td>{scores["identity_verified_matches"]}</td><td>{scores["unmatched_references"]}</td><td>{scores["unverified_predictions"]}</td><td>{scores["timing_errors_over_one_second"]}</td></tr>')
    doc = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Retail entrance pilot</title><style>body{margin:0;background:#eef3f5;color:#17303d;font:16px system-ui}main{max-width:1250px;margin:auto;padding:30px}p,li{line-height:1.6}h1{font-size:34px}.notice{background:#fff1cb;padding:18px;border-radius:12px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:12px;border-bottom:1px solid #dde5e9;text-align:left}th{background:#173944;color:white}.table{overflow:auto}.cards{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{background:white;border-radius:12px;padding:18px;margin-top:20px}h2{font-size:20px}img{width:100%;border-radius:8px}a{color:#006d77}details{margin:24px 0}@media(max-width:700px){main{padding:15px}.cards{display:block}}</style><main><h1>Retail entrance pilot</h1><p>A real shopping-centre doorway, with temporary person IDs and directional entry/exit counts. Six complete takes used for development/validation; six other takes reserved for the final comparison. All original frames retained at 25 FPS. Processing is offline.</p><p class="notice">This is a small same-camera indoor pilot, not proof of 95% accuracy at outdoor malls. The older 384×288 footage includes occlusions, groups, waiting and passers-by. Published human activity intervals and tracks support 37 visually reviewed reference events; 23 are in the reserved test. Reference timing has some uncertainty; matching requires the correct direction within 1 second and 38.4 horizontal pixels. No events were excluded as ambiguous.</p><p><b>Selected on validation:</b> ''' + html.escape(r['selected']) + '''. Compared full-frame pretrained YOLO11s, doorway-focused pretrained YOLO11s, and a doorway-focused YOLO11s trained on 180 development images, plus a small validation-only line/threshold calibration sweep. Every candidate outcome is preserved. All use ByteTrack. The small training experiment uses 60 separate validation images; holdout images are excluded from training and selection.</p><div class="table"><table><tr><th>Split</th><th>Version</th><th>Timely-event precision</th><th>Timely-event recall</th><th>F1</th><th>Matched</th><th>Unmatched reference</th><th>Unmatched prediction</th></tr>''' + ''.join(rows) + '''</table></div><p>The table above retains the original one-second event-matching rule as a timing diagnostic. It did not select the final version. A late count may appear as both an unmatched reference and an unmatched prediction. Precision measures how many reported crossings are correct within that timing rule. Recall measures how many real crossings were counted. F1 combines both. These are event scores, not box-detection accuracy or unique visitors. Exact per-direction and per-clip count errors are in the shared report.</p><h2>Verified crossing counts on the reserved recordings</h2><p>This counting metric checks the published human track ID through box overlap, requires at least three supporting frames, and allows up to three seconds of timing difference. It separates late confirmations from wrong or duplicate counts. It selected the final version on validation. This rule replaced the original one-second rule after validation inspection, before holdout model outcomes; both scores and the change history remain visible. No face or appearance recognition is used.</p><div class="table"><table><tr><th>Version</th><th>Verified-count precision</th><th>Verified-count recall</th><th>Verified</th><th>Unmatched reference</th><th>Unverified prediction</th><th>Timing error over 1s</th></tr>''' + ''.join(audit_rows) + '''</table></div><p>Blue boxes: baseline tracks. Green boxes: selected-version tracks. Yellow: counting boundary. Gray: selected detector crop. Box colors identify versions, not correctness. The displayed resolution is enlarged for review; no new source detail is created.</p><div class="cards">''' + ''.join(cards) + '''</div><details><summary>Sources, method and limitations</summary><p>Data: <a href="https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/">EC Funded CAVIAR project / IST 2001 37540, University of Edinburgh</a>, publicly released under CC BY-SA. Both development and test recordings use the same camera, day and some of the same actors; synchronized corridor-camera copies are excluded. This is not a new-site or night/weather test. Pretrained-model exposure to this public benchmark is unknown.</p><p>Reference events were proposed from human activity intervals and footpoints and visually checked before detector evaluation; there is no second independent human audit. Crossings outside the predeclared finite counting segment remain in scoring. The selected model and settings were frozen before any holdout detector results. A failed result is preserved without retuning on this holdout.</p></details><p><a href="report.json">One shared report: settings, training history, references, observations, event matches and source hashes</a> · <a href="../../data/retail-entrance/manifest.json">Original footage and reference manifest</a></p></main></html>'''
    (OUTPUT / 'review.html').write_text(doc)
    print('REVIEW_COMPLETE', OUTPUT / 'review.html', flush=True)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def acquire():
    DATA.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    previous = json.loads((DATA / 'manifest.json').read_text()) if (DATA / 'manifest.json').exists() else {}
    manifest = {'created_at_utc': previous.get('created_at_utc', stamp()), 'plan': PLAN, 'plan_sha256': sha(PLAN_PATH), 'sequences': {}}
    labels = {}

    def get(item):
        split, seq, label = item
        folder = DATA / split
        folder.mkdir(exist_ok=True)
        urls = {}
        for filename, destination in [(seq + '.mpg', folder / (seq + '.mpg')), (label, TEMP / label)]:
            url = f'https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA2/{seq}/{filename}'
            if not destination.exists():
                cached = TEMP / filename
                if cached.exists() and cached != destination:
                    destination.write_bytes(cached.read_bytes())
                else:
                    with urllib.request.urlopen(url, timeout=90) as response:
                        destination.write_bytes(response.read())
            urls[filename] = url
        video = folder / (seq + '.mpg')
        xml = (TEMP / label).read_bytes()
        print(f'Acquired {split}: {seq} ({video.stat().st_size // 1024} KiB)', flush=True)
        return seq, xml, {'split': split, 'video': str(video.relative_to(DATA)), 'video_sha256': sha(video), 'annotation': seq + '.xml', 'annotation_sha256': hashlib.sha256(xml).hexdigest(), 'urls': urls}

    items = [(split, seq, label) for split in ['development', 'validation', 'holdout'] for seq, label in PLAN[split].items()]
    with ThreadPoolExecutor(max_workers=3) as executor:
        for seq, xml, info in executor.map(get, items):
            labels[seq] = xml
            manifest['sequences'][seq] = info
    with zipfile.ZipFile(DATA / 'annotations.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for seq, xml in labels.items():
            archive.writestr(seq + '.xml', xml)
    manifest['annotation_archive_sha256'] = sha(DATA / 'annotations.zip')
    if previous.get('entrance_references'):
        if any(previous['sequences'][s]['video_sha256'] != i['video_sha256'] or previous['sequences'][s]['annotation_sha256'] != i['annotation_sha256'] for s, i in manifest['sequences'].items()):
            raise RuntimeError('Source changed; existing reviewed references need a new audit')
        manifest['entrance_references'] = previous['entrance_references']
    write_json(DATA / 'manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['acquire', 'validate', 'refine', 'holdout', 'gallery'])
    args = parser.parse_args()
    if args.phase == 'acquire':
        acquire()
    elif args.phase == 'validate':
        validate()
    elif args.phase == 'holdout':
        holdout()
    elif args.phase == 'refine':
        refine()
    elif args.phase == 'gallery':
        gallery()
