"""A disclosed development recalibration of the already inspected zoomed take."""
import base64
import copy
import html
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import confirm_counter as control

OUTPUT = ROOT / 'outputs/zoom-calibration'
CONFIG = ROOT / 'configs/caviar-zoomed-entrance.json'
SEQ = 'WalkByShop1front'


def evaluate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / 'report.json'
    if target.exists():
        raise RuntimeError('Preserve the completed recalibration; use render to regenerate presentation')
    prior = control.report()
    profile = copy.deepcopy(prior['frozen_profile'])
    # Chosen from the opening image's doorway floor edge, before this replay.
    pixels = [[85, 203], [353, 191]]
    profile['camera']['camera_id'] = 'shop-front-zoomed'
    profile['camera']['line'] = [[x/383, y/287] for x, y in pixels]
    control.write(CONFIG, profile['camera'])
    refs = control.manifest()['entrance_references']['sequences'][SEQ]
    r = {
        'sequence': SEQ, 'status': 'development recalibration on an already inspected take',
        'registered_at_utc': control.stamp(),
        'calibration': {'pixel_endpoints': pixels, 'image_frame': 0,
                        'basis': 'Visible upper edge of the dark doorway threshold in the opening source image. No event timestamps or human crossing points used to select endpoints.',
                        'profile': profile},
        'unchanged': ['YOLO weights and predictions', 'ByteTrack settings', 'duplicate suppression',
                      'crossing confirmation settings', 'human references and scoring rules'],
        'references': refs,
        'source': control.manifest()['sequences'][SEQ],
        'previous_report_sha256': control.sha(control.OUTPUT / 'report.json'),
        'implementation_sha256': {p: control.sha(ROOT/p) for p in [
            'counting.py', 'tracker.py', 'detection.py', 'experiments/calibrate_zoom.py',
            'configs/caviar-zoomed-entrance.json', 'yolo11s.pt']},
        'runs': {'previous-calibration': copy.deepcopy(prior['runs']['fixed'][SEQ])},
        'limitations': ['Same indoor camera and known clip; this is not an independent test or outdoor-mall accuracy estimate.',
                        'A camera movement or zoom needs a new calibration. No automatic movement alarm is implemented.',
                        'Stationary shop displays still receive person boxes. Zero unverified crossing events does not mean zero false person detections.',
                        'Three reference crossings cannot establish general accuracy.'],
        'attribution': {'source': 'https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/',
                        'credit': 'EC Funded CAVIAR project/IST 2001 37540, University of Edinburgh; CC BY-SA'},
    }
    control.write(target, r)  # Geometry recorded before opening the replay outcomes.
    for mode in ['pre-fix', 'fixed']:
        r['runs'][mode] = control.evaluate(SEQ, profile, mode)
        control.write(target, r)
    if control.sha(control.OUTPUT/'report.json') != r['previous_report_sha256']:
        raise RuntimeError('The historical report changed')
    r['finished_at_utc'] = control.stamp()
    control.write(target, r)


def render():
    import cv2
    import numpy as np
    target = OUTPUT/'report.json'
    r = json.loads(target.read_text())
    records = [r['runs']['previous-calibration'], r['runs']['fixed']]
    names = ['Previous boundary - repaired counter', 'Doorway boundary - repaired counter']
    video = OUTPUT/'zoomed-doorway.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 25, (1536, 636))
    if not writer.isOpened():
        raise RuntimeError('Video encoder unavailable')
    counts, truth = [Counter(), Counter()], Counter()
    events = [{e['frame']: [] for e in record['events']} for record in records]
    for side, record in enumerate(records):
        for event in record['events']:
            events[side][event['frame']].append(event)
    opening = None
    frames = 0
    try:
        for n, image in enumerate(control.previous.decode(SEQ)):
            for event in r['references']['events']:
                if event['frame'] == n:
                    truth[event['direction']] += 1
            panes = []
            for side, record in enumerate(records):
                im = image.copy()
                color = (255, 160, 30) if side == 0 else (50, 220, 50)
                for identity, x1, y1, x2, y2, confidence in record['observations'][n]:
                    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                    cv2.putText(im, str(identity), (int(x1), max(22, int(y1)-2)), 0, .28, color, 1)
                points = [tuple(round(v*s) for v,s in zip(p, (383,287)))
                          for p in record['profile']['camera']['line']]
                cv2.line(im, *points, (0,255,255), 1)
                for event in events[side].get(n, []):
                    counts[side][event['direction']] += 1
                header = np.zeros((60,768,3), np.uint8)
                cv2.putText(header, names[side], (10,23), 0, .57, color, 1)
                cv2.putText(header, f'IN {counts[side]["entry"]} OUT {counts[side]["exit"]} | Reference IN {truth["entry"]} OUT {truth["exit"]} | {n/25:.1f}s',
                            (10,47), 0, .48, (255,255,255), 1)
                panes.append(cv2.vconcat([header, cv2.resize(im,(768,576))]))
            combined = cv2.hconcat(panes)
            writer.write(combined)
            if n == 0:
                opening = base64.b64encode(cv2.imencode('.jpg', combined)[1]).decode()
            frames += 1
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != frames or frames != records[1]['frames']:
        raise RuntimeError('Video frame count mismatch')
    rows = []
    for key, title in [('previous-calibration','Earlier boundary / repaired counter'),
                       ('pre-fix','Calibrated boundary / old counter'), ('fixed','Calibrated boundary / repaired counter')]:
        record = r['runs'][key]
        a = record['count_audit']
        directions = Counter(e['direction'] for e in record['events'])
        rows.append(f'<tr><td>{title}</td><td>{a["identity_verified_matches"]}/3</td><td>{a["unmatched_references"]}</td><td>{a["unverified_predictions"]}</td><td>{directions["entry"]} / {directions["exit"]}</td></tr>')
    matches = {m['reference_index']: m for m in r['runs']['fixed']['count_audit']['matches']}
    event_rows = []
    for i, e in enumerate(r['references']['events']):
        m = matches.get(i)
        result = f'Verified; {m["delay_seconds"]:+.2f}s' if m else 'Missed'
        event_rows.append(f'<tr><td>{e["frame"]/25:.2f}s</td><td>{html.escape(e["direction"])}</td><td>{result}</td></tr>')
    source = Path(r['source']['video']).relative_to(ROOT)
    doc = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Zoomed doorway calibration</title>
<style>body{{margin:0;background:#eef3f5;color:#17303d;font:16px system-ui}}main{{max-width:1200px;margin:auto;padding:28px}}p{{line-height:1.6}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #ccd8de}}th{{background:#173944;color:white}}video,img{{width:100%;border-radius:10px}}.notice{{padding:18px;background:#fff1cb;border-radius:12px}}.scroll{{overflow:auto}}a{{color:#006d77}}</style>
<main><h1>Zoomed doorway calibration</h1><p class="notice">Development follow-up on the already inspected WalkByShop1front recording. This is not a fresh independent test. The previous frozen results remain unchanged.</p>
<p>The old counting line sat inside the shop after the camera zoom changed. A new sloping boundary follows the visible doorway threshold from pixel (85,203) to (353,191), chosen from the opening image before replay. YOLO detections, ByteTrack, duplicate suppression and reference events remain the same.</p>
<div class="scroll"><table><tr><th>Version</th><th>Verified crossings</th><th>Missed</th><th>Unverified events</th><th>In / out</th></tr>{''.join(rows)}</table></div>
<h2>Same footage, different boundary</h2><p>Both video panes use the repaired counter. Left: earlier boundary. Right: doorway calibration. All {frames} source frames retained at 25 FPS; processing is offline. Yellow marks the counting boundary.</p>
<video controls preload="metadata" poster="data:image/jpeg;base64,{opening}" src="zoomed-doorway.mp4"></video>
<h2>Unchanged reference events</h2><p>Matching requires the same direction within three seconds, supported by at least three one-to-one box associations at IoU ≥ 0.3 to the publisher's anonymous human track. Timing relative to the approximate reference is shown below.</p>
<table><tr><th>Reference time</th><th>Direction</th><th>Repaired counter</th></tr>{''.join(event_rows)}</table>
<h2>Calibration chosen from the opening frame</h2><img alt="Previous line and new doorway boundary in opening frame" src="data:image/jpeg;base64,{opening}">
<p>Stationary shop displays still receive person boxes in this footage. Zero unverified crossing events does not mean zero false person detections; filtering every stationary box would also remove real people standing still.</p>
<p>Three crossings at one indoor camera cannot establish performance at outdoor malls or parking lots. Each camera needs its own boundary; movement or zoom changes require recalibration. No automatic camera-movement alarm is implemented.</p>
<p><a href="report.json">Shared report</a> · <a href="../../{source.as_posix()}">Original footage</a> · <a href="../counter-confirmation/review.html">Previous independent test</a> · <a href="../../configs/caviar-zoomed-entrance.json">Zoomed camera profile</a></p>
<p>Source: <a href="https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/">EC Funded CAVIAR project / IST 2001 37540, University of Edinburgh</a>, CC BY-SA. IDs are temporary within the clip; no appearance recognition.</p></main></html>'''
    (OUTPUT/'review.html').write_text(doc)
    r['export'] = {'video':video.name, 'decoded_frames':decoded, 'review':'review.html'}
    r['historical_report_unchanged'] = control.sha(control.OUTPUT/'report.json') == r['previous_report_sha256']
    control.write(target, r)
    print(json.dumps({key: record['count_audit'] for key,record in r['runs'].items()}, indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['render']:
        render()
    else:
        evaluate()
        render()
