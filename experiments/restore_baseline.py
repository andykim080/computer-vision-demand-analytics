"""Restore pretrained retail settings, retain duplicate filtering and zoom calibration."""
import base64
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import confirm_counter as control

OUTPUT = ROOT/'outputs/restored-baseline'
STANDARD = ROOT/'configs/retail-baseline.json'
ZOOMED = ROOT/'configs/retail-baseline-zoomed.json'
ZOOM_SEQUENCE = 'WalkByShop1front'


def evaluate():
    target = OUTPUT/'report.json'
    if target.exists():
        raise RuntimeError('Preserve this completed replay; use render for presentation only')
    old = json.loads((ROOT/'outputs/retail-entrance/report.json').read_text())
    baseline = copy.deepcopy(old['protocol']['candidates']['full-frame'])
    restored = copy.deepcopy(baseline)
    restored['camera']['person_suppression'] = {'iou': .6, 'nested_height_ratio': .7}
    zoomed = copy.deepcopy(restored)
    zoom_camera = json.loads((ROOT/'configs/caviar-zoomed-entrance.json').read_text())
    zoomed['camera'].update(camera_id='shop-front-zoomed', line=zoom_camera['line'])
    control.write(STANDARD, restored['camera'])
    control.write(ZOOMED, zoomed['camera'])
    sequences = list(old['protocol']['plan']['holdout'])
    current = control.manifest()
    original_manifest = json.loads((ROOT/'data/retail-entrance/manifest.json').read_text())
    for seq in sequences:
        assert current['entrance_references']['sequences'][seq]['events'] == original_manifest['entrance_references']['sequences'][seq]['events']
    r = {'registered_at_utc': control.stamp(),
         'status': 'Requested restoration, replayed on previously inspected recordings; not an independent test',
         'profiles': {'historical-baseline': baseline, 'restored': restored, 'zoomed': zoomed},
         'historical_baseline_totals': old['holdout_count_audit']['full-frame'],
         'sequences': sequences, 'zoom_sequence': ZOOM_SEQUENCE,
         'references': {s: current['entrance_references']['sequences'][s] for s in sequences+[ZOOM_SEQUENCE]},
         'sources': {s: current['sequences'][s] for s in sequences+[ZOOM_SEQUENCE]},
         'historical_report_sha256': {p: control.sha(ROOT/p) for p in [
             'outputs/retail-entrance/report.json', 'outputs/entrance-improvement/report.json',
             'outputs/counter-confirmation/report.json', 'outputs/zoom-calibration/report.json']},
         'runs': {'historical-baseline': {s: copy.deepcopy(old['runs']['full-frame'][s]) for s in sequences},
                  'current-counter-only': {}, 'restored': {}},
         'retained': ['person-only duplicate-box suppression', 'corrected counting boundary for zoomed camera',
                      'crossing-confirmation bug repair'],
         'restored': ['pretrained YOLO11s / 640, full image, confidence 0.1, IoU 0.7',
                      'person new-track threshold 0.55; ByteTrack buffer 30 frames',
                      'standard camera line x=205..300 at y=145; deadband 8 pixels; confirmation 3 frames'],
         'limitations': ['Known same-camera clips; old precision/recall does not transfer automatically to this combination.',
                         'Crossing metrics allow three seconds and anonymous reference-box identity support, not detection accuracy.',
                         'Stationary shop displays may retain false person boxes; no motion-only filtering.',
                         'No outdoor-property or vehicle accuracy claim; camera geometry must be calibrated for each view.']}
    control.write(target, r)
    for key, profile in [('current-counter-only', baseline), ('restored', restored)]:
        for seq in sequences:
            r['runs'][key][seq] = control.evaluate(seq, profile, 'fixed')
            control.write(target, r)
    r['runs']['restored'][ZOOM_SEQUENCE] = control.evaluate(ZOOM_SEQUENCE, zoomed, 'fixed')
    r['runs']['historical-baseline'][ZOOM_SEQUENCE] = copy.deepcopy(control.report()['runs']['fixed'][ZOOM_SEQUENCE])
    r['totals'] = {key: control.previous.count_audit_totals([runs[s] for s in sequences])
                   for key, runs in r['runs'].items()}
    r['zoom_totals'] = control.previous.count_audit_totals([r['runs']['restored'][ZOOM_SEQUENCE]])
    r['finished_at_utc'] = control.stamp()
    for path, digest in r['historical_report_sha256'].items():
        assert control.sha(ROOT/path) == digest, path
    r['historical_reports_unchanged'] = True
    control.write(target, r)
    print('RESTORATION_RESULTS', json.dumps({'six_original_clips': r['totals'], 'zoomed_clip': r['zoom_totals']}), flush=True)


def render():
    import cv2
    import numpy as np
    target = OUTPUT/'report.json'
    r = json.loads(target.read_text())
    cards = []
    for seq in r['sequences']+[ZOOM_SEQUENCE]:
        old = r['runs']['historical-baseline'][seq]
        new = r['runs']['restored'][seq]
        filename = ''.join(('-'+c.lower()) if c.isupper() else c for c in seq).lstrip('-').replace('front','')+'.mp4'
        video = OUTPUT/filename
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'),25,(1536,636))
        if not writer.isOpened():
            raise RuntimeError('Video encoder unavailable')
        counts = [Counter(),Counter()]
        event_maps = []
        for rec in [old,new]:
            mapping = {}
            for e in rec['events']:
                mapping.setdefault(e['frame'],[]).append(e)
            event_maps.append(mapping)
        reference = r['references'][seq]['events']
        preview = None
        frames = 0
        try:
            for n, image in enumerate(control.previous.decode(seq)):
                panes = []
                for side, rec in enumerate([old,new]):
                    im = image.copy()
                    color = (255,160,30) if side==0 else (50,220,50)
                    for identity,x1,y1,x2,y2,conf in rec['observations'][n]:
                        cv2.rectangle(im,(round(x1),round(y1)),(round(x2),round(y2)),color,1)
                        cv2.putText(im,str(identity),(round(x1),max(22,round(y1)-2)),0,.28,color,1)
                    points = [tuple(round(v*s) for v,s in zip(p,(383,287))) for p in rec['profile']['camera']['line']]
                    cv2.line(im,*points,(0,255,255),1)
                    for e in event_maps[side].get(n,[]):
                        counts[side][e['direction']] += 1
                    header = np.zeros((60,768,3),np.uint8)
                    title = ('Previous incorrect zoom boundary' if seq==ZOOM_SEQUENCE else 'Historical pretrained baseline') if side==0 else 'Restored baseline + duplicate filtering'
                    cv2.putText(header,title,(10,23),0,.54,color,1)
                    cv2.putText(header,f'IN {counts[side]["entry"]} OUT {counts[side]["exit"]} | {n/25:.1f}s',(10,47),0,.5,(255,255,255),1)
                    panes.append(cv2.vconcat([header,cv2.resize(im,(768,576))]))
                combined = cv2.hconcat(panes)
                writer.write(combined)
                if n==(reference[0]['frame'] if reference else new['frames']//2):
                    preview = base64.b64encode(cv2.imencode('.jpg',cv2.resize(combined,(960,398)))[1]).decode()
                frames += 1
        finally:
            writer.release()
        cap = cv2.VideoCapture(str(video)); decoded = 0
        while cap.read()[0]:
            decoded += 1
        cap.release()
        assert decoded == frames == new['frames'], seq
        new['export'] = {'video':filename,'decoded_frames':decoded}
        a = new['count_audit']
        original = Path(r['sources'][seq]['video']).relative_to(ROOT)
        cards.append(f'<article><h2>{seq.replace("front","")}</h2><p>{a["identity_verified_matches"]}/{len(reference)} verified crossings · {a["unverified_predictions"]} unverified events · {frames/25:.1f}s</p><video controls preload="none" poster="data:image/jpeg;base64,{preview}" src="{filename}"></video><p><a href="../../{original.as_posix()}">Original footage</a></p></article>')
        control.write(target,r)
        print('EXPORTED', seq, decoded, flush=True)
    rows = []
    for key,title in [('historical-baseline','Historical baseline'),('current-counter-only','Baseline + counting repair'),('restored','Restored baseline + repair + duplicate filter')]:
        a = r['totals'][key]
        rows.append(f'<tr><td>{title}</td><td>{a["identity_verified_matches"]}/23</td><td>{a["precision"]:.1%}</td><td>{a["recall"]:.1%}</td><td>{a["unverified_predictions"]}</td></tr>')
    z = r['zoom_totals']
    doc = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Restored pretrained baseline</title><style>body{{margin:0;background:#eef3f5;color:#17303d;font:16px system-ui}}main{{max-width:1250px;margin:auto;padding:28px}}p{{line-height:1.6}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #cbd8df}}th{{background:#173944;color:white}}.notice{{background:#fff1cb;padding:18px;border-radius:12px}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}article{{background:white;padding:16px;border-radius:12px;margin:16px 0}}video{{width:100%;border-radius:8px}}a{{color:#006d77}}.scroll{{overflow:auto}}@media(max-width:750px){{.cards{{display:block}}}}</style>
<main><h1>Restored pretrained baseline</h1><p>YOLO11s / 640, full-frame detections, the original ByteTrack gates and counting settings. Person-only duplicate filtering and the counting-confirmation repair remain enabled. The zoomed camera has a separate corrected doorway boundary.</p>
<p class="notice">This replays six previously inspected recordings with the same 23 reference crossings. These are development comparisons, not fresh independent accuracy estimates. Historical reports remain unchanged.</p>
<div class="scroll"><table><tr><th>Six original recordings</th><th>Verified</th><th>Precision</th><th>Recall</th><th>Unverified events</th></tr>{''.join(rows)}</table></div>
<p>The separately calibrated zoomed recording verifies {z['identity_verified_matches']}/3 crossings, with {z['unverified_predictions']} unverified events. It is not pooled into the original 23-event score.</p>
<p>Verification requires the same direction within three seconds, with at least three one-to-one box associations at IoU ≥ 0.3 to the publisher's anonymous human track. Strict one-second results and event details remain in the shared report.</p>
<h2>Active settings</h2><p>Standard view: <a href="../../configs/retail-baseline.json">retail-baseline.json</a>. Zoomed view: <a href="../../configs/retail-baseline-zoomed.json">retail-baseline-zoomed.json</a>. Camera profiles are view-specific; the tracker does not automatically select a zoomed calibration.</p>
<h2>Compare the recordings</h2><p>Left: historical baseline (zoom clip: earlier wrong boundary). Right: restored settings with duplicate filtering, repaired counter and the camera's appropriate line. All source frames retained at 25 FPS; processing is offline.</p><div class="cards">{''.join(cards)}</div>
<p>False person boxes on stationary shop displays can remain. Counting precision is not detector precision. The footage does not establish performance at outdoor malls, new properties or overlapping cameras.</p>
<p><a href="report.json">Shared settings, observations, references and scores</a> · <a href="../retail-entrance/review.html">Historical training comparison</a> · <a href="../zoom-calibration/review.html">Earlier zoom calibration follow-up</a></p>
<p>Source: <a href="https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/">EC Funded CAVIAR project / IST 2001 37540, University of Edinburgh</a>, CC BY-SA.</p></main></html>'''
    (OUTPUT/'review.html').write_text(doc)
    print(OUTPUT/'review.html', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['render']:
        render()
    else:
        evaluate()
        render()
