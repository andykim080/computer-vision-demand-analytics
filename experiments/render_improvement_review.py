"""Render the frozen second entrance experiment without altering its scores."""
import base64
import html
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/entrance-improvement'


def percent(value):
    return '—' if value is None else f'{value:.1%}'


def main():
    r = json.loads((OUTPUT/'report.json').read_text())
    if not r.get('finished_at_utc'):
        raise RuntimeError('Complete the reserved evaluation first')
    selected = r['selected']
    names = list(dict.fromkeys(['full-frame',selected]))
    reference_count = sum(len(r['protocol']['reference_notes']['sequences'][s]['events']) for s in r['protocol']['plan']['holdout'])
    final = r['holdout_count_audit'][selected]
    target_met = (final['precision'] or 0) >= .95 and (final['recall'] or 0) >= .95
    outcome = 'This sample meets the 95% precision/recall target; it is too small and narrow to establish dependable 95% performance.' if target_met else 'This test remains below the 95% precision/recall target. The failures remain visible; further tuning requires another reserved test.'
    summary_rows = []
    for name in names:
        score = r['holdout_count_audit'][name]
        summary_rows.append(f'<tr><th>{"Earlier baseline" if name=="full-frame" else "Validation-selected fix"}</th><td>{percent(score["precision"])}</td><td>{percent(score["recall"])}</td><td>{score["identity_verified_matches"]}/{reference_count}</td><td>{score["unmatched_references"]}</td><td>{score["unverified_predictions"]}</td><td>{score["timing_errors_over_one_second"]}</td></tr>')
    cards = []
    titles = {'OneLeaveShop1front':'Single person leaving · take 1','OneLeaveShop2front':'Single person leaving · take 2',
              'OneLeaveShopReenter1front':'Leaving and re-entering','ShopAssistant1front':'Customers meeting inside and leaving',
              'TwoLeaveShop1front':'Pair leaving · take 1','TwoLeaveShop2front':'Pair leaving · take 2',
              'OneStopNoEnter1front':'Stopping outside and walking past','OneStopNoEnter2front':'Passing, entering and returning'}
    for seq in r['protocol']['plan']['holdout']:
        record = r['runs'][selected][seq]
        video = record['comparison_video']
        references = r['protocol']['reference_notes']['sequences'][seq]['events']
        cap = cv2.VideoCapture(str(OUTPUT/video))
        cap.set(cv2.CAP_PROP_POS_FRAMES,references[0]['frame'] if references else record['frames']//2)
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f'Cannot read {video}')
        ok, pixels = cv2.imencode('.jpg',cv2.resize(image,(960,398)),[cv2.IMWRITE_JPEG_QUALITY,80])
        if not ok:
            raise RuntimeError('Cannot encode preview')
        encoded = base64.b64encode(pixels).decode()
        score = record['count_audit']
        expected = record['scores']['reference_counts']
        predicted = record['scores']['model_counts']
        cards.append(f'<article><h2>{html.escape(titles[seq])}</h2><p>{record["frames"]/25:.1f}s · Reference {expected.get("entry",0)} in / {expected.get("exit",0)} out · Selected {predicted.get("entry",0)} in / {predicted.get("exit",0)} out</p><a href="{video}"><img alt="Baseline and frozen selected tracking comparison" src="data:image/jpeg;base64,{encoded}"></a><p><b>{score["identity_verified_matches"]} verified</b> · {score["unmatched_references"]} unmatched references · {score["unverified_predictions"]} unverified predictions</p><p><a href="{video}">Watch comparison</a> · <a href="../../data/entrance-improvement/holdout/{seq}.mpg">Original test footage</a></p></article>')
    trial_rows = []
    for name, score in r['validation_count_audit'].items():
        dev = r['development_count_audit'][name]
        trial_rows.append(f'<tr><td>{html.escape(name)}</td><td>{percent(score["precision"])}</td><td>{percent(score["recall"])}</td><td>{percent(score["f1"])}</td><td>{percent(dev["f1"])}</td></tr>')
    seconds = sum(r['runs'][selected][s]['frames'] for s in r['protocol']['plan']['holdout'])/25
    config = json.dumps(r['frozen_profile']['camera'],indent=2)
    doc = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Doorway tracking · second test</title><style>body{margin:0;background:#eef3f5;color:#19343e;font:16px system-ui}main{max-width:1250px;margin:auto;padding:30px}p,li{line-height:1.6;color:#47616c}h1{font-size:34px}.notice{background:#fff1cf;padding:18px;border-radius:12px}.table{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:12px;border-bottom:1px solid #dfe7ea;text-align:left;white-space:nowrap}thead th{background:#173c48;color:white}.cards{display:grid;grid-template-columns:1fr 1fr;gap:20px}article,details{background:white;padding:18px;border-radius:12px;margin:18px 0}article h2{font-size:19px;margin:0}img{width:100%;border-radius:8px}summary{cursor:pointer;font-weight:650}a{color:#006e78}pre{overflow:auto;font-size:13px}@media(max-width:720px){main{padding:15px}.cards{display:block}}</style><main><h1>Doorway coverage and duplicate-track fixes</h1><p>8 newly reserved takes · '''+str(reference_count)+''' reference crossings · '''+f'{seconds/60:.1f}'+''' minutes · original 25 FPS · offline processing</p><p class="notice">'''+outcome+''' All takes use the same older 384×288 indoor camera, recording day and some actors. This is not validation of a modern outdoor mall, darkness, weather or cross-camera unique visitors.</p><h2>Results on the new reserved test</h2><div class="table"><table><thead><tr><th>Version</th><th>Precision</th><th>Recall</th><th>Verified crossings</th><th>Unmatched references</th><th>Unverified predictions</th><th>Timing error &gt;1s</th></tr></thead><tbody>'''+''.join(summary_rows)+'''</tbody></table></div><p>The scoring rule is unchanged from the previous experiment: correct direction, at least three one-to-one box associations to the published human track ID at IoU ≥0.3, and timing within three seconds. Unverified predictions can include duplicates, false events, partial-box association failures or very late counts. One-second timing scores are retained separately in the report. No appearance recognition is used.</p><p><b>Selected before opening this test:</b> '''+html.escape(selected)+'''. Both versions use the same pretrained YOLO11s/640 detections. No model weights were trained. The candidates expand the counting segment across the doorway, use a narrower deadband and test person-only suppression of overlapping or top-aligned nested body boxes. No stationary-object rejection is used.</p><h2>Watch the eight comparisons</h2><p>Left / blue: earlier baseline. Right / green: frozen selected version. Yellow: each counting boundary. Box colors identify versions, not correctness. Videos are enlarged for review without adding source detail.</p><div class="cards">'''+''.join(cards)+'''</div><details><summary>Development, validation and source separation</summary><p>All 12 prior CAVIAR takes—including the former test—are now declared development data, totaling 37 reference events. Three previously unused takes supply six validation events. Eight other complete takes were reserved in advance. Synchronized corridor-camera copies are excluded. Settings and source/code hashes were frozen before even opening the new holdout activity labels, then those reference events were visually reviewed before detector outcomes. The prior experiment remains intact.</p><p>Hand-labeled source tracks and activity intervals propose reference events; source frames are visually checked. There is no second independent human annotation audit. All references are retained, including those outside a proposed counting boundary. Small samples and public-model training overlap limit the conclusions.</p></details><details><summary>All candidate outcomes and selected camera settings</summary><div class="table"><table><thead><tr><th>Version</th><th>Validation precision</th><th>Validation recall</th><th>Validation F1</th><th>Development F1</th></tr></thead><tbody>'''+''.join(trial_rows)+'''</tbody></table></div><p>Highest validation F1 selects the version; development F1 breaks ties, followed by fewer unverified validation events and simpler rules.</p><pre>'''+html.escape(config)+'''</pre></details><p>Source: <a href="https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/">EC Funded CAVIAR project / IST 2001 37540, University of Edinburgh</a>, publicly released CC BY-SA footage and hand annotations.</p><p><a href="report.json">One shared report</a> · <a href="../../data/entrance-improvement/manifest.json">Sources and reviewed references</a> · <a href="../retail-entrance/review.html">Previous experiment</a></p></main></html>'''
    (OUTPUT/'review.html').write_text(doc)
    print(OUTPUT/'review.html')


if __name__ == '__main__':
    main()
