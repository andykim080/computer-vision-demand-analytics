"""Present frozen entrance results; does not select settings or recompute scores."""
import base64
import html
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/retail-entrance'


def percent(value):
    return '—' if value is None else f'{value:.1%}'


def main():
    report = json.loads((OUTPUT / 'report.json').read_text())
    if not report.get('finished_at_utc'):
        raise RuntimeError('The reserved test must finish before presenting results')
    selected = report['selected']
    final = report['holdout_count_audit'][selected]
    baseline = report['holdout_count_audit']['full-frame']
    titles = {
        'TwoEnterShop2front': 'Group leaving near the doorway edge',
        'TwoEnterShop3front': 'Couples entering and leaving',
        'OneStopMoveEnter2front': 'Repeated entry and exit',
        'OneLeaveShopReenter2front': 'Leaving, then returning',
        'OneShopOneWait2front': 'Shopping, waiting and passing groups',
        'ThreePastShop2front': 'Passers-by only',
    }
    summaries = []
    for name, score in [('Pretrained baseline', baseline), ('Selected setup', final)]:
        summaries.append(f'<tr><th>{name}</th><td>{percent(score["precision"])}</td><td>{percent(score["recall"])}</td><td>{score["identity_verified_matches"]}</td><td>{score["unmatched_references"]}</td><td>{score["unverified_predictions"]}</td><td>{score["timing_errors_over_one_second"]}</td></tr>')
    cards = []
    for seq in report['protocol']['plan']['holdout']:
        record = report['runs'][selected][seq]
        video = record['comparison_video']
        refs = report['protocol']['reference_notes']['sequences'][seq]['events']
        capture = cv2.VideoCapture(str(OUTPUT / video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, refs[0]['frame'] if refs else record['frames'] // 2)
        ok, image = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f'Cannot read comparison preview: {video}')
        ok, pixels = cv2.imencode('.jpg', cv2.resize(image, (960, 398)), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError('Cannot encode embedded preview')
        poster = base64.b64encode(pixels).decode()
        reference_counts = record['scores']['reference_counts']
        predicted_counts = record['scores']['model_counts']
        score = record['count_audit']
        description = f'Reference: {reference_counts.get("entry",0)} in / {reference_counts.get("exit",0)} out. Selected: {predicted_counts.get("entry",0)} in / {predicted_counts.get("exit",0)} out.'
        cards.append(f'<article><h2>{titles[seq]}</h2><p>{record["frames"]/25:.1f} seconds · {description}</p><a href="{video}"><img src="data:image/jpeg;base64,{poster}" alt="Side-by-side tracker comparison at this entrance"></a><p><b>{score["identity_verified_matches"]} verified</b> · {score["unmatched_references"]} unmatched references · {score["unverified_predictions"]} unverified predictions</p><p><a href="{video}">Watch comparison</a> · <a href="../../data/retail-entrance/holdout/{seq}.mpg">Original footage</a></p></article>')
    trials = []
    for name, scores in report['validation_count_audit'].items():
        timing = report['validation_totals'][name]
        trials.append(f'<tr><td>{html.escape(name)}</td><td>{percent(scores["precision"])}</td><td>{percent(scores["recall"])}</td><td>{scores["identity_verified_matches"]}/5</td><td>{percent(timing["f1"])}</td></tr>')
    threshold = final['precision'] is not None and final['recall'] is not None and final['precision'] >= .95 and final['recall'] >= .95
    status = 'This small sample meets the 95% precision/recall threshold; it does not establish dependable 95% accuracy.' if threshold else 'The reserved test does not meet the 95% precision/recall target. Further development needs a new reserved test.'
    if final['f1'] < baseline['f1']:
        status += ' The trained trial regressed; keep the pretrained baseline for further development.'
    timing_rows = []
    for name in ['full-frame', selected]:
        scores = report['holdout_totals'][name]
        timing_rows.append(f'<tr><td>{html.escape(name)}</td><td>{percent(scores["precision"])}</td><td>{percent(scores["recall"])}</td><td>{percent(scores["f1"])}</td></tr>')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Retail entrance · test review</title><style>body{margin:0;background:#eef3f5;color:#18333d;font:16px system-ui}main{max-width:1250px;margin:auto;padding:32px}h1{font-size:36px;margin-bottom:8px}p,li{line-height:1.6;color:#46606b}.notice{background:#fff1ce;padding:18px;border-radius:12px}.table{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:12px;text-align:left;border-bottom:1px solid #dde6ea;white-space:nowrap}thead th{background:#173d49;color:white}.cards{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{background:white;padding:18px;border-radius:12px;margin:8px 0}article h2{font-size:20px;margin:0}img{width:100%;border-radius:8px}a{color:#006e78}details{background:white;padding:18px;border-radius:12px;margin:22px 0}summary{cursor:pointer;font-weight:650}code{font-size:13px}@media(max-width:720px){main{padding:16px}.cards{display:block}h1{font-size:29px}}</style><main><h1>Retail entrance test</h1><p>One shopping-centre doorway · 6 reserved recordings · 23 reference crossings · original 25 FPS · offline processing</p><p class="notice">''' + status + ''' These older 384×288 indoor recordings share the camera, day and some actors with development footage. They do not validate outdoor malls, nighttime, weather or cross-camera unique visitors.</p><h2>Verified crossing counts</h2><div class="table"><table><thead><tr><th>Version</th><th>Precision</th><th>Recall</th><th>Verified</th><th>Unmatched reference</th><th>Unverified prediction</th><th>Timing error &gt;1s</th></tr></thead><tbody>''' + ''.join(summaries) + '''</tbody></table></div><p><b>Precision:</b> the share of reported crossings verified against a real reference event. <b>Recall:</b> the share of reference crossings counted. Matching requires the correct direction, at least three frames of box overlap with the published human track ID, and timing within three seconds. Unverified predictions may include false, duplicate or very late counts. No face or appearance recognition is used.</p><p>The trained detector was selected on five validation events before these test results were opened. The original pretrained baseline verified four of five validation events; the selected setup verified all five, with three late confirmations. The review below preserves failures from the independent takes.</p><h2>Watch the six test comparisons</h2><p>Left / blue: pretrained baseline. Right / green: selected setup. Yellow: each version’s counting line. Gray: selected detector crop. Box colors identify the version, not correctness. The videos are enlarged for review; their source detail remains 384×288.</p><div class="cards">''' + ''.join(cards) + '''</div><details><summary>What was trained and how the test was kept separate</summary><p>Three complete takes supplied 180 training images. Three other takes supplied 60 training-validation images and the five validation crossings used for configuration selection. Six additional takes supplied the final test. No synchronized corridor-view copies or older local webcam clips are part of this test. YOLO11s was trained for eight epochs with early backbone layers frozen; its separate checkpoint is <code>models/retail-entrance-person.pt</code>. The library’s YOLO/ByteTrack source and earlier camera profiles were not modified.</p><p>Eight configurations were compared on validation, including cropping, a small line/threshold sweep and trained weights. The source references stayed unchanged. Settings, code and source hashes were frozen before test detector outcomes; no changes were made to improve these test scores.</p><p>The 37 source reference events were proposed from published human activity intervals and tracks, then visually checked in source frames before detector evaluation. All events were retained, including crossings outside the finite counting segment. There is no second independent human annotation audit. Public pretrained-model exposure to CAVIAR is unknown.</p></details><details><summary>All validation results and the timing-score revision</summary><p>The initial one-second/direction/location rule could pair a late event with a different nearby person. Before opening any test detector results, selection was revised to identity-verified crossing F1 with three seconds of timing tolerance. The original score remains below as a stricter timing diagnostic. Both rules, choices, reasons and timestamps are preserved in the report.</p><div class="table"><table><thead><tr><th>Validation version</th><th>Count precision</th><th>Count recall</th><th>Verified</th><th>Original 1s F1</th></tr></thead><tbody>''' + ''.join(trials) + '''</tbody></table></div><h3>Original one-second scores on the reserved test</h3><div class="table"><table><thead><tr><th>Version</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>''' + ''.join(timing_rows) + '''</tbody></table></div></details><p>Source: <a href="https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/">EC Funded CAVIAR project / IST 2001 37540, University of Edinburgh</a>, publicly released under CC BY-SA. Activities include scripted scenarios and ordinary passers-by.</p><p><a href="report.json">One shared report: all settings, events, predictions, training history and source audit</a> · <a href="../../data/retail-entrance/manifest.json">Original footage and reference manifest</a></p></main></html>'''
    (OUTPUT / 'review.html').write_text(document)
    print(OUTPUT / 'review.html')


if __name__ == '__main__':
    main()
