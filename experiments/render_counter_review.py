"""Present the isolated counter repair, including the predeclared zoom stress test."""
import base64
import html
import json
from pathlib import Path

import cv2

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/counter-confirmation'


def pct(value):
    return '—' if value is None else f'{value:.1%}'


def aggregate(records):
    tp=sum(v['count_audit']['identity_verified_matches'] for v in records)
    fp=sum(v['count_audit']['unverified_predictions'] for v in records)
    fn=sum(v['count_audit']['unmatched_references'] for v in records)
    return tp,fp,fn,tp/(tp+fp) if tp+fp else None,tp/(tp+fn) if tp+fn else None


def main():
    r=json.loads((OUTPUT/'report.json').read_text())
    if not r.get('finished_at_utc'):
        raise RuntimeError('Complete the reserved test first')
    groups={'All three reserved takes':list(r['protocol']['plan']['holdout']),
            'Matching camera framing':r['pre_test_camera_review']['same_framing'],
            'Zoom-changed stress test':r['pre_test_camera_review']['zoom_changed_stress']}
    rows=[]
    for scope,sequences in groups.items():
        for mode in ['pre-fix','fixed']:
            tp,fp,fn,precision,recall=aggregate([r['runs'][mode][s] for s in sequences])
            rows.append(f'<tr><td>{scope}</td><td>{"Before repair" if mode=="pre-fix" else "After repair"}</td><td>{tp}/{tp+fn}</td><td>{pct(precision)}</td><td>{pct(recall)}</td><td>{fp}</td></tr>')
    cards=[]
    titles={'WalkByShop1front':'Zoom-changed camera · stress test','OneStopMoveNoEnter1front':'Entering, browsing and leaving','ShopAssistant2front':'Shop visitors and overlapping groups'}
    for seq in r['protocol']['plan']['holdout']:
        rec=r['runs']['fixed'][seq]
        video=rec['comparison_video']
        refs=r['protocol']['reference_notes']['sequences'][seq]['events']
        cap=cv2.VideoCapture(str(OUTPUT/video));cap.set(cv2.CAP_PROP_POS_FRAMES,refs[0]['frame'] if refs else rec['frames']//2);ok,im=cap.read();cap.release()
        if not ok:raise RuntimeError(f'Cannot read {video}')
        ok,b=cv2.imencode('.jpg',cv2.resize(im,(960,398)),[cv2.IMWRITE_JPEG_QUALITY,80])
        if not ok:raise RuntimeError('Cannot encode preview')
        encoded=base64.b64encode(b).decode();score=rec['count_audit'];expected=rec['scores']['reference_counts'];predicted=rec['scores']['model_counts']
        cards.append(f'<article><h2>{titles[seq]}</h2><p>{rec["frames"]/25:.1f}s · Reference {expected.get("entry",0)} in / {expected.get("exit",0)} out · After repair {predicted.get("entry",0)} in / {predicted.get("exit",0)} out</p><a href="{video}"><img alt="The same detector tracks with before/after crossing confirmation" src="data:image/jpeg;base64,{encoded}"></a><p><b>{score["identity_verified_matches"]} verified</b> · {score["unmatched_references"]} unmatched references · {score["unverified_predictions"]} unverified predictions</p><p><a href="{video}">Watch comparison</a> · <a href="../../data/counter-confirmation/holdout/{seq}.mpg">Original footage</a></p></article>')
    dev=r['development_count_audit'];devtext=f'{dev["pre-fix"]["identity_verified_matches"]}/59 before → {dev["fixed"]["identity_verified_matches"]}/59 after; {dev["fixed"]["unverified_predictions"]} unverified events after repair.'
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Crossing confirmation repair</title><style>body{margin:0;background:#eef3f5;color:#19353f;font:16px system-ui}main{max-width:1250px;margin:auto;padding:30px}h1{font-size:34px}p,li{line-height:1.6;color:#46616c}.notice{background:#fff0cb;padding:18px;border-radius:12px}.table{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:12px;border-bottom:1px solid #dde6ea;text-align:left;white-space:nowrap}thead th{background:#183c48;color:white}.cards{display:grid;grid-template-columns:1fr 1fr;gap:20px}article,details{background:white;padding:18px;border-radius:12px;margin:18px 0}h2{font-size:21px}img{width:100%;border-radius:8px}a{color:#006d78}summary{font-weight:650;cursor:pointer}@media(max-width:720px){main{padding:15px}.cards{display:block}}</style><main><h1>Crossing confirmation repair</h1><p>The same YOLO11s detections, ByteTrack tracks, doorway geometry and duplicate suppression. Only the crossing-confirmation code differs. Original 25 FPS; offline processing.</p><p class="notice">These small tests do not establish dependable 95% accuracy. One reserved recording has a much tighter zoom and an incompatible doorway calibration. That condition and the matching-framing subset were recorded before detector outcomes. The zoomed clip remains visible in the complete test; it is not silently excluded.</p><h2>Fresh reserved test results</h2><div class="table"><table><thead><tr><th>Scope</th><th>Counter</th><th>Verified crossings</th><th>Precision</th><th>Recall</th><th>Unverified events</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div><p>Matching requires the correct direction, at least three one-to-one box associations with the published human track ID at IoU ≥0.3, and timing within three seconds. Unverified events can include false/duplicate counts, partial-box association failures or very late confirmations. One-second timing results remain in the shared report.</p><h2>What was fixed</h2><p>A valid crossing could be forgotten if an unconfirmed destination observation briefly returned to the line’s tolerance band. The repair retains that crossing until the destination side is confirmed, cancels it on a return to the original side, and resets it after a long gap. Initial observations still do not count as an entry. The issue is reproduced and covered by regression tests.</p><p>Development replay across all 23 previously used takes: '''+devtext+''' These are development outcomes. The eight earlier test takes were explicitly reclassified as development after the bug was diagnosed.</p><h2>Watch the three comparisons</h2><p>Left / blue: saved pre-repair counter. Right / green: repaired counter. Both panes intentionally have the same detection and tracking boxes. The yellow counting line is identical. The zoom-stress clip deliberately retains the frozen, incompatible boundary to expose calibration failure.</p><div class="cards">'''+''.join(cards)+'''</div><details><summary>Separation, camera framing and limitations</summary><p>All 23 earlier recordings were used only for development/regression. Three previously unused front-view takes were reserved. Code, detector/profile and source hashes were frozen before their activity labels were opened; reference source frames were reviewed before detector outcomes. No weights were trained, no score rule was changed, and no parameter search was performed for the repair.</p><p>WalkByShop1front is a tighter zoom despite retaining the same 384×288 resolution. Its physical doorway floor is near y=190 rather than the frozen y=145. Source review registered this condition before test outcomes; the source and all its reference events remain in the aggregate. Deployments need calibration appropriate to each fixed camera view.</p><p>All footage comes from the same legacy indoor camera/day and some actors; the results do not validate new properties, modern outdoor malls, darkness, weather, cars or cross-camera unique visitors. Public pretrained-model exposure to this benchmark is unknown. References use hand-labeled human trajectories/activities with visual review, not a second independent human audit.</p></details><p>Source: <a href="https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/">EC Funded CAVIAR project / IST 2001 37540, University of Edinburgh</a>, publicly released CC BY-SA footage and labels.</p><p><a href="report.json">One shared report: old counter source, settings, references, scores and audit</a> · <a href="../../data/counter-confirmation/manifest.json">Sources and references</a> · <a href="../entrance-improvement/review.html">Doorway and duplicate-suppression experiment</a> · <a href="../retail-entrance/review.html">Initial training trial</a></p></main></html>'''
    (OUTPUT/'review.html').write_text(doc)
    print(OUTPUT/'review.html')


if __name__=='__main__':main()
