"""Audit saved benchmark results and show frame-rate effects on identical frames."""
import json,hashlib,sys,platform
from pathlib import Path
from datetime import datetime,timezone
import cv2,numpy as np
import benchmark as b
from evaluation import tracking_metrics,box_iou
from scipy.optimize import linear_sum_assignment


def overlay(im,truth,pred):
 truth=[g for g in truth if g[5] and g[6]==1];ious=box_iou([g[1:5] for g in truth],[p[1:5] for p in pred]);matched_gt=set();matched_pred=set()
 if truth and pred:
  scores=ious.copy();scores[scores<.5]=0;x,y=linear_sum_assignment(-scores);pairs=[(int(i),int(j)) for i,j in zip(x,y) if scores[i,j]>0];matched_gt={i for i,j in pairs};matched_pred={j for i,j in pairs}
 for i,g in enumerate(truth):
  if i not in matched_gt:
   x1,y1,x2,y2=map(int,g[1:5]);cv2.rectangle(im,(x1,y1),(x2,y2),(0,185,255),2)
 for j,p in enumerate(pred):
  x1,y1,x2,y2=map(int,p[1:5]);color=(40,220,40) if j in matched_pred else (40,40,240);cv2.rectangle(im,(x1,y1),(x2,y2),color,2);cv2.putText(im,f'#{int(p[0])}',(x1,max(20,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
 return im


def main():
 report=b.read_report(b.OUTPUT)
 if not report.get('finished_at_utc'):raise RuntimeError('Complete the frozen benchmark before this diagnostic')
 selected=report['rounds']['3']['candidates']['frozen-final']['results'];report['native_on_common_frames']={}
 path=b.OUTPUT/'frame-rate-comparison.mp4';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),5,(1920,1080));frames_written=0
 if not writer.isOpened():raise RuntimeError('Comparison encoder unavailable')
 for seq,native in report['native_fps_check'].items():
  five=selected[seq];indices=five['source_frame_indices'];gt,regions=b.labels(seq);common=[native['observations'][index] for index in indices]
  metrics,_=tracking_metrics([gt[index] for index in indices],common,five['kind'],regions)
  rawfive,_=b.raw(seq,five['profile'],5);rawn,_=b.raw(seq,native['profile'],native['fps']);agree=True;maximum=0
  for i,index in enumerate(indices):
   x=np.asarray(rawfive[i],float).reshape(-1,6);y=np.asarray(rawn[index],float).reshape(-1,6)
   if x.shape!=y.shape:agree=False;continue
   if len(x):maximum=max(maximum,float(np.max(np.abs(x-y))))
   if not np.allclose(x,y,atol=1e-4,rtol=1e-5):agree=False
  report['native_on_common_frames'][seq]={'metrics':metrics,'frames':len(indices),'raw_detections_match_on_shared_source_frames':agree,'raw_max_absolute_difference':maximum,'comparison':'Native-rate tracker outputs scored at exactly the same source frames used in the 5-FPS run; only access to intermediate frames differs. No holdout tuning. Three-frame crossing confirmation has different durations at different FPS, so crossing denominators are not directly comparable.'}
  for i,index in enumerate(indices):
   original=b.image(seq,index);canvas=np.full((1080,1920,3),(24,35,43),np.uint8)
   for col,obs in enumerate([five['observations'][i],common[i]]):
    eligible=b.eligible_predictions(gt[index],obs,five['kind'],regions);picture=overlay(original.copy(),gt[index],eligible);picture=cv2.resize(picture,(960,540));canvas[170:710,col*960:(col+1)*960]=picture
   title=b.PLAN['sequences'][seq]['title'];cv2.putText(canvas,f'{seq}: {title}',(24,45),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
   cv2.putText(canvas,'Same source frames and identical YOLO detections; the right tracker receives intermediate frames.',(24,88),cv2.FONT_HERSHEY_SIMPLEX,.65,(215,225,233),1)
   for col,(label,m) in enumerate([('Tracker receives 5 FPS',five['metrics']),(f'Tracker receives original {native["fps"]} FPS',metrics)]):
    x=24+col*960;cv2.putText(canvas,label,(x,143),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,235,235),2)
    cv2.putText(canvas,f'Matched labeled boxes: {m["recall"]:.1f}%',(x,770),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    cv2.putText(canvas,f'Precision: {m["precision"]:.1f}% | ID continuity: {m["IDF1"]:.1f}',(x,820),cv2.FONT_HERSHEY_SIMPLEX,.65,(215,225,233),1)
    cv2.putText(canvas,f'ID switches on these frames: {m["ID_switches"]}',(x,866),cv2.FONT_HERSHEY_SIMPLEX,.65,(215,225,233),1)
   cv2.putText(canvas,'GREEN: matched prediction   RED: unmatched prediction   AMBER: missed human label',(24,950),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,215,255),1)
   cv2.putText(canvas,f'Source time {index/native["fps"]:.1f}s | Offline processing; comparison video renders at 5 FPS.',(24,1010),cv2.FONT_HERSHEY_SIMPLEX,.65,(215,225,233),1)
   writer.write(canvas);frames_written+=1
 writer.release();cap=cv2.VideoCapture(str(path));decoded=0
 while cap.read()[0]:decoded+=1
 cap.release()
 if decoded!=frames_written:raise RuntimeError('Incomplete frame-rate comparison video')
 report['frame_rate_comparison_video']={'file':path.name,'frames':decoded,'fps':5,'description':'Two reserved scenes in one side-by-side video, with identical source frames and the same frozen checkpoint. This is temporal-input analysis, not new model tuning or a real-time speed claim.'}
 manifest=report['protocol']['source_manifest'];expected={'mot17-selected.zip':manifest['archive_sha256'],'detrac-selected.zip':manifest['vehicle_archive_sha256']}
 report['final_audit']={'at_utc':datetime.now(timezone.utc).isoformat(),'source_archive_sha256':{},'all_frozen_file_hashes_match':True,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__},'review_script_sha256':b.sha(Path(__file__))}
 for name,digest in expected.items():
  actual=b.sha(b.DATA/name);assert actual==digest,name;report['final_audit']['source_archive_sha256'][name]=actual
 for name,digest in report['protocol']['frozen_sha256'].items():assert b.sha(b.ROOT/name)==digest,name
 report['status']='Three source-disjoint rounds completed. The selected checkpoint gives modest holdout gains, while serious errors remain. On identical held-out traffic frames, native-rate tracking raises recall from 42.3% to 82.6%; the shared-frame YOLO detections are unchanged. This is offline processing, not a demonstrated live frame rate.'
 report['limitations_after_review']=['The three-frame confirmation rule is 0.6s at 5 FPS, 0.1s at 30 FPS and 0.12s at 25 FPS. Native and subsampled crossing references therefore differ; compare their own matched/false/missed events rather than treating denominators as identical.','Reference event matching uses confirmed-event timestamps and positions, within 1s and 10% image width. It is a declared spatiotemporal check of label-derived events, not manually verified doorway accuracy.','Counts and IDs are per-camera/per-recording, not unique visitors or occupancy.','Published MOT distractors and UA-DETRAC ignored areas are excluded. Scores do not validate every background object or standing-person use case; older regression checks cover selected examples only.']
 b.save_report(b.OUTPUT,report);b.gallery(report)
 html=(b.OUTPUT/'review.html').read_text();rows=[]
 for seq,a in report['native_on_common_frames'].items():
  five=selected[seq]['metrics'];m=a['metrics'];rows.append(f'<tr><td>{seq}</td><td>{five["recall"]:.1f}% → {m["recall"]:.1f}%</td><td>{five["IDF1"]:.1f} → {m["IDF1"]:.1f}</td><td>{five["ID_switches"]} → {m["ID_switches"]}</td></tr>')
 section='<h2>Frame-rate effect on identical scored frames</h2><p>Both versions use the same frozen model and exactly the same YOLO detections on shared frames. The native-rate tracker also sees intervening frames. These are recorded-clip results, not measured live throughput.</p><table><tr><th>Scene</th><th>Labeled boxes found: 5 FPS → native</th><th>ID continuity</th><th>ID switches</th></tr>'+''.join(rows)+'</table><p><a href="frame-rate-comparison.mp4">Watch both frame-rate comparisons in one video</a></p><p>Precision measures the fraction of tracked boxes matching labels; recall measures labeled boxes found. HOTA and IDF1 summarize tracking and ID continuity (higher is better). False/missed boxes are per-frame observations, not unique people or vehicles.</p>'
 html=html.replace('<h2>Native frame-rate check',section+'<h2>Native frame-rate check',1)
 footer='<p>Sources: <a href="https://motchallenge.net/data/MOT17/">MOT17 archive</a> · <a href="https://sites.google.com/view/daweidu/projects/ua-detrac">UA-DETRAC author downloads</a> · <a href="https://faculty.ucmerced.edu/mhyang/papers/cviu2020_detrac.pdf">UA-DETRAC paper</a> · <a href="https://github.com/JonathonLuiten/TrackEval">TrackEval</a>.</p><p>Crossing references are derived from human tracks. Low-visibility/overlap references are flagged as ambiguous, with matching predictions excluded from event scoring. Raw model totals can include these unscored events. The fixed three-frame rule has different durations at different frame rates; crossing denominators differ.</p>'
 (b.OUTPUT/'review.html').write_text(html.replace('</main>',footer+'</main>'))
 print(json.dumps(report['native_on_common_frames'],indent=2));print('Comparison video, provenance audit and review page complete.',flush=True)

if __name__=='__main__':main()
