"""Plot confidence sweeps of frozen tracked outputs; never rerun or retune trackers."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/demand-pr-matplotlib')
import sys,json,hashlib,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import benchmark as source
from evaluation import tracking_metrics


def main():
    path=ROOT/'docs/metrics/confidence-study/larger-confidence-recovery/report.json'
    report=json.loads(path.read_text());curves={};labels={};inverse={'person':0,'bicycle':1,'car':2,'motorcycle':3,'bus':5,'truck':7}
    for seq,record in report['runs']['initial'].items():
        gt,regions=source.labels(seq);first=record['source_start_frame'];labels[seq]=(gt[first:first+record['frames']],regions)
    for mode in ['initial','selected']:
        points=[]
        for threshold in np.round(np.arange(.1,1.0,.05),2):
            tp=fp=fn=0
            for seq,record in report['runs'][mode].items():
                pred=[[] for _ in range(record['frames'])]
                for o in record['observations']:
                    if o['confidence']>=threshold:
                        pred[o['frame']].append([int(o['track_id'].split('-')[-1]),*[o[k] for k in ['x1','y1','x2','y2']],o['confidence'],inverse[o['class']]])
                gt,regions=labels[seq];m,_=tracking_metrics(gt,pred,record['kind'],regions)
                tp+=m['matched_labeled_boxes'];fp+=m['false_detections'];fn+=m['missed_labeled_boxes']
            points.append({'threshold':float(threshold),'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn)})
        expected=report['later_totals'][mode]
        assert all(points[0][k]==expected[k] for k in ['tp','fp','fn'])
        curves[mode]=points
    out=ROOT/'docs/metrics/precision-recall';out.mkdir(exist_ok=True)
    evidence={'source_report':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'440 frames, 4179 publisher-labeled appearances; MOT17-09 and UA-DETRAC MVI_63561 later windows, source seconds 8-16. Known historical cameras, not independent validation.','method':'Filter saved tracked boxes by output confidence >= threshold, then rescore with existing pinned TrackEval and published ignore rules. Track states and initiation/class thresholds remain frozen. This is not a detector AP curve, tracker rerun or threshold recommendation. No AP/AUC computed.','curves':curves}
    (out/'points.json').write_text(json.dumps(evidence,indent=2)+'\n')
    with (out/'points.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['profile','threshold','tp','fp','fn','precision','recall'],lineterminator='\n');w.writeheader()
        for mode,points in curves.items():
            for p in points:w.writerow({'profile':mode,**p})
    fig,ax=plt.subplots(figsize=(9,6.2));colors={'initial':'#136f63','selected':'#bd5a28'}
    for mode,name in [('initial','Retained profile'),('selected','Rejected adjustment')]:
        pts=[p for p in curves[mode] if p['precision'] is not None]
        ax.plot([p['recall'] for p in pts],[p['precision'] for p in pts],'-o',ms=4,lw=2,color=colors[mode],label=name)
        p=pts[0];ax.scatter([p['recall']],[p['precision']],s=115,color=colors[mode],edgecolor='white',zorder=4)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Recall',ylabel='Precision',title='Precision–recall: saved tracked outputs')
    ax.grid(alpha=.2);ax.legend(loc='lower left');ax.set_aspect('equal',adjustable='box')
    fig.text(.5,.045,'MOT17-09 + UA-DETRAC MVI_63561 · 440 frames · 4,179 labeled appearances',ha='center',fontsize=10)
    fig.text(.5,.015,'Output-confidence sweep 0.10–0.95; trackers frozen. Large dots: threshold 0.10.',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.075,1,1));fig.savefig(out/'curve.png',dpi=180);fig.savefig(out/'curve.svg');plt.close(fig)
    svg=out/'curve.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print('Saved PR figure, raw points, and source provenance. Baseline counts reproduce published report.')

if __name__=='__main__':main()
