"""Summarize saved crossing counts and score saved tracks against CAVIAR labels."""
import hashlib
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import tracking_metrics

OUTPUT = ROOT/'outputs/restored-baseline'
ANNOTATIONS = ROOT/'data/counter-confirmation/annotations.zip'


def count_summary(report, sequences):
    reference, predicted = Counter(), Counter()
    rows = []
    duration = 0
    for seq in sequences:
        rec = report['runs']['restored'][seq]
        gt = Counter(e['direction'] for e in report['references'][seq]['events'])
        pr = Counter(e['direction'] for e in rec['events'])
        reference.update(gt); predicted.update(pr)
        duration += rec['frames']/25
        rows.append({'sequence': seq, 'seconds':rec['frames']/25,
                     'reference':dict(gt),'reported':dict(pr),
                     'signed_total_error':sum(pr.values())-sum(gt.values()),
                     'absolute_total_error':abs(sum(pr.values())-sum(gt.values())),
                     'directional_absolute_error':sum(abs(pr[d]-gt[d]) for d in ['entry','exit'])})
    actual, reported = sum(reference.values()), sum(predicted.values())
    return {'seconds':duration, 'reference':dict(reference),'reported':dict(predicted),
            'reference_total':actual,'reported_total':reported,
            'signed_count_error':reported-actual,
            'aggregate_absolute_percentage_error':abs(reported-actual)/actual if actual else None,
            'mean_absolute_count_error_per_clip':sum(r['absolute_total_error'] for r in rows)/len(rows),
            'directional_absolute_error':sum(r['directional_absolute_error'] for r in rows),
            'hourly_count_error':None,
            'hourly_count_error_reason':'No complete, continuously recorded and manually verified hour. Short separate takes cannot measure hourly count error.',
            'per_clip':rows}


def track_sequence(archive, sequence, record):
    frames = ET.fromstring(archive.read(sequence+'.xml')).findall('frame')
    assert len(frames) == len(record['observations']) == record['frames']
    identities = sorted({row[0] for rows in record['observations'] for row in rows})
    mapping = {identity:n for n,identity in enumerate(identities)}
    ground, predictions = [], []
    for frame, rows in zip(frames,record['observations']):
        truth = []
        for obj in frame.findall('objectlist/object'):
            box = obj.find('box').attrib
            x,y,w,h = [float(box[k]) for k in ['xc','yc','w','h']]
            truth.append([int(obj.attrib['id']),x-w/2,y-h/2,x+w/2,y+h/2,1,1,1])
        ground.append(truth)
        predictions.append([[mapping[row[0]],*row[1:5],row[5],0] for row in rows])
    return tracking_metrics(ground,predictions,'person')


def combine(records):
    import numpy as np
    import trackeval
    combined = {}
    for name, cls in [('HOTA',trackeval.metrics.HOTA),('CLEAR',trackeval.metrics.CLEAR),('Identity',trackeval.metrics.Identity)]:
        inputs = {seq:{key:np.asarray(value) if isinstance(value,list) else value
                       for key,value in record['raw'][name].items()} for seq,record in records.items()}
        combined[name] = cls({'THRESHOLD':.5,'PRINT_CONFIG':False}).combine_sequences(inputs)
    h,c,i = [combined[name] for name in ['HOTA','CLEAR','Identity']]
    return {'HOTA':float(np.mean(h['HOTA'])*100),'IDF1':float(i['IDF1']*100),
            'association_accuracy':float(np.mean(h['AssA'])*100),
            'ID_switches':int(c['IDSW']),'fragmentations':int(c['Frag']),
            'matched_labeled_boxes':int(c['CLR_TP']),'unmatched_predictions':int(c['CLR_FP']),
            'missed_labeled_boxes':int(c['CLR_FN']),
            'box_precision':float(c['CLR_Pr']*100),'box_recall':float(c['CLR_Re']*100)}


def main():
    path = OUTPUT/'report.json'
    r = json.loads(path.read_text())
    m = {'computed_at_utc':datetime.now(timezone.utc).isoformat(),
         'source_observations':'Saved restored-profile tracks; no inference or tuning performed.',
         'crossings':r['totals']['restored'],'count_error':count_summary(r,r['sequences']),
         'zoom_crossings':r['zoom_totals'],'zoom_count_error':count_summary(r,[r['zoom_sequence']]),
         'tracking_protocol':{
             'implementation':'Official TrackEval pinned commit 12c8791b303e0a0b50f753af204249e622d0281a, existing local CAVIAR-to-MOT adapter; NumPy alias compatibility only.',
             'scope':'All published individual-person boxes and all saved person predictions, every frame. No post-hoc filtering or ROI exclusions.',
             'aggregation':'Official combine_sequences, not arithmetic mean of sequence percentages. IDs remain local to each clip.',
             'thresholds':'HOTA averaged over official IoU thresholds 0.05..0.95; IDF1/CLEAR use IoU 0.5.',
             'annotations_sha256':hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest(),
             'limitation':'Publisher labels are not an independently complete annotation of every background person/display. Unmatched predictions can include unlabeled real people and false display boxes. These full-scene scores are diagnostics, not doorway crossing scores.',
             'class':'Person only; vehicle continuity is not measured by these clips.'},
         'tracking_per_sequence':{}}
    with zipfile.ZipFile(ANNOTATIONS) as archive:
        for seq in r['sequences']+[r['zoom_sequence']]:
            summary,raw = track_sequence(archive,seq,r['runs']['restored'][seq])
            m['tracking_per_sequence'][seq] = {'summary':summary,'raw':raw}
            print(seq,summary,flush=True)
    m['tracking'] = combine({s:m['tracking_per_sequence'][s] for s in r['sequences']})
    m['zoom_tracking'] = combine({r['zoom_sequence']:m['tracking_per_sequence'][r['zoom_sequence']]})
    r['business_metrics'] = m
    path.write_text(json.dumps(r,indent=2)+'\n')
    a=m['crossings'];t=m['tracking'];b=m['count_error']
    rows = [('Crossing precision',f'{a["precision"]:.1%}'),('Crossing recall',f'{a["recall"]:.1%}'),
            ('Crossing F1',f'{a["f1"]:.1%}'),('Hourly count error','Not measured: no complete verified hour'),
            ('Observed total count error',f'{b["signed_count_error"]:+d} crossings ({b["aggregate_absolute_percentage_error"]:.1%} undercount)'),
            ('Full-scene person HOTA',f'{t["HOTA"]:.1f} / 100'),('Full-scene person IDF1',f'{t["IDF1"]:.1f} / 100'),
            ('ID switches / fragmentations',f'{t["ID_switches"]} / {t["fragmentations"]}')]
    section='<section id="business-metrics"><h2>Requested metrics — restored profile</h2><table><tr><th>Metric</th><th>Six original clips</th></tr>'+''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k,v in rows)+'</table>'
    section+=f'<p>Counted {b["reported_total"]} of {b["reference_total"]} crossings across {b["seconds"]:.1f} seconds of separate clips. Entries: {b["reported"]["entry"]}/{b["reference"]["entry"]}; exits: {b["reported"]["exit"]}/{b["reference"]["exit"]}. Mean absolute error {b["mean_absolute_count_error_per_clip"]:.2f} crossings per clip. No hourly extrapolation is presented.</p>'
    section+='<p>HOTA and IDF1 score all saved person boxes against published individual-person labels, including areas outside the doorway. Unlabeled background people and false display boxes affect these diagnostics. They measure a different task from crossing precision/recall. No vehicle continuity was scored. The zoomed recording remains separate; its results and per-sequence tracking details are in the shared report.</p></section>'
    page = OUTPUT/'review.html';doc=page.read_text()
    if '<section id="business-metrics">' in doc:
        import re
        doc=re.sub(r'<section id="business-metrics">.*?</section>',section,doc,flags=re.S)
    else:doc=doc.replace('<h2>Active settings</h2>',section+'<h2>Active settings</h2>')
    page.write_text(doc)
    print('METRIC_SUMMARY',json.dumps({'crossings':a,'count_error':b,'tracking':t,'zoom_tracking':m['zoom_tracking']},indent=2),flush=True)


if __name__=='__main__':main()
