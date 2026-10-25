"""Human-label evaluation helpers. Reference crossings do not call LineCounter."""
from collections import defaultdict
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(a, b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    intersection=np.maximum(hi-lo,0).prod(2)
    union=np.maximum(a[:,2:]-a[:,:2],0).prod(1)[:,None]+np.maximum(b[:,2:]-b[:,:2],0).prod(1)[None,:]-intersection
    return intersection/np.maximum(union,1e-12)


def ignore_fraction(box, regions):
    """Exact area of the union of clipped ignore rectangles / prediction area."""
    x1,y1,x2,y2=box;rects=[]
    for a,b,c,d in regions:
        r=(max(x1,a),max(y1,b),min(x2,c),min(y2,d))
        if r[2]>r[0] and r[3]>r[1]:rects.append(r)
    xs=sorted({x for r in rects for x in (r[0],r[2])});area=0
    for left,right in zip(xs,xs[1:]):
        intervals=sorted((r[1],r[3]) for r in rects if r[0]<right and r[2]>left)
        total=0;end=-math.inf
        for low,high in intervals:
            total+=max(0,high-max(low,end));end=max(end,high)
        area+=(right-left)*total
    return area/max(1e-12,(x2-x1)*(y2-y1))


def tracking_metrics(gt_frames,pred_frames,kind,ignore_regions=None):
    # The pinned upstream evaluator still uses NumPy's removed scalar aliases.
    # Restore the names in this process; metric formulas remain upstream code.
    if 'int' not in np.__dict__:np.int=np.int64
    if 'float' not in np.__dict__:np.float=float
    import trackeval
    n=len(gt_frames);assert len(pred_frames)==n
    raw={k:[] for k in ['gt_ids','gt_classes','gt_dets','gt_extras','tracker_ids','tracker_classes','tracker_dets','tracker_confidences','similarity_scores']}
    raw.update(num_timesteps=n,seq='local_sequence')
    ignored_predictions=0
    for truth,pred in zip(gt_frames,pred_frames):
        truth=np.asarray(truth,float).reshape(-1,8) # id,x1,y1,x2,y2,valid,class,visibility
        pred=np.asarray(pred,float).reshape(-1,7) # id,x1,y1,x2,y2,confidence,coco_class
        ious=box_iou(truth[:,1:5],pred[:,1:5])
        if kind=='vehicle' and len(pred):
            matched=set()
            if len(truth):
                scores=ious.copy();scores[scores<.5]=0;x,y=linear_sum_assignment(-scores)
                matched={int(j) for i,j in zip(x,y) if scores[i,j]>0}
            keep=[j for j in range(len(pred)) if j in matched or ignore_fraction(pred[j,1:5],ignore_regions or [])<.5]
            ignored_predictions+=len(pred)-len(keep);pred=pred[keep];ious=ious[:,keep]
        gt_boxes=truth[:,1:5].copy();gt_boxes[:,2:]-=gt_boxes[:,:2]
        boxes=pred[:,1:5].copy();boxes[:,2:]-=boxes[:,:2]
        raw['gt_ids'].append(truth[:,0].astype(int));raw['gt_classes'].append(truth[:,6].astype(int))
        raw['gt_dets'].append(gt_boxes);raw['gt_extras'].append({'zero_marked':truth[:,5].astype(int)})
        raw['tracker_ids'].append(pred[:,0].astype(int));raw['tracker_classes'].append(np.ones(len(pred),int))
        raw['tracker_dets'].append(boxes);raw['tracker_confidences'].append(pred[:,5]);raw['similarity_scores'].append(ious)
    # Use MOTChallenge's official distractor/zero-marked preprocessing. Vehicle
    # targets have already been grouped into class 1 with explicit ignore areas.
    dataset=trackeval.datasets.MotChallenge2DBox.__new__(trackeval.datasets.MotChallenge2DBox)
    dataset.benchmark='MOT17';dataset.do_preproc=True
    dataset.class_name_to_class_id={'pedestrian':1,'person_on_vehicle':2,'static_person':7,'distractor':8,'reflection':12}
    dataset.valid_class_numbers=list(range(1,14))
    data=dataset.get_preprocessed_seq_data(raw,'pedestrian')
    config={'THRESHOLD':.5,'PRINT_CONFIG':False}
    hota=trackeval.metrics.HOTA(config).eval_sequence(data)
    clear=trackeval.metrics.CLEAR(config).eval_sequence(data)
    identity=trackeval.metrics.Identity(config).eval_sequence(data)
    summary={'precision':float(clear['CLR_Pr']*100),'recall':float(clear['CLR_Re']*100),
      'HOTA':float(np.mean(hota['HOTA'])*100),'IDF1':float(identity['IDF1']*100),
      'MOTA':float(clear['MOTA']*100),'false_detections':int(clear['CLR_FP']),
      'missed_labeled_boxes':int(clear['CLR_FN']),'matched_labeled_boxes':int(clear['CLR_TP']),
      'ID_switches':int(clear['IDSW']),'fragmentations':int(clear['Frag']),
      'labeled_boxes':int(data['num_gt_dets']),'predicted_boxes_after_ignores':int(data['num_tracker_dets']),
      'ignored_vehicle_predictions':ignored_predictions}
    def convert(value):
        if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        return value
    return summary,convert({'HOTA':hota,'CLEAR':clear,'Identity':identity})


def reference_crossings(gt_frames,line,fps,kind,source_indices,width,height):
    """Independent trajectory-side runs, finite intersection and quality flags."""
    a,b=np.asarray(line,float);v=b-a;length=float(np.linalg.norm(v));states={};events=[]
    def cross(x,y):return x[0]*y[1]-x[1]*y[0]
    deadband=3 if kind=='vehicle' else 8;initial=1 if kind=='vehicle' else 3;confirm=3;gap=max(1,round(fps*.5))
    for frame,rows in enumerate(gt_frames):
        for row in rows:
            identity,x1,y1,x2,y2,valid,cls,visibility=row
            if not valid or cls!=1:continue
            identity=int(identity);point=np.array([(x1+x2)/2,y2]);distance=cross(v,point-a)/length
            s=states.get(identity)
            if s is None or frame-s['last']>gap:
                s={'origin':None,'sign':None,'run':[],'last':frame};states[identity]=s
            s['last']=frame
            if abs(distance)<=deadband:s['sign']=None;s['run']=[];continue
            sign=1 if distance>0 else -1
            reliable=visibility>=.5 and 0<=point[0]<width and 0<=point[1]<height
            if s['sign']!=sign:s['sign']=sign;s['run']=[]
            s['run'].append((point,reliable));s['run']=s['run'][-confirm:]
            needed=initial if s['origin'] is None else confirm
            if len(s['run'])<needed:continue
            if s['origin'] is None or s['origin']==sign:
                s['origin']=sign;s['point']=point;s['quality']=all(q for _,q in s['run']);continue
            previous=s['point'];denom=float(cross(v,point-previous));crosses=False
            if abs(denom)>1e-9:
                t=float(cross(v,a-previous)/denom);hit=previous+t*(point-previous);u=float(np.dot(hit-a,v)/np.dot(v,v));crosses=0<=t<=1 and 0<=u<=1
            if crosses:
                events.append({'frame':frame,'source_frame':int(source_indices[frame])+1,'video_seconds':frame/fps,
                  'track_id':identity,'direction':'entry' if sign>0 else 'exit','point':point.tolist(),
                  'reliable':bool(s['quality'] and all(q for _,q in s['run']))})
            s['origin']=sign;s['point']=point;s['quality']=all(q for _,q in s['run'])
    return events


def crossing_metrics(reference,predicted,fps,width):
    """Match same-direction events within one second and 10% image width."""
    references=[r for r in reference if r['reliable']];ambiguous=[r for r in reference if not r['reliable']]
    def assign(ref,preds):
        if not ref or not preds:return [],set()
        costs=np.full((len(ref),len(preds)),1e6)
        for i,r in enumerate(ref):
            for j,p in enumerate(preds):
                dt=abs(r['frame']-p['frame'])/fps;distance=np.linalg.norm(np.asarray(r['point'])-p['point'])
                if r['direction']==p['direction'] and dt<=1 and distance<=width*.1:costs[i,j]=dt+distance/max(1,width*.1)
        x,y=linear_sum_assignment(costs);pairs=[(int(i),int(j)) for i,j in zip(x,y) if costs[i,j]<1e6]
        return pairs,{j for _,j in pairs}
    pairs,matched=assign(references,predicted)
    remaining=[p for j,p in enumerate(predicted) if j not in matched];ambiguous_pairs,ignored=assign(ambiguous,remaining)
    eligible_predictions=len(predicted)-len(ignored);tp=len(pairs);fp=eligible_predictions-tp;fn=len(references)-tp
    def counts(rows):return {d:sum(x['direction']==d for x in rows) for d in ['entry','exit']}
    return {'reference_confirmed':counts(references),'reference_ambiguous':counts(ambiguous),'model_reported':counts(predicted),
      'matched_events':tp,'false_events':fp,'missed_events':fn,'predictions_matching_ambiguous_reference':len(ignored),
      'event_precision':100*tp/eligible_predictions if eligible_predictions else None,'event_recall':100*tp/len(references) if references else None,
      'reference_events':reference,'model_events':predicted,
      'definition':'Reference crossings independently derived from human-labeled trajectories; visibility/overlap reliability >=0.5, finite line, current deadband and confirmation. Ambiguous references and matching predictions are excluded. Same-direction spatiotemporal matching within 1s / 10% image width; not manually labeled property doorway events.'}
