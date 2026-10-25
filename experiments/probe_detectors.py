"""Compare raw detector outputs before changing tracker association settings."""
import json
from collections import Counter
from pathlib import Path
import time
import cv2
from ultralytics import YOLO

root=Path(__file__).resolve().parents[1]
cap=cv2.VideoCapture(str(root/'data/demo-cars.mp4'))
frames=[]
for index in (0,40,80):
    cap.set(cv2.CAP_PROP_POS_FRAMES,index)
    ok,frame=cap.read()
    if ok: frames.append((index,frame))
cap.release()
out=root/'experiments/raw-detector-comparison.json'
records=[]
for model_name in ('yolo11n.pt','yolo11s.pt'):
    model=YOLO(str(root/model_name))
    for size,rotation in ((640,False),(1280,False),(640,True)):
        for index,frame in frames:
            if rotation: frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
            started=time.perf_counter()
            result=model.predict(frame,conf=.01,imgsz=size,device='cpu',verbose=False)[0]
            boxes=[{'class':model.names[int(b.cls.item())],'confidence':round(b.conf.item(),4),'xyxy':b.xyxy[0].tolist()} for b in result.boxes]
            counts=Counter(b['class'] for b in boxes if b['confidence']>=.25)
            record={'model':model_name,'imgsz':size,'rotated_90':rotation,'frame':index,'detections_at_025':dict(counts),'vehicle_candidates_at_001':sum(b['class'] in ('car','truck','bus') for b in boxes),'seconds':round(time.perf_counter()-started,3),'boxes':boxes}
            records.append(record)
            print({k:v for k,v in record.items() if k!='boxes'},flush=True)
# Keep the entire comparison in one existing report; do not generate preview JPGs.
out.write_text(json.dumps(records,indent=2)+'\n')
