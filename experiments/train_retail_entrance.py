"""Small camera-specific detector trial; temporary frame/label files stay outside outputs."""
import os
os.environ.update(YOLO_CONFIG_DIR='/private/tmp/propsight-yolo-config',MPLCONFIGDIR='/private/tmp/propsight-mpl',XDG_CACHE_HOME='/private/tmp/propsight-cache')
import sys,cv2,json,zipfile,xml.etree.ElementTree as ET,numpy as np,yaml,time
from pathlib import Path
from ultralytics import YOLO
root=Path(__file__).resolve().parents[1];temp=Path('/private/tmp/propsight-entrance');data=root/'data/retail-entrance';plan=json.loads((root/'configs/retail-entrance-evaluation.json').read_text());m=json.loads((data/'manifest.json').read_text());z=zipfile.ZipFile(data/'annotations.zip');work=temp/'training';crop=plan['camera']['detection_crop'];x0,y0,x3,y3=crop;W,H=x3-x0,y3-y0
selection={}
for split,nper in [('development',60),('validation',20)]:
 folder='train' if split=='development' else 'val'; (work/'images'/folder).mkdir(parents=True,exist_ok=True);(work/'labels'/folder).mkdir(parents=True,exist_ok=True)
 for seq in plan[split]:
  frames=ET.fromstring(z.read(seq+'.xml')).findall('frame');indices=set(np.linspace(0,len(frames)-1,nper,dtype=int).tolist());selection[seq]=sorted(indices);cap=cv2.VideoCapture(str(data/m['sequences'][seq]['video']));n=0
  while True:
   ok,im=cap.read()
   if not ok:break
   if n in indices:
    name=f'{seq}-{n:05d}';cv2.imwrite(str(work/'images'/folder/(name+'.jpg')),im[y0:y3,x0:x3]);rows=[]
    for o in frames[n].findall('objectlist/object'):
     b=o.find('box').attrib;x,y,w,h=[float(b[k]) for k in ['xc','yc','w','h']];left,top,right,bottom=max(x0,x-w/2),max(y0,y-h/2),min(x3,x+w/2),min(y3,y+h/2)
     if right-left<3 or bottom-top<6:continue
     rows.append(f'0 {((left+right)/2-x0)/W} {((top+bottom)/2-y0)/H} {(right-left)/W} {(bottom-top)/H}')
    (work/'labels'/folder/(name+'.txt')).write_text('\n'.join(rows)+'\n')
   n+=1
  cap.release()
(work/'dataset.yaml').write_text(yaml.safe_dump({'path':str(work),'train':'images/train','val':'images/val','names':YOLO(str(root/'yolo11s.pt')).names}));(temp/'training-selection.json').write_text(json.dumps(selection,indent=2))
model=YOLO(str(root/'yolo11s.pt'));model.train(data=str(work/'dataset.yaml'),epochs=8,imgsz=416,batch=8,freeze=10,device='cpu',workers=0,seed=42,deterministic=True,project=str(temp),name='training-run',exist_ok=True,plots=False,cache=True,amp=False,optimizer='AdamW',lr0=.001,lrf=.1,patience=3,mosaic=.2,mixup=0,fliplr=.5,verbose=False,save=True)
out=root/'models';out.mkdir(exist_ok=True);best=temp/'training-run/weights/best.pt';(out/'retail-entrance-person.pt').write_bytes(best.read_bytes());print('TRAINING_COMPLETE',out/'retail-entrance-person.pt',flush=True)
