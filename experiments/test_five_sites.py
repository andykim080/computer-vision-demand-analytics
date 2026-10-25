"""Evaluate only the new five-location test manifest, using frozen settings."""
import argparse
from argparse import Namespace
import base64
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tracker import run,read_report,save_report


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gallery(output, report, settings, dataset):
    import cv2
    cards=[]
    for site,spec in settings['sites'].items():
        for number in range(1,6):
            key=f'{site}/{number:02d}'
            if key not in report['runs']:continue
            record=report['runs'][key];capture=record['source_context'];pictures=[]
            preview_frame=record['validation'].get('preview_frame',45)
            for label,path in [('Source',dataset/capture['file']),('Tracker',output/record['video'])]:
                cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,preview_frame);ok,im=cap.read();cap.release()
                if not ok:raise RuntimeError(f'Missing gallery frame: {path}')
                im=cv2.resize(im,(480,round(im.shape[0]*480/im.shape[1])))
                ok,jpeg=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,78]);assert ok
                uri='data:image/jpeg;base64,'+base64.b64encode(jpeg).decode()
                pictures.append(f'<figure><figcaption>{label} · {preview_frame/5:.1f} seconds</figcaption><img src="{uri}" alt="{html.escape(spec["title"])} {label}"></figure>')
            counts=record['summary']['counts']
            count_text=' · '.join(f'{name}: {v["entry"]} in / {v["exit"]} out' for name,v in counts.items() if v['entry'] or v['exit']) or 'No model-reported crossings'
            source_path=Path('../../data/test-five-sites')/capture['file']
            utc=capture['captured_at_utc'].split('.')[0].replace('T',' ')+' UTC'
            review=html.escape(record['validation'].get('visual_review','Model outputs; visual review pending.'))
            cards.append(f'<article data-site="{site}"><h2>{html.escape(spec["title"])} · interval {number}</h2><p>Captured {utc} · offset {capture["observed_collection_offset_seconds"]:.0f}s from this location’s collection start</p><div class="pictures">{"".join(pictures)}</div><p class="counts">Final model counts · {html.escape(count_text)}</p><p>{review}</p><p><a href="{source_path.as_posix()}">Open source clip</a><a href="{record["video"]}">Open annotated clip</a></p></article>')
    options=''.join(f'<option value="{site}">{html.escape(spec["title"])}</option>' for site,spec in settings['sites'].items())
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Five-location tracker test</title><style>
body{margin:0;background:#f1f4f6;color:#14242f;font:16px system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:32px}h1{font-size:34px;margin-bottom:8px}header{margin-bottom:28px}.badge{color:#087b70;font-weight:700}select{padding:10px;font:inherit;border:1px solid #b9c8d0;border-radius:8px}article{background:white;border:1px solid #dce4e9;border-radius:14px;margin:20px 0;padding:22px}h2{font-size:21px;margin:0}p{line-height:1.5;color:#48606d}.pictures{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}figcaption{font-size:13px;margin:8px 0}img{width:100%;border-radius:8px}a{display:inline-block;color:#006c7c;font-weight:600;margin-right:24px}.counts{color:#14242f;font-weight:600}@media(max-width:650px){main{padding:15px}.pictures{display:block}figure{margin:12px 0}}</style><main><header><div class="badge">TEST DATA · FROZEN TRACKER SETTINGS</div><h1>Five locations, five intervals</h1><p>25 new 18-second clips. Earlier tuning footage is retained separately and excluded. Source and tracker views use the same frame. Counts below are model outputs; sampled visual checks do not establish a full accuracy score.</p><p>Capture times are recorded individually. Camera replacements and retries caused some intervals to exceed two minutes. These clips sample nighttime traffic, not different times of day.</p><select id="site"><option value="all">All locations</option>'''+options+'''</select><p><a href="report.json">Shared results and review notes</a></p></header>'''+''.join(cards)+'''</main><script>document.querySelector('#site').onchange=e=>{document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.site!==e.target.value)}</script></html>'''
    (output/'review.html').write_text(document)


def evaluate(dataset,output,wait_seconds):
    import cv2
    settings_path=ROOT/'configs/five-site-test.json';settings=json.loads(settings_path.read_text())
    detector=settings['detector'];frozen={str(path.relative_to(ROOT)):sha(path) for path in [settings_path,ROOT/'tracker.py',ROOT/'counting.py',ROOT/settings['tracker'],ROOT/detector['model']]}
    excluded=[]
    for path in sorted((ROOT/'data').glob('*.mp4'))+sorted((ROOT/'data/kathy-pub-five-tests').glob('*.mp4')):
        excluded.append({'file':str(path.relative_to(ROOT)),'sha256':sha(path)})
    old_hashes={x['sha256'] for x in excluded}
    report=read_report(output)
    report.setdefault('test_protocol',{
      'split':'test','started_at_utc':datetime.now(timezone.utc).isoformat(),
      'settings':settings,'frozen_file_sha256':frozen,'older_footage_excluded':excluded,
      'scope':'25 newly captured clips at five physical locations; no local tuning footage or outcomes used to change detector/tracker thresholds.',
      'limitations':['Not different times of day: current collection is nighttime at these locations.','Counting lines are illustrative image boundaries, not actual property entrances.','Temporary IDs restart per clip; counts are crossings, not unique visitors.','Visual spot checks are incomplete ground truth and do not establish precision, recall, MOTA or IDF1.','Pretrained model training sources are not audited; exclusion applies to local development footage.']})
    save_report(output,report);deadline=time.monotonic()+wait_seconds
    while len(report['runs'])<25:
        if time.monotonic()>deadline:raise TimeoutError(f'Only {len(report["runs"])}/25 test clips available before deadline')
        for relative,digest in frozen.items():
            if sha(ROOT/relative)!=digest:raise RuntimeError(f'Frozen test input changed: {relative}')
        manifest=json.loads((dataset/'manifest.json').read_text())
        if set(manifest['sites'])!=set(settings['sites']):raise ValueError('Dataset sites differ from frozen test plan')
        ready=sorted((c['interval'],site,c) for site,spec in manifest['sites'].items() for c in spec['clips'])
        progress=False
        for interval,site,capture in ready:
            key=f'{site}/{interval:02d}'
            if key in report['runs']:continue
            source=dataset/capture['file'];digest=sha(source)
            if digest!=capture['source_sha256'] or digest in old_hashes:raise ValueError('Source integrity or development/test exclusion failed')
            camera_config=dict(settings['profile'],camera_id=site,line=settings['sites'][site]['line'])
            site_output=output/site;started=time.monotonic()
            with tempfile.TemporaryDirectory(prefix='propsight-test-config-') as tmp:
                cfg=Path(tmp)/'camera.json';cfg.write_text(json.dumps(camera_config))
                name,record=run(Namespace(video=str(source),config=str(cfg),tracker=str(ROOT/settings['tracker']),model=str(ROOT/detector['model']),classes=detector['classes'],output=str(site_output),name=f'{interval:02d}',save_observations=True,preview=None,conf=detector['conf'],iou=detector['iou'],imgsz=detector['imgsz'],device=detector['device'],fps=None,max_frames=None))
            elapsed=time.monotonic()-started
            observation=site_output/record.pop('observations_file')
            with observation.open(newline='') as handle:
                reader=csv.reader(handle);columns=next(reader);rows=list(reader)
            cap=cv2.VideoCapture(str(site_output/record['video']));decoded=0
            while cap.read()[0]:decoded+=1
            cap.release()
            if decoded!=90 or record['summary']['frames_processed']!=90:raise RuntimeError(f'{key}: incomplete video')
            record['video']=f'{site}/{record["video"]}';record['source_context']=dict(capture,source_page=manifest['sites'][site]['source_page'])
            record['observations']={'columns':columns,'rows':rows}
            record['validation']={'complete_decode':True,'decoded_frames':decoded,'processing_seconds':round(elapsed,3),'source_hash_matches_manifest':True,'not_local_development_footage':True,'visual_review':'Pending source/output spot check; counts are unverified model outputs.'}
            report['runs'][key]=record;save_report(output,report)
            observation.unlink();(site_output/'report.json').unlink()
            print(f'Completed {len(report["runs"])}/25: {key}',flush=True);progress=True
        if not progress:time.sleep(2)
    report['capture_manifest']=json.loads((dataset/'manifest.json').read_text())
    report['test_protocol']['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    save_report(output,report);gallery(output,report,settings,dataset)
    print(f'Test set complete: {output.resolve()}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/test-five-sites')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/test-five-sites')
    parser.add_argument('--wait-seconds',type=float,default=1200)
    args=parser.parse_args();evaluate(args.input,args.output,args.wait_seconds)
