#!/usr/bin/env python3
"""Reproduce the complete retail entrance Metal replay without replacing previous outputs.
Run with the project's .venv/bin/python. Every invocation creates a new result folder.
Examples:
  .venv/bin/python /absolute/path/run_metal_dataset.py --mode pilot
  .venv/bin/python /absolute/path/run_metal_dataset.py --mode full --devices mps
--fps is deliberately omitted: it changes timestamps, it does not subsample frames.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

DEFAULT_PROJECT = Path('/Users/andykim/project/Propsight 2')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def parse_record(folder, sequence, expected):
    import cv2
    report = json.loads((folder / 'report.json').read_text())
    if len(report['runs']) != 1:
        raise RuntimeError('Expected exactly one run in isolated output directory')
    rec = next(iter(report['runs'].values()))
    n = rec['summary']['frames_processed']
    if n != expected:
        raise RuntimeError(f'{sequence}: decoded {n}, annotation expects {expected}')
    cap = cv2.VideoCapture(str(folder / rec['video']))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    if decoded != expected:
        raise RuntimeError(f'{sequence}: rendered video has {decoded} frames, expected {expected}')
    observations = [[] for _ in range(n)]
    with (folder / rec['observations_file']).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['class'] == 'person':
                observations[int(row['frame'])].append([
                    row['track_id'], *[float(row[k]) for k in ['x1','y1','x2','y2','confidence']]])
    events = []
    # Tracker events omit points; recover the same-frame bottom-center from CSV.
    by_frame = {i: {r[0]: r for r in rows} for i, rows in enumerate(observations)}
    for event in rec['events']:
        if event['class'] != 'person':
            continue
        r = by_frame[event['frame']][event['track_id']]
        events.append({**event, 'point': [(r[1]+r[3])/2, r[4]]})
    return rec, {'frames': n, 'observations': observations, 'events': events}, decoded

def aggregate(records, module):
    return {
        'timely_crossings': module.totals(records),
        'identity_verified_crossings': module.count_audit_totals(records),
        'reference_counts': {d: sum(r['scores']['reference_counts'].get(d,0) for r in records) for d in ['entry','exit']},
        'predicted_counts': {d: sum(r['scores']['model_counts'].get(d,0) for r in records) for d in ['entry','exit']},
        'directional_absolute_error': sum(sum(abs(r['scores']['reference_counts'].get(d,0)-r['scores']['model_counts'].get(d,0)) for d in ['entry','exit']) for r in records)
    }

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--output-parent', type=Path, default=DEFAULT_PROJECT / 'outputs/metal-accuracy')
    ap.add_argument('--mode', choices=['pilot','full'], default='full')
    ap.add_argument('--devices', nargs='+', choices=['mps'], default=['mps'])
    ap.add_argument('--sequences', nargs='+', help='Optional manifest sequence names; default is all 12 videos')
    args = ap.parse_args()
    project = args.project.resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    import tempfile
    import shutil
    args.output_parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f'{args.mode}-{stamp}-', dir=args.output_parent.resolve()))
    shutil.copy2(Path(__file__).resolve(), out / 'runner_snapshot.py')
    os.environ['MPLCONFIGDIR'] = str(out / 'runtime-cache' / 'matplotlib')
    os.environ['YOLO_CONFIG_DIR'] = str(out / 'runtime-cache' / 'ultralytics')
    sys.path.insert(0, str(project))
    import torch
    hardware = {'platform': platform.platform(), 'machine': platform.machine(), 'python': sys.version,
                'torch': torch.__version__, 'mps_built': torch.backends.mps.is_built(),
                'mps_available': torch.backends.mps.is_available(), 'cpu_threads': torch.get_num_threads(),
                'relevant_environment': {k:os.environ.get(k) for k in ['PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','OMP_NUM_THREADS','MKL_NUM_THREADS']},
                'gpu': subprocess.run(['system_profiler','SPDisplaysDataType'],capture_output=True,text=True).stdout,
                'memory_bytes': subprocess.run(['sysctl','-n','hw.memsize'],capture_output=True,text=True).stdout.strip()}
    if 'mps' in args.devices:
        if not hardware['mps_available']:
            raise RuntimeError('MPS unavailable in this process. On this Mac, run in Terminal outside the restricted sandbox.')
        torch.ones(1, device='mps'); torch.mps.synchronize()
    manifest = json.loads((project / 'data/retail-entrance/manifest.json').read_text())
    refs = manifest['entrance_references']['sequences']
    sequences = args.sequences or (['TwoEnterShop2front'] if args.mode == 'pilot' else list(manifest['sequences']))
    unknown = set(sequences) - set(manifest['sequences'])
    if unknown:
        ap.error(f'Unknown sequence names: {sorted(unknown)}')
    protected = {str(p.relative_to(project)): sha(p) for p in sorted((project/'outputs').rglob('report.json'))}
    frozen = {str(p.relative_to(project)): sha(p) for p in [
        project/'tracker.py', project/'counting.py', project/'detection.py', project/'evaluation.py', project/'performance.py',
        project/'configs/retail-metal.json', project/'configs/bytetrack-low-fps.yaml',
        project/'yolo11s.pt', project/'data/retail-entrance/manifest.json', project/'data/retail-entrance/annotations.zip',
        project/'experiments/retail_entrance.py']}
    experiment = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'mode': args.mode,
        'hardware': hardware, 'project': str(project), 'output': str(out), 'devices':args.devices,
        'sequences':sequences, 'all_manifest_sequences_included':set(sequences)==set(manifest['sequences']),
        'runner_sha256':sha(out/'runner_snapshot.py'), 'frozen_sha256':frozen, 'prior_report_sha256':protected,
        'settings': {'model':'yolo11s.pt','imgsz':640,'conf':.1,'iou':.7,'classes':['person','car','truck','bus'],
                     'config':'configs/retail-metal.json','tracker':'configs/bytetrack-low-fps.yaml','frame_sampling':None},
        'timing_definition':'Separate fresh tracker subprocess per video; perf_counter includes Python/import/model startup, decoding, inference, tracking, overlay, CSV and MP4 writing. Excludes output verification and scoring. No detection caches. Each job independently timed; one sequential pass, not a repeated statistical benchmark.',
        'jobs': [], 'status':'running'}
    save(out/'experiment.json', experiment)
    print(f'OUTPUT {out}', flush=True)
    for device in args.devices:
        for sequence in sequences:
            spec = manifest['sequences'][sequence]
            source = project / 'data/retail-entrance' / spec['video']
            if sha(source) != spec['video_sha256']:
                raise RuntimeError(f'Source hash mismatch: {source}')
            folder = out / device / spec['split'] / sequence
            folder.mkdir(parents=True, exist_ok=False)
            cmd = [sys.executable,'-u',str(project/'tracker.py'),str(source),
                   '--device',device,'--model',str(project/'yolo11s.pt'),
                   '--config',str(project/'configs/retail-metal.json'),
                   '--tracker',str(project/'configs/bytetrack-low-fps.yaml'),
                   '--imgsz','640','--conf','0.1','--iou','0.7',
                   '--classes','person','car','truck','bus','--save-observations',
                   '--output',str(folder),'--name',sequence]
            expected = refs[sequence]['frames']
            if args.mode == 'pilot':
                expected = min(250, expected)
                cmd.extend(['--max-frames',str(expected)])
            print(f'START {device} {sequence} {expected} frames', flush=True)
            started = time.perf_counter()
            with (folder/'run.log').open('w') as log:
                result = subprocess.run(cmd, cwd=project, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            elapsed = time.perf_counter()-started
            job = {'device':device,'sequence':sequence,'split':spec['split'],'command':cmd,
                   'elapsed_seconds':elapsed,'returncode':result.returncode,'expected_frames':expected,
                   'source_sha256':spec['video_sha256'],'folder':str(folder.relative_to(out))}
            experiment['jobs'].append(job)
            save(out/'experiment.json', experiment)
            if result.returncode:
                experiment['status']='failed';save(out/'experiment.json', experiment)
                raise RuntimeError(f'Tracker failed; see {folder / "run.log"}')
            record, converted, decoded = parse_record(folder, sequence, expected)
            job.update(frames_processed=expected, output_decoded_frames=decoded,
                       end_to_end_fps=expected/elapsed, counts=record['summary']['counts'])
            save(out/'experiment.json', experiment)
            print(f'DONE {device} {sequence}: {elapsed:.2f}s, {expected/elapsed:.2f} FPS, {record["summary"]["counts"]["person"]}', flush=True)
    if args.mode == 'full':
        from experiments import retail_entrance as metrics
        from experiments import measure_restored as track_metrics
        import zipfile
        scored = {}
        for device in args.devices:
            scored[device] = {}
            for job in experiment['jobs']:
                if job['device'] != device:
                    continue
                sequence = job['sequence']
                _, record, _ = parse_record(out / job['folder'], sequence, job['expected_frames'])
                record['scores'] = metrics.score_events(refs[sequence]['events'],record['events'])
                metrics.audit_events(sequence,record)
                with zipfile.ZipFile(project/'data/retail-entrance/annotations.zip') as archive:
                    summary, raw = track_metrics.track_sequence(archive,sequence,record)
                record['tracking'] = {'summary':summary,'raw':raw}
                save(out/job['folder']/'scored.json', record)
                scored[device][sequence] = record
            aggregates = {}
            for split in ['all','development','validation','holdout']:
                selected = {s:r for s,r in scored[device].items() if split=='all' or manifest['sequences'][s]['split']==split}
                if not selected:
                    continue
                group = aggregate(list(selected.values()),metrics)
                group['tracking'] = track_metrics.combine({s:r['tracking'] for s,r in selected.items()})
                jobs = [j for j in experiment['jobs'] if j['device']==device and j['sequence'] in selected]
                group['elapsed_seconds'] = sum(j['elapsed_seconds'] for j in jobs)
                group['frames'] = sum(j['frames_processed'] for j in jobs)
                group['end_to_end_fps'] = group['frames']/group['elapsed_seconds']
                aggregates[split] = group
            experiment.setdefault('aggregates',{})[device] = aggregates
    for path, digest in {**protected, **frozen}.items():
        if sha(project/path) != digest:
            raise RuntimeError(f'Pre-existing project input/report changed during experiment: {path}')
    experiment['prior_reports_unchanged'] = True
    experiment['status'] = 'complete'
    experiment['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    save(out/'experiment.json', experiment)
    fields = ['device','sequence','split','expected_frames','frames_processed','elapsed_seconds','end_to_end_fps','returncode','folder']
    with (out/'timing.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(experiment['jobs'])
    print('COMPLETE '+str(out/'experiment.json'),flush=True)

if __name__ == '__main__':
    main()
