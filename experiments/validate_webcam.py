"""Rerun a folder of short recorded clips; keep videos and one shared report."""
import argparse
from argparse import Namespace
import csv
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracker import run, read_report, save_report


def validate(input_dir, output_dir):
    import cv2
    sources = sorted(input_dir.glob('clip-*.mp4'))
    if not sources:
        raise ValueError(f'No clip-*.mp4 recordings in {input_dir}')
    for source in sources:
        started = time.monotonic()
        name, record = run(Namespace(
            video=str(source), config=str(ROOT/'configs/kathy-pub.json'),
            tracker=str(ROOT/'configs/bytetrack-low-fps.yaml'), model=str(ROOT/'yolo11s.pt'),
            classes=['person','car','truck','bus','motorcycle','bicycle'],
            output=str(output_dir), name=source.stem, save_observations=True,
            preview=None, conf=.1, iou=.7, imgsz=1280, device='cpu', fps=None, max_frames=None))
        elapsed = time.monotonic()-started
        observation_path=output_dir/record.pop('observations_file')
        with observation_path.open(newline='') as handle:
            reader=csv.reader(handle)
            columns=next(reader)
            rows=list(reader)
        record['observations']={'columns':columns,'rows':rows}
        cap=cv2.VideoCapture(str(output_dir/record['video']))
        decoded=0
        while cap.read()[0]:
            decoded+=1
        cap.release()
        if decoded != record['summary']['frames_processed']:
            raise RuntimeError(f'{name}: incomplete output video ({decoded} frames)')
        record['validation']={
            'decoded_output_frames':decoded,'complete_decode':True,
            'processing_seconds':round(elapsed,3),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':'Offline same-camera development clips; not an independent accuracy benchmark.'}
        report=read_report(output_dir)
        report['runs'][name]=record
        save_report(output_dir,report)
        observation_path.unlink()
        print(f'Validated {name}: {decoded} frames; observations consolidated in report.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/kathy-pub-five-tests')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/kathy-pub-five-tests')
    args=parser.parse_args()
    validate(args.input,args.output)
