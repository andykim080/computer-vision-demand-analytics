"""Pace existing original 1080p/30fps MOT17 frames for a sustained Metal load test.

The 35-second MOT17-04 sequence repeats; all frames are processed, with no drops,
no inference cache, no upscaling, and no temporal subsampling. This measures local
playback backlog, not RTSP/network/camera latency or counting accuracy on this view.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'tracker.py').is_file():
    ROOT = Path('/Users/andykim/project/Propsight 2')
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=300)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--output-parent', type=Path, default=ROOT/'outputs/metal-accuracy')
    args = ap.parse_args()
    if args.imgsz < 32:
        ap.error('--imgsz must be at least 32')
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        ap.error('--seconds must be finite and positive')
    args.output_parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='sustained-1080p-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-', dir=args.output_parent))
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'work/metal-accuracy/ultralytics')
    os.environ['MPLCONFIGDIR'] = str(ROOT/'work/metal-accuracy/matplotlib')
    import cv2
    import numpy as np
    import torch
    import tracker
    if not torch.backends.mps.is_available():
        raise RuntimeError('Apple Metal is unavailable to this process')
    sequence = 'MOT17-04'
    metadata = json.loads((ROOT/'data/human-labeled/manifest.json').read_text())['sequences'][sequence]
    native_fps = metadata['fps']
    count = math.ceil(args.seconds * native_fps)
    archive_path = ROOT/'data/human-labeled/mot17-selected.zip'
    config = json.loads((ROOT/'configs/retail-metal.json').read_text())
    config.update(property_id='local-playback-load-test', camera_id='mot17-04-playback',
                  line=[[.2,.7],[.8,.7]], deadband_pixels=11, confirm_seconds=.12)
    config_path = out/'load-test-config.json'
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    original_capture = cv2.VideoCapture

    class ZipCapture:
        def __init__(self):
            self.archive = zipfile.ZipFile(archive_path)
            self.position = 0
        def isOpened(self):
            return self.archive is not None
        def get(self, prop):
            return {cv2.CAP_PROP_FPS:native_fps, cv2.CAP_PROP_FRAME_COUNT:count,
                    cv2.CAP_PROP_FRAME_WIDTH:metadata['width'], cv2.CAP_PROP_FRAME_HEIGHT:metadata['height']}.get(prop,0)
        def read(self):
            if self.position >= count:
                return False, None
            index = self.position % metadata['frames'] + 1
            encoded = self.archive.read(f'{sequence}/img1/{index:06d}.jpg')
            image = cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (1080,1920):
                raise RuntimeError('Expected an original 1920x1080 frame')
            self.position += 1
            return True,image
        def release(self):
            if self.archive:
                self.archive.close()
                self.archive = None

    cv2.VideoCapture = lambda source,*a,**kw: ZipCapture() if Path(str(source)) == archive_path else original_capture(source,*a,**kw)
    spec = SimpleNamespace(video=str(archive_path),config=str(config_path),tracker=str(ROOT/'configs/bytetrack-low-fps.yaml'),
        model=str(ROOT/'yolo11s.pt'),classes=['person','car','truck','bus'],output=str(out),
        name='native-1080p-30fps-metal',save_observations=False,preview=None,conf=.1,iou=.7,
        imgsz=args.imgsz,device='mps',fps=None,max_frames=count,realtime=True,save_timing=True)
    started = time.perf_counter()
    try:
        name,record = tracker.run(spec)
    finally:
        cv2.VideoCapture = original_capture
    wall = time.perf_counter()-started
    summary = dict(source_archive=str(archive_path),sequence=sequence,source_native_fps=native_fps,
        source_size=[1920,1080],original_sequence_frames=metadata['frames'],
        original_sequence_seconds=metadata['frames']/native_fps,scheduled_frames=count,
        scheduled_video_seconds=count/native_fps,device='mps',model='yolo11s.pt',detector_imgsz=args.imgsz,
        wall_seconds_including_model_loading=wall,performance=record['summary']['performance'],
        video=record['video'],timing_file=record['timing_file'],source_archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        limitations=['Original 35-second sequence repeated to sustain load, not five minutes of independent footage.',
          'Native 1080p input; detector resizes according to imgsz. This is not full-resolution neural-network inference.',
          'Scheduled file playback with serial JPEG/ZIP decoding, inference, tracking, overlays and full 1080p MP4 encoding. Network, camera capture and display latency are not included.',
          'No dropped frames and no cached model predictions. Repeated camera content may retain tracker state across loop boundaries; counts are not an accuracy measurement.'])
    cap=original_capture(str(out/record['video']));decoded=0
    while cap.read()[0]:
        decoded+=1
    cap.release()
    if decoded!=count or record['summary']['frames_processed']!=count:
        raise RuntimeError(f'Incomplete playback output: {decoded}/{count}')
    summary['decoded_output_frames']=decoded
    (out/'sustained-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SUSTAINED_RESULT',json.dumps({'output':str(out),'performance':summary['performance'],'decoded_frames':decoded}),flush=True)


if __name__ == '__main__':
    main()
