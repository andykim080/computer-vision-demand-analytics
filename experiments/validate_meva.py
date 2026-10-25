"""Run frozen MEVA boundaries; save four readable videos and one shared report."""
from pathlib import Path
import argparse
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracker import read_report, run, save_report

root = Path(__file__).resolve().parents[1]
source = Path('/Users/andykim/project/propsight/data/meva/demo-excerpts')
output = root / 'outputs/meva'
for quality, label in [('18', 'light'), ('38', 'heavy')]:
    for task, classes in [('vehicle', ['car', 'truck', 'bus']), ('person', ['person'])]:
        name = f'{"vehicles" if task == "vehicle" else "people"}-{label}-compression'
        args = argparse.Namespace(
            video=str(source / f'forward-5fps-crf{quality}.mp4'),
            config=str(root / f'configs/meva-{task}.json'),
            tracker=str(root / ('configs/bytetrack-low-fps.yaml' if task == 'person' else 'configs/bytetrack.yaml')),
            model=str(root / ('yolo11s.pt' if task == 'person' else 'yolo11n.pt')),
            classes=classes, output=str(output), name=name, save_observations=False,
            preview=None, conf=.1, iou=.7, imgsz=1280 if task == 'person' else 640, device='cpu', fps=None, max_frames=None)
        started = time.perf_counter()
        key, record = run(args)
        record['summary']['runtime_seconds'] = time.perf_counter() - started
        references = [('entry', 3.0)] if task == 'vehicle' else [
            ('entry', 1.1), ('entry', 3.7), ('entry', 5.5), ('exit', 8.6)]
        unmatched = list(record['events'])
        matches = []
        for direction, seconds in references:
            candidates = [event for event in unmatched if event['direction'] == direction
                          and abs(float(event['video_seconds']) - seconds) <= 1]
            match = min(candidates, key=lambda event: abs(float(event['video_seconds']) - seconds)) if candidates else None
            if match:
                unmatched.remove(match)
            matches.append(dict(direction=direction, reference_seconds=seconds,
                                predicted_seconds=float(match['video_seconds']) if match else None,
                                reference_status='post-run visual check' if direction == 'exit' else 'pre-existing manual reference'))
        record['validation'] = dict(reference_matches=matches, unmatched_predictions=unmatched)
        report = read_report(output)
        report['runs'][key] = record
        report['validation_context'] = dict(
            scope='Two compression variants of one selected scene, not a held-out benchmark',
            event_tolerance_seconds=1,
            person_reference_scope='Three entries from the original seven-second reference; reverse crossing at 8.6s added by visual review')
        save_report(output, report)
