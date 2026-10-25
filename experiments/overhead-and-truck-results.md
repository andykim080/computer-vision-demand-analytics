# Overhead parking and truck diagnosis — 2026-10-24

## Findings

The original road run explicitly filtered detections to `person` and `car`. YOLO11n detected the truck in its first raw frame at confidence 0.773, but the application discarded that class. The default allowlist in `tracker.py` now includes `person`, `car`, `truck`, and `bus`.

The overhead failure is a detector classification failure, before ByteTrack. On sampled frames 0, 40, and 80, the general YOLO11n model mostly labeled vehicles as toilets at image size 640 and cell phones at size 1280. YOLO11s also failed. A 90-degree rotation did not provide confident vehicle detections. Lowering confidence to 0.01 exposed these predictions but did not recover useful cars. Low-confidence raw outputs are diagnostics, not adopted production thresholds.

## Controlled runs

| Run | Footage / frames | Detector | Settings / result |
| --- | --- | --- | --- |
| `b9562b43a6ea` | Overhead / 82 | YOLO11n | 640px; zero tracked cars |
| `overhead-yolo11s-1280/cd7dedf5554a` | Same overhead / 82 | YOLO11s | 1280px; still zero tracked cars |
| `overhead-parking-specialist/788efd4df20d` | Same overhead / 82 | Official parking-specific YOLO checkpoint | 640px; 3,111 car observation rows and 335 van rows; first frame 39 car tracks and 4 van tracks; one car exit at frame 41 |
| `road-expanded-classes/0a48b04d4190` | Same road excerpt / 90 | YOLO11n | 640px; expanded class allowlist; truck observed in all 90 frames, one truck entry at frame 80 |

The run IDs above are historical record identifiers. Retained working videos are now `outputs/demos/overhead-parking.mp4` and `outputs/demos/road-traffic.mp4`; their settings, events, and original observations are consolidated in `outputs/demos/report.json`. The videos were moved without re-encoding. Failed and superseded run files were subsequently removed at the user’s request; the table retains their historical results. Only the successful pedestrian baseline and improved parking/road runs remain. All new output videos were decoded and their frame counts checked against their summaries. Eleven counting tests passed after the change.

**Rows are repeated observations, not distinct vehicles or accuracy scores.** No independent ground-truth labeling or precision/recall measurement was performed. The parking clip and specialized checkpoint come from the same official demo collection; overlap with model training is unknown. The apparent improvement must be validated on independent authorized camera footage. Vehicle class flicker and false positives remain possible. The general road model still shows a person false positive inside a windshield.

The counting line and direction are unchanged example settings; entry/exit events describe crossings of that line, not a calibrated property boundary.

## What was changed

- Application default class allowlist: added truck and bus in `tracker.py`.
- General detector trials: switched existing pretrained checkpoints (nano → small), changed image size (640 → 1280), inspected raw predictions at confidence 0.01, and tested rotation on sampled frames.
- Overhead working trial: selected a separate, already-trained parking checkpoint, `parking-yolo.pt`. This model uses `pedestrian` rather than `person`, and has a separate `van` class; the command explicitly chooses those names. Group label `people` is excluded because it is not an individual pedestrian.
- **No training, weight updates, library source changes, or ByteTrack parameter changes were performed.** Association thresholds stayed at high=0.25, low=0.1, new=0.25, buffer=30 frames, match=0.8, fuse_score=true. Detector confidence stayed 0.1 for full tracking runs.

## Files and reproduction

From the project root:

```sh
.venv/bin/python tracker.py data/demo-cars.mp4 --model parking-yolo.pt --classes pedestrian car van truck bus --output outputs/demos --name overhead-parking
.venv/bin/python tracker.py data/demo-road.mp4 --max-frames 90 --output outputs/demos --name road-traffic
.venv/bin/python tracker.py data/demo-cars.mp4 --model yolo11s.pt --imgsz 1280 --output outputs/diagnostics --name overhead-yolo11s-1280
.venv/bin/python experiments/probe_detectors.py
```

The raw comparison is preserved in `experiments/raw-detector-comparison.json`; obsolete diagnostic JPGs were removed. `probe_detectors.py` compares all classes so excluded/misclassified objects can be diagnosed before tracking.

Checkpoint sources:
- YOLO11n: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
- YOLO11s: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
- Parking-specific checkpoint, locally named `parking-yolo.pt`: https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_parking_model.pt

Installed library source is under `.venv/lib/python3.12/site-packages/ultralytics/`. The ByteTrack implementation is `trackers/byte_tracker.py`; YOLO model loading is `models/yolo/model.py`, with the architecture/task implementations in `nn/`. These installed files were only inspected, not modified. Our editable files are `tracker.py`, `configs/bytetrack.yaml`, `configs/example.json`, and `counting.py`.

True detector fine-tuning would require labeled overhead images, a training run that updates checkpoint weights, and evaluation on independent held-out recordings. This has not been done.
