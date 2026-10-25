"""Single-camera recorded video prototype. No persistent identity recognition."""
import argparse
from collections import Counter
from collections import defaultdict
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import uuid
from types import SimpleNamespace

from counting import LineCounter, confirmation_frames

ROOT = Path(__file__).resolve().parent


def vehicle_nms(boxes, confidence, iou):
    """Avoid separate car/truck tracks on the same vehicle; never suppress riders."""
    import cv2
    if not len(boxes):
        return boxes
    rectangles = [[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in boxes.xyxy.tolist()]
    keep = cv2.dnn.NMSBoxes(rectangles, boxes.conf.tolist(), confidence, iou)
    indices = keep.reshape(-1).tolist() if len(keep) else []
    return boxes[indices]


class SeparateVehicleTracker:
    """One detector pass, isolated person/vehicle ByteTrack state and thresholds."""
    def __init__(self, tracker_path, vehicle_settings, person_settings=None,
                 frame_rate=30, track_buffer_seconds=None, person_suppression=None,
                 class_confidence=None):
        import yaml
        from ultralytics.trackers.byte_tracker import BYTETracker
        settings = yaml.safe_load(Path(tracker_path).read_text())
        if settings.get("tracker_type") != "bytetrack":
            raise ValueError("Separate vehicle settings require ByteTrack")
        self.settings = {"person": dict(settings), "vehicle": dict(settings)}
        self.settings["person"].update(person_settings or {})
        for key in ("track_high_thresh", "new_track_thresh"):
            self.settings["vehicle"][key] = vehicle_settings[key]
        self.vehicle_nms_iou = vehicle_settings.get("nms_iou", 0.5)
        self.small_vehicle_height = vehicle_settings.get("small_box_height_fraction", 0)
        self.small_vehicle_conf = vehicle_settings.get("small_box_min_conf", 0)
        if track_buffer_seconds is not None:
            for cfg in self.settings.values():
                # This pinned Ultralytics version takes the buffer directly in frames.
                cfg['track_buffer'] = max(1, math.ceil(frame_rate * track_buffer_seconds))
        self.trackers = {group: BYTETracker(SimpleNamespace(**cfg))
                         for group, cfg in self.settings.items()}
        self.class_evidence = defaultdict(Counter)
        self.person_suppression = person_suppression
        self.person_boxes_suppressed = 0
        self.class_confidence = class_confidence or {}
        self.class_boxes_filtered = Counter()

    def update(self, result, image, names, confidence, iou):
        boxes = result.boxes.cpu().numpy()
        if self.class_confidence:
            from detection import class_confidence_indices
            retained = class_confidence_indices(boxes.data, names, self.class_confidence)
            rejected = set(range(len(boxes))) - set(retained)
            self.class_boxes_filtered.update(names[int(boxes.cls[i])] for i in rejected)
            boxes = boxes[retained]
        person_mask = [names[int(cls)] == "person" for cls in boxes.cls]
        import numpy as np
        mask = np.asarray(person_mask, dtype=bool)
        vehicles = boxes[~mask]
        # Weak, tiny vehicle proposals are particularly vulnerable to street
        # fixtures. This optional camera calibration leaves people untouched.
        keep = ((vehicles.xyxy[:, 3] - vehicles.xyxy[:, 1] >= image.shape[0] * self.small_vehicle_height)
                | (vehicles.conf >= self.small_vehicle_conf))
        people = boxes[mask]
        if self.person_suppression:
            from detection import person_keep_indices
            retained = person_keep_indices(people.data, self.person_suppression)
            self.person_boxes_suppressed += len(people) - len(retained)
            people = people[retained]
        groups = {"person": people, "vehicle": vehicle_nms(vehicles[keep], confidence, self.vehicle_nms_iou)}
        output = []
        for group, detections in groups.items():
            rows = self.trackers[group].update(detections, image)
            active = {f"{group}-{track.track_id}" for track in (
                self.trackers[group].tracked_stracks + self.trackers[group].lost_stracks)}
            for key in list(self.class_evidence):
                if key.startswith(group + "-") and key not in active:
                    del self.class_evidence[key]
            for row in rows:
                identity = f"{group}-{int(row[4])}"
                cls = int(row[6])
                # A motorcycle can look like a car for the first frame. Accumulate
                # evidence rather than permanently freezing that first guess.
                self.class_evidence[identity][cls] += float(row[5])
                label = names[self.class_evidence[identity].most_common(1)[0][0]]
                # Match Ultralytics' normal tracking callback, which clips its
                # output boxes to the frame before rendering and counting.
                coordinates = row[:4].copy()
                coordinates[[0, 2]] = np.clip(coordinates[[0, 2]], 0, image.shape[1])
                coordinates[[1, 3]] = np.clip(coordinates[[1, 3]], 0, image.shape[0])
                output.append((coordinates.tolist(), identity, label, float(row[5]), group))
        return output


def output_name(base, requested):
    """Readable filenames; repeat runs never overwrite existing results."""
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", requested).strip("-").lower()
    if not name:
        raise ValueError("Output name must contain letters or numbers")
    report = read_report(base)
    candidate, number = name, 2
    while (base / f"{candidate}.mp4").exists() or (base / f"{candidate}-observations.csv").exists() or candidate in report["runs"]:
        candidate = f"{name}-{number}"
        number += 1
    return candidate


def read_report(base):
    path = base / "report.json"
    if not path.exists():
        return {"format_version": 1, "runs": {}}
    report = json.loads(path.read_text())
    if not isinstance(report, dict) or not isinstance(report.get("runs"), dict):
        raise ValueError(f"Invalid report: {path}")
    return report


def save_report(base, report):
    """Replace only after writing a complete report, preserving prior run records."""
    base.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=base, suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(report, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, base / "report.json")
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


def load_config(path):
    config = json.loads(Path(path).read_text())
    for field in ("property_id", "camera_id"):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    points = config.get("line")
    if not isinstance(points, list) or len(points) != 2:
        raise ValueError("line requires two normalized [x, y] endpoints")
    for point in points:
        if (not isinstance(point, list) or len(point) != 2 or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) or
                    not 0 <= v <= 1 for v in point)):
            raise ValueError("Line coordinates must be finite values between 0 and 1")
    if points[0] == points[1]:
        raise ValueError("Line endpoints must differ")
    if config.get("entry_side", "positive") not in ("positive", "negative"):
        raise ValueError("entry_side must be positive or negative")
    for field, default in (("deadband_pixels", 8), ("max_gap_seconds", 0.5)):
        value = config.get(field, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
    confirm = config.get("confirm_frames", 3)
    if not isinstance(confirm, int) or confirm < 1:
        raise ValueError("confirm_frames must be a positive integer")
    for field in ('confirm_seconds', 'track_buffer_seconds'):
        value = config.get(field)
        if field in config and (not isinstance(value, (int, float)) or
                                  not math.isfinite(value) or value <= 0):
            raise ValueError(f'{field} must be finite and positive')
    crop = config.get('detection_crop')
    if crop is not None:
        if (not isinstance(crop, list) or len(crop) != 4 or
                any(not isinstance(v, int) or v < 0 for v in crop) or
                crop[0] >= crop[2] or crop[1] >= crop[3]):
            raise ValueError('detection_crop requires pixel [left, top, right, bottom]')
        if config.get('person_detection') or config.get('vehicle_tracking') is None:
            raise ValueError('detection_crop requires separate tracking without extra person views')
    vehicle = config.get("vehicle_tracking")
    if vehicle is not None:
        if not isinstance(vehicle, dict):
            raise ValueError("vehicle_tracking must be an object")
        for key in ("track_high_thresh", "new_track_thresh"):
            value = vehicle.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"vehicle_tracking.{key} must be in (0, 1]")
        value = vehicle.get("initial_confirm_frames", confirm)
        if not isinstance(value, int) or value < 1:
            raise ValueError("vehicle_tracking.initial_confirm_frames must be positive")
        value = vehicle.get("nms_iou", 0.5)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("vehicle_tracking.nms_iou must be in (0, 1]")
        value = vehicle.get("deadband_pixels", config.get("deadband_pixels", 8))
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("vehicle_tracking.deadband_pixels must be finite and nonnegative")
        for key in ('small_box_height_fraction', 'small_box_min_conf'):
            value = vehicle.get(key, 0)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'vehicle_tracking.{key} must be in [0, 1]')
    person = config.get('person_tracking', {})
    if not isinstance(person, dict) or set(person) - {'new_track_thresh', 'track_high_thresh'}:
        raise ValueError('person_tracking allows new_track_thresh and track_high_thresh')
    for key, value in person.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f'person_tracking.{key} must be in (0, 1]')
    suppression = config.get('person_suppression')
    if suppression is not None:
        if (not isinstance(suppression, dict) or
                set(suppression) - {'iou', 'nested_height_ratio'} or vehicle is None):
            raise ValueError('person_suppression requires separate tracking and allows iou/nested_height_ratio')
        for key, value in suppression.items():
            if (not isinstance(value, (int, float)) or not math.isfinite(value) or
                    not (0 < value <= 1 if key == 'iou' else 0 <= value < 1)):
                raise ValueError(f'Invalid person_suppression.{key}')
    thresholds = config.get('class_confidence')
    if thresholds is not None:
        if not isinstance(thresholds, dict) or not thresholds or vehicle is None:
            raise ValueError('class_confidence requires separate tracking and a nonempty class-to-cutoff object')
        for name, value in thresholds.items():
            if (not isinstance(name, str) or not name.strip() or isinstance(value, bool) or
                    not isinstance(value, (int,float)) or not math.isfinite(value) or not 0 < value <= 1):
                raise ValueError('class_confidence requires nonempty class names and finite cutoffs in (0, 1]')
    views = config.get('person_detection')
    if views is not None:
        if vehicle is None:
            raise ValueError('person_detection requires vehicle_tracking (separate tracking)')
        if not isinstance(views, dict) or not isinstance(views.get('model'), str):
            raise ValueError('person_detection requires a model filename')
        passes = views.get('passes')
        if not isinstance(passes, list) or not passes or len(passes) > 4:
            raise ValueError('person_detection requires 1–4 extra passes')
        for spec in passes:
            if not isinstance(spec, dict) or not isinstance(spec.get('imgsz'), int) or spec['imgsz'] < 32:
                raise ValueError('Each person pass requires imgsz >= 32')
            if not isinstance(spec.get('flip', False), bool):
                raise ValueError('Person pass flip must be true or false')
        if not isinstance(views.get('tiles', False), bool):
            raise ValueError('person_detection.tiles must be true or false')
        if not isinstance(views.get('tile_imgsz', 960), int) or views.get('tile_imgsz', 960) < 32:
            raise ValueError('tile_imgsz must be >= 32')
        value = views.get('merge_iou', .5)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError('person_detection.merge_iou must be in (0, 1]')
    minimum = config.get('person_counting_min_confidence', 0)
    if not isinstance(minimum, (int, float)) or not math.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError('person_counting_min_confidence must be in [0, 1]')
    return config


def run(args):
    import cv2
    from performance import PlaybackTiming

    run_started = time.perf_counter()

    source = Path(args.video).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Recorded video not found: {source}")
    config = load_config(args.config)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    writer = None
    try:
        ok, first = cap.read()
        if not ok:
            raise ValueError("Video contains no decodable frames")
        height, width = first.shape[:2]
        crop = config.get('detection_crop')
        if crop and (crop[2] > width or crop[3] > height):
            raise ValueError('detection_crop exceeds the source image')
        points = [tuple(round(v * (size - 1)) for v, size in zip(p, (width, height)))
                  for p in config["line"]]
        fps = args.fps if args.fps else cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS missing; supply --fps with the recording frame rate")
        counter = LineCounter(*points, entry_side=config.get("entry_side", "positive"),
                              deadband=config.get("deadband_pixels", 8),
                              confirm_frames=confirmation_frames(config, fps),
                              max_gap_frames=max(1, round(fps * config.get("max_gap_seconds", 0.5))))
        if args.preview:
            preview = first.copy()
            cv2.line(preview, *points, (0, 255, 255), 3)
            for label, point in zip(("A", "B"), points):
                cv2.putText(preview, label, point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            target = Path(args.preview)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), preview):
                raise ValueError("Could not write preview")
            print(f"Line preview: {target.resolve()}")
            return
        from ultralytics import YOLO
        model = YOLO(args.model)
        names = model.names
        selected = {i: name for i, name in names.items() if name in args.classes}
        if set(selected.values()) != set(args.classes):
            raise ValueError("Model must include the requested class names")
        thresholds = config.get('class_confidence', {})
        if set(thresholds)-set(names.values()):
            raise ValueError('class_confidence contains class names absent from this model')
        if thresholds and args.conf > min(thresholds.values()):
            raise ValueError('Base --conf must not exceed the smallest class_confidence cutoff')
        vehicle_settings = config.get("vehicle_tracking")
        separate = SeparateVehicleTracker(
            args.tracker, vehicle_settings, config.get('person_tracking'),
            frame_rate=fps if config.get('track_buffer_seconds') else 30,
            track_buffer_seconds=config.get('track_buffer_seconds'),
            person_suppression=config.get('person_suppression'),
            class_confidence=thresholds) if vehicle_settings else None
        from detection import PersonViews
        person_views = PersonViews(config['person_detection'], ROOT) if config.get('person_detection') and 'person' in selected.values() else None
        vehicle_counter = LineCounter(
            *points, entry_side=config.get("entry_side", "positive"),
            deadband=vehicle_settings.get("deadband_pixels", config.get("deadband_pixels", 8)),
            confirm_frames=confirmation_frames(config, fps),
            initial_confirm_frames=vehicle_settings.get("initial_confirm_frames", confirmation_frames(config, fps)),
            max_gap_frames=max(1, round(fps * config.get("max_gap_seconds", 0.5)))) if separate else None
        if args.conf > 0.1:
            print("Note: conf > 0.1 removes some low-score detections ByteTrack could recover.")
        run_id = uuid.uuid4().hex[:12]
        base = Path(args.output)
        base.mkdir(parents=True, exist_ok=True)
        output = output_name(base, getattr(args, "name", None) or source.stem)
        video_path = base / f"{output}.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise ValueError("MP4 encoder unavailable")
        observations_fields = ["run_id", "property_id", "camera_id", "frame", "video_seconds", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"]
        events = []
        totals = {name: Counter(entry=0, exit=0) for name in args.classes}
        tracked_observations = Counter({name: 0 for name in args.classes})
        stable_classes = {}
        frame = 0
        observation_path = base / f"{output}-observations.csv"
        observation_context = observation_path.open("w", newline="") if getattr(args, "save_observations", False) else nullcontext(None)
        with observation_context as obs_file:
            obs = csv.DictWriter(obs_file, fieldnames=observations_fields) if obs_file else None
            if obs:
                obs.writeheader()
            image = first
            timing = PlaybackTiming(fps, paced=getattr(args, 'realtime', False))
            while True:
                frame_started = timing.begin_frame(frame)
                if frame:
                    ok, image = cap.read()
                    if not ok:
                        break
                counter.prune(frame)
                if vehicle_counter:
                    vehicle_counter.prune(frame)
                stable_classes = {k: v for k, v in stable_classes.items() if k in counter.states}
                if separate:
                    detector_image = image[crop[1]:crop[3], crop[0]:crop[2]] if crop else image
                    result = model.predict(detector_image, classes=list(selected), conf=args.conf,
                                           iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False)[0]
                    if crop:
                        result.boxes.data = result.boxes.data.clone()
                        result.boxes.data[:, [0, 2]] += crop[0]
                        result.boxes.data[:, [1, 3]] += crop[1]
                    if person_views:
                        result = person_views.predict(result, model, image, args.device, args.conf, args.iou)
                    observations = separate.update(result, image, names, args.conf, args.iou)
                else:
                    result = model.track(image, persist=True, tracker=str(args.tracker), classes=list(selected),
                                         conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                                         device=args.device, verbose=False)[0]
                    boxes = result.boxes
                    observations = []
                    if boxes is not None and boxes.id is not None:
                        for xyxy, identity, cls, confidence in zip(boxes.xyxy.cpu().tolist(), boxes.id.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
                            local_id = int(identity)
                            name = stable_classes.setdefault(local_id, names[int(cls)])
                            observations.append((xyxy, local_id, name, confidence, "default"))
                annotated = image.copy()
                if observations:
                    for xyxy, local_id, name, confidence, group in observations:
                        track_id = f"{config['camera_id']}:{run_id}:{local_id}"
                        x1, y1, x2, y2 = xyxy
                        common = dict(run_id=run_id, property_id=config["property_id"], camera_id=config["camera_id"], frame=frame,
                                      video_seconds=round(frame / fps, 6), track_id=track_id, **{"class": name})
                        if obs:
                            obs.writerow(dict(common, confidence=round(confidence, 5), x1=x1, y1=y1, x2=x2, y2=y2))
                        tracked_observations[name] += 1
                        active_counter = vehicle_counter if group == "vehicle" else counter
                        # Low-score boxes remain available to ByteTrack and the
                        # observation export, but need not establish a crossing.
                        eligible = name != 'person' or confidence >= config.get('person_counting_min_confidence', 0)
                        direction = active_counter.update(local_id, ((x1 + x2) / 2, y2), frame) if eligible else None
                        if direction:
                            totals[name][direction] += 1
                            events.append(dict(common, direction=direction))
                        color = (40, 220, 40) if name == "person" else (255, 150, 30)
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.circle(annotated, (int((x1+x2)/2), int(y2)), 4, color, -1)
                        display_id = str(local_id).replace("person-", "P").replace("vehicle-", "V")
                        cv2.putText(annotated, f"{name} #{display_id}", (int(x1), max(18, int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
                cv2.line(annotated, *points, (0, 255, 255), 2)
                if config.get('compact_overlay', False):
                    label = 'IN/OUT  |  ' + '  |  '.join(
                        f"{name} {counts['entry']}/{counts['exit']}" for name, counts in totals.items())
                    scale = min(.5, width / 2400)
                    cv2.rectangle(annotated, (0, 0), (width, 27), (20, 20, 20), -1)
                    cv2.putText(annotated, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 1)
                else:
                    for row, (name, counts) in enumerate(totals.items()):
                        cv2.putText(annotated, f"{name}: IN {counts['entry']} OUT {counts['exit']}", (15, 30 + 30*row), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                writer.write(annotated)
                timing.finish_frame(frame, frame_started)
                frame += 1
                if frame % 100 == 0:
                    print(f"Processed {frame} frames ({frame / fps:.1f} video seconds)", flush=True)
                if args.max_frames and frame >= args.max_frames:
                    break
        summary = dict(run_id=run_id, processed_at_utc=datetime.now(timezone.utc).isoformat(), source=str(source),
                       frames_processed=frame, processed_seconds=frame/fps, fps=fps, model=args.model,
                       device=args.device, performance=timing.summary(),
                       detector_conf=args.conf, detector_iou=args.iou, imgsz=args.imgsz, config=config,
                       tracker_config=Path(args.tracker).read_text(), counts=totals,
                       tracked_observations=tracked_observations,
                       separate_tracker_settings=separate.settings if separate else None,
                       person_boxes_suppressed=separate.person_boxes_suppressed if separate else 0,
                       class_boxes_filtered=dict(separate.class_boxes_filtered) if separate else {},
                       timestamp_basis="zero-based frame index / nominal FPS; not wall-clock time",
                       limitations=["Single camera; temporary IDs are not unique visitors", "Counts are crossings, not occupancy", "ID switches, occlusions and long gaps can miss or duplicate counts", "Variable-frame-rate input requires conversion to constant frame rate"])
        writer.release()
        writer = None
        summary['wall_seconds'] = time.perf_counter() - run_started
        record = dict(video=video_path.name, summary=summary, events=events)
        if getattr(args, 'save_timing', False):
            record['timing_file'] = f'{output}-timing.csv'
            timing.save_csv(base / record['timing_file'])
        if obs is not None:
            record["observations_file"] = observation_path.name
        report = read_report(base)
        report["runs"][output] = record
        save_report(base, report)
        for name, observations in tracked_observations.items():
            if observations == 0:
                print(f"Warning: no tracked {name} observations. Zero crossings do not establish absence of traffic.")
        print(json.dumps({"video": str(video_path.resolve()), "report": str((base / "report.json").resolve()), "counts": totals}, indent=2))
        return output, record
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to a local constant-frame-rate recorded video")
    parser.add_argument("--config", default=str(ROOT / "configs/retail-metal.json"))
    parser.add_argument("--tracker", default=str(ROOT / "configs/bytetrack-low-fps.yaml"))
    parser.add_argument("--model", default=str(ROOT / "yolo11s.pt"), help="Pretrained COCO YOLO weights or local checkpoint")
    parser.add_argument("--classes", nargs="+", default=["person", "car", "truck", "bus"],
                        help="Model class names to include (default: person car truck bus)")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "runs"))
    parser.add_argument("--name", help="Readable video name; repeated names get -2, -3, etc.")
    parser.add_argument("--save-observations", action="store_true", help="Also export per-frame CSV for debugging (off by default)")
    parser.add_argument("--preview", help="Write first frame with counting line, then exit without inference")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="mps", help="cpu, mps (Apple GPU), or 0 (CUDA GPU)")
    parser.add_argument("--fps", type=float, help="Override missing/incorrect recording FPS")
    parser.add_argument("--max-frames", type=int, help="Limit frames for a short trial")
    parser.add_argument("--realtime", action="store_true", help="Pace prerecorded frames at recording FPS and measure delay; never skip frames")
    parser.add_argument("--save-timing", action="store_true", help="Export per-frame processing time and playback delay CSV")
    args = parser.parse_args()
    if not 0 < args.conf <= 1 or not 0 < args.iou <= 1:
        parser.error("conf and iou must be in (0, 1]")
    if args.imgsz < 32 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("imgsz must be >=32; max-frames must be positive")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("fps must be finite and positive")
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
