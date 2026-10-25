# Sustained playback test and timing definitions

Commands and local paths below are relative to the repository root. The test uses an existing local MOT17 archive.

## Measure sustained playback delay

```sh
.venv/bin/python experiments/sustained_metal.py --seconds 300
```

The load test uses original 1920×1080, 30 FPS MOT17-04 frames already stored in `data/human-labeled/mot17-selected.zip`. Its 35-second source sequence repeats to sustain five minutes of load. Every frame is decoded and processed; no predictions are cached and no frames are dropped. Input is native 1080p, while the detector still resizes according to `imgsz=640`. This does not measure neural-network inference at full 1920×1080 resolution.

Scheduled arrivals pace a local playback loop. Timing includes ZIP/JPEG decoding, detection, tracking, drawing, and full-resolution MP4 encoding. Model loading precedes the arrival schedule; first-inference startup is included. Camera capture, network transport, buffering policies and display delay are not measured. Counts in this unrelated load-test view are not an accuracy evaluation.

For a normal constant-frame-rate video, the same timing instrumentation is available directly:

```sh
.venv/bin/python tracker.py /absolute/path/to/video.mp4 \
  --device mps --config /absolute/path/to/camera.json \
  --realtime --save-timing --output outputs/metal-playback
```

- `wall_seconds` is actual processing wall time; `processed_seconds` is source video duration.
- `effective_fps` includes arrival pacing. `active_processing_fps` excludes deliberate waiting.
- Frame age is completion time minus the scheduled arrival time.
- Behind-schedule time is completion time beyond the next frame deadline. It measures the no-drop playback delay; equivalent lag frames are not an allocated camera queue.
- Per-frame CSV and per-minute timing summaries expose accumulated delay and stalls.

`--max-frames` truncates from the beginning. `--fps` overrides source timestamps; it does not subsample frames. A 5 FPS saved webcam excerpt cannot establish native 30 FPS motion accuracy. Upscaling small footage does not create native 1080p detail.
