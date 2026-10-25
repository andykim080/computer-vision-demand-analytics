# Computer Vision for Customer & Demand Analytics

**Repository name:** `computer-vision-demand-analytics`

A computer vision–based analytics system for understanding customer behavior, monitoring foot traffic, and extracting demand insights from visual data.

The current implementation provides recorded-video detection, temporary tracking IDs, and directional crossing counts as inputs for customer and demand analysis. Engagement scoring and demand forecasting are not implemented.

## Metrics on labeled datasets

**Measured on saved evaluation runs, not a new independent benchmark.** The CAVIAR clips below were inspected and used for camera-specific tuning. Results apply to this selected footage and configuration, not to the New York, Korea, or Taiwan gallery clips.

### CAVIAR retail entrance: 12 clips · 16,605 frames · 37 reference crossings

Pretrained YOLO11s/640 + ByteTrack, one detector pass, Apple Metal. Crossing references use published human track IDs and reviewed entrance events.

| Metric | Earlier configuration | Selected configuration |
| --- | ---: | ---: |
| Crossing precision, within 1 second | 51.6% | **94.3%** |
| Crossing recall, within 1 second | 43.2% | **89.2%** |
| Crossing F1, within 1 second | 47.1% | **91.7%** |
| Matched / extra / missed events, within 1 second | 16 / 15 / 21 | **33 / 2 / 4** |
| Identity-supported crossing F1, within 3 seconds | 88.2% | **97.2%** |
| Identity-supported matches / extra / missed, within 3 seconds | 30 / 1 / 7 | **35 / 0 / 2** |
| Entry / exit totals (reference: 19 / 18) | 20 / 11 | **19 / 16** |
| Sum of absolute per-clip directional count errors | 8 | **2** |
| Full-scene tracking HOTA / IDF1 (0–100) | 41.84 / 58.44 | 41.84 / 58.44 |
| Full-scene ID switches | 65 | 65 |
| End-to-end processing throughput | 30.9 FPS | 30.2 FPS |

The one-second metric matches events one-to-one by direction and timing. The three-second metric additionally requires box associations to the labeled human track ID; its wider timing tolerance explains the higher score. It is not personal identity recognition. Tracking scores are separate from counting scores: the selected counter improves crossing results without improving HOTA or IDF1. Labels may omit real background people. Throughput includes process/model startup, decoding, tracking, overlays and output writing for one sequential run; it is not a guaranteed live-camera rate.

### Later MOT17 / UA-DETRAC tracked-box study: adjustment rejected

On **440 later-window native-FPS frames / 4,179 publisher-labeled appearances**, retain the initial class-specific profile:

| Tracked-box metric | Adopted initial profile | Rejected tracker adjustment |
| --- | ---: | ---: |
| Recall | **75.21%** | 75.28% |
| F1 | **79.97%** | 79.71% |
| Precision | **85.38%** | 84.68% |
| Matched / unmatched / missed boxes | **3,143 / 538 / 1,036** | 3,146 / 569 / 1,033 |

Only three extra matches cost 31 extra unmatched predictions. The user retained person track initiation 0.55 and 30-frame retention, rejecting initiation 0.50 with 2.4-second retention. Class cutoffs remain car 0.50, truck 0.40, bus 0.55 and person/motorcycle 0.10. General application defaults were never promoted to the rejected experiment, and no detector weights changed. These box metrics are separate from CAVIAR crossing scores; earlier source windows/cameras were already used in development.

[Review the rollback rationale, actual output comparisons, and unchanged source reports](docs/metrics/confidence-study/README.md).

![MOT17 output comparison: adopted initial profile left; rejected tracker candidate right](docs/metrics/confidence-study/mot17-09-later.jpg)

### Dataset coverage

| Labeled dataset | Use in this project | Available result and limits |
| --- | --- | --- |
| [CAVIAR](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) | Published person boxes/track IDs and reviewed directional crossings; selected 12-clip retail replay | Metrics above. Previously inspected development, validation and holdout-named clips are pooled; this is known-data regression evidence. |
| [MOT17](https://motchallenge.net/data/MOT17/) | Labeled pedestrian footage; MOT17-02/04/09 in the new box study, plus MOT17-04 sustained playback | Native 1920×1080, 30 FPS input; 35 seconds repeated to schedule 9,000 frames over five minutes. **24.0 FPS** active processing and **75.36 s** final backlog. All 9,000 frames retained; YOLO resizes to 640. This run measures local processing load, not detection/tracking accuracy or network latency. |
| [UA-DETRAC](https://sites.google.com/view/daweidu/projects/ua-detrac) | Labeled vehicle footage retained locally for earlier experiments | New class-confidence study includes MVI_20011/40152/63561; later-window aggregate box scores above pool person and vehicle data. Categories are grouped, not scored as car-versus-truck classification. |

[Saved metric values and provenance](docs/metrics/labeled-evaluation.json) are included in the repository. Dataset videos, labels and generated output videos remain local. No detector training was performed for the selected Metal run; pretrained training-data overlap has not been audited. The location-gallery clips have no complete manual labels and are not included in these scores.

## Before and after

Original footage and processed output at the same frame. “Before” means source footage; “after” means tracking overlays, not a comparison of model accuracy.

### Location test replays

Examples from our saved New York, South Korea, and Taiwan test clips. Each pair shows frame 40 (8.00 seconds) of the same recording, at its original 1280×720 resolution.

| Location | Before: original footage | After: tracking output |
| --- | --- | --- |
| Times Square, New York, USA | ![Times Square original footage at night](docs/screenshots/new-york-times-square-before.jpg) | ![Times Square pedestrian and vehicle tracking](docs/screenshots/new-york-times-square-after.jpg) |
| Songridan-gil, Seoul, South Korea | ![Songridan-gil original street footage](docs/screenshots/seoul-songridan-gil-before.jpg) | ![Songridan-gil person and vehicle tracking](docs/screenshots/seoul-songridan-gil-after.jpg) |
| Linkou Old Street, New Taipei City, Taiwan | ![Linkou Old Street original footage](docs/screenshots/taiwan-linkou-before.jpg) | ![Linkou Old Street person and vehicle tracking](docs/screenshots/taiwan-linkou-after.jpg) |

These are fresh short replays of previously collected test clips, using pretrained YOLO11s/1280 and ByteTrack on CPU. They are not recovered images of the deleted historical runs or an independent accuracy benchmark. Lines represent illustrative image-boundary crossings, not verified store entrances. See [source attribution and replay settings](docs/screenshots/README.md#location-test-replays).

### New York observation: idle person at the beginning

A person standing idle near the beginning of the New York footage was reported as not counted. This system counts confirmed line crossings, so an idle person does not increase the counter until they cross the configured boundary. Counts do not measure everyone visible, occupancy, or unique visitors. If the person has no detection box or temporary track ID, that is a separate detection/tracking miss that requires frame-level review.

A confidence-threshold explanation is unconfirmed. The saved New York test plan and this replay use detector confidence **0.10 (10%)**; the active low-FPS ByteTrack profile uses a **0.55 (55%)** new-track threshold, close to the suggested 60%. That threshold can prevent a lower-confidence detection from starting a track, but this has not been verified for the reported idle person. A box's displayed confidence, when present, is its prediction score rather than the configured cutoff. We have not established which mechanism explains the reported person at the start of the clip.

### Known limitation: mannequin detected as a person

The original retail walkthrough illustrates a false positive: a shop mannequin is classified and tracked as a person. A stable tracking ID does not establish that a detection is a real customer. Such detections can distort customer-activity measures and could affect crossing counts if their tracked position crosses the configured line; this image alone does not establish a false crossing event.

| Before: original retail footage | After: mannequin false positive in tracking output |
| --- | --- |
| ![Original retail walkthrough with a shop display](docs/screenshots/retail-crossings-before.jpg) | ![Retail walkthrough showing the reported mannequin classified as a person](docs/screenshots/retail-crossings-after.jpg) |

This existing CAVIAR replay frame is retained as a failure example, not evidence of successful customer identification. It is frame 700 (28.00 seconds) of `OneShopOneWait2front`, using YOLO11s/640 on Apple Metal. Scene-specific exclusion zones and representative person/mannequin labels are possible mitigations; they have not been validated here.

## Current implementation

A single-camera recorded-video prototype using pretrained YOLO11s, ByteTrack, temporary track IDs, and a finite entrance line. The active workflow uses Apple Metal on the local Mac. It detects people, cars, trucks, and buses; crossing counts are not unique visitors or occupancy.

Open [the Metal review](outputs/metal-accuracy/review.html) for the selected accuracy replay, the original eight crossing errors, all 12 output videos, and the sustained native-1080p playback test. Only Metal outputs appear in the active results folder.

## Install

Requires Python 3.10 or newer. From the repository folder:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
computer-vision-demand-analytics --help
```

The installed command is available when `.venv/bin` is on your PATH (for example, after `source .venv/bin/activate`); otherwise use `.venv/bin/computer-vision-demand-analytics` or `.venv/bin/python tracker.py`.

Downloaded datasets, model weights, and generated outputs are excluded from this repository. Supply your own authorized constant-frame-rate recording and calibrate the entrance line for that camera. The `data/` examples and `outputs/` review links below refer to local evaluation artifacts and are not included in a fresh clone. The README screenshots are included under `docs/screenshots/`.

## Run a recording

From the project folder, using the existing virtual environment:

```sh
.venv/bin/python tracker.py data/retail-entrance/holdout/TwoEnterShop2front.mpg \
  --device mps --output outputs/metal-runs --name two-enter-shop2
```

The selected default camera profile is `configs/retail-metal.json`, with YOLO11s/640, detector confidence 0.1, IoU 0.7, and the existing low-FPS ByteTrack file. The source retail videos retain all frames at 25 FPS. No detector training or additional inference pass is enabled.

Compared with `configs/retail-baseline.json`, the selected standard-view profile extends the doorway boundary from x=300 to x=335 in the 384×288 camera image, narrows the deadband from 8 to 3 pixels, allows 0.8 seconds of counting continuity, and requires confidence 0.3 for person crossing measurements. The original person track-start threshold 0.55 and duplicate-box filtering remain. Weak detections still support ByteTrack and remain in observation exports; the extra crossing-confidence condition does not remove them from tracking.

A lower track-start threshold was rejected because it improved one more crossing but produced more unmatched boxes and worse tracking scores. The selected profile prioritizes crossing improvement without that regression. Settings are camera-specific and were tuned on known recordings, not evaluated on a fresh independent holdout.

For a different camera, supply its calibrated JSON with `--config`. The old zoomed CAVIAR view has a separate profile, `configs/retail-baseline-zoomed.json`; the new wide-view adjustments have not been validated for that view. `--preview path.jpg` draws the line on the first frame without inference.

## Repeat the complete Metal accuracy run

```sh
.venv/bin/python experiments/metal_replay.py --mode full --devices mps
```

This runs all 12 videos in `data/retail-entrance/manifest.json` without truncation or temporal sampling. Each invocation creates a new folder. Every video has its own annotated MP4, shared-format report, per-frame observations, log, and scored results. The batch records actual subprocess wall time, verifies output frame counts, preserves source hashes, and computes both the existing three-second identity-supported crossing metric and one-second event timing scores. Human-label tracking scores are separate diagnostics.

Source clips and labels have already been inspected and used for development. Improvement on these clips is regression evidence, not new-site accuracy. Full-scene labels may omit real background people; unmatched boxes can include real unlabeled people and false detections.

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

## Outputs and cleanup

Output names never replace an existing run. Reports are written atomically; use one writer per output folder. Observation CSV export is optional with `--save-observations`; per-frame timing export uses `--save-timing`.

Older CPU-generated result folders were removed at the user's request after confirmation. All source footage, labels, archives, model weights, camera profiles, and backend-independent code tests were retained. The current Metal runner does not depend on those deleted result folders. Historical experiment scripts remain as source history and may require rebuilding their earlier reports before their old multi-stage workflows can resume.

Metal accelerates the detector. Decoding, tracking, drawing, report generation and encoding still use CPU work; deleting CPU comparison outputs does not remove those stages.

## Source data and checks

CAVIAR retail recordings and published annotations remain in `data/retail-entrance`, `data/entrance-improvement`, and `data/counter-confirmation`, including their manifests and annotation archives. Original MOT17 and UA-DETRAC frames/labels remain in `data/human-labeled`; the other saved webcam/demo sources remain under `data/`. Dataset split names are retained for provenance, but previously inspected clips are not a fresh holdout.

Sources: [CAVIAR](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/), [MOT17](https://motchallenge.net/data/MOT17/), and [UA-DETRAC](https://sites.google.com/view/daweidu/projects/ua-detrac). Existing capture manifests retain webcam source attribution.

```sh
.venv/bin/python -m unittest discover -s tests -q
```

This prototype handles local recorded files and arrival-paced replay. Direct RTSP/live-camera ingestion is not implemented. Temporary IDs may change during occlusion; no face recognition, persistent personal identity, cross-camera deduplication, or hourly occupancy claim is supported.

## Class-specific confidence development test

The optional `class_confidence` camera setting applies detector cutoffs by class before ByteTrack. Keep the base `--conf` at or below the smallest class cutoff so lower-confidence people/motorcycles remain available. Existing profiles without this field retain their previous behavior; global defaults and the installed YOLO/ByteTrack libraries are unchanged.

On the same Seoul and Taiwan clips, cars/trucks at 0.55 with people/motorcycles at 0.10 restores the two checked scooters, but a lamp can acquire the unchanged low-confidence bus label. A development search of 50 grid combinations, plus baseline/global/requested controls, ranks by **recall, then F1, then precision**. Exact ties use fewer reported fixture/cargo false-box frames and then lower cutoffs. The best tested combination is **car 0.50, truck 0.40, bus 0.55, person 0.10, motorcycle 0.10**, with bicycle unchanged at 0.10.

| Cutoffs | Foreground box recall | F1 | Precision |
| --- | --- | --- | --- |
| Original 0.10 for all classes | 67.6% | 71.9% | 76.7% |
| Global 0.55 | 50.0% | 65.4% | 94.4% |
| Requested car/truck 0.55; others 0.10 | 67.6% | 76.7% | 88.5% |
| Selected class cutoffs | 70.6% | 78.7% | 88.9% |

These scores use **34 approximate agent-reviewed object appearances in four foreground frames**, with correct-class, one-to-one IoU≥0.5 matching. They are not the earlier crossing metrics, a full-scene accuracy estimate, or an independent test. Reference boxes and scoring boundaries are evaluation-only and never filter predictions. There are no real bus/truck reference targets in this tiny sample, so their recall is unmeasured. Some occluded, partial and moving objects remain missed or mislabeled.

The selected version was rerun with fresh YOLO11s/1280 inference over both complete 18-second clips. It retains the checked scooters and has zero matched instances of the reported lamp/manhole/cargo false labels across these clips. No location exclusions, manual prediction relabeling, training or stationary-object removal is used. All 51 code tests pass, including class cutoffs and low-confidence person/motorcycle track recovery.

Local review: `outputs/class-confidence/review.html`; shared references, all candidates and results: `outputs/class-confidence/report.json`. Only two comparison videos, one report and one review page are generated. The saved test profiles are `configs/class-confidence-seoul-songridan-gil.json` and `configs/class-confidence-taiwan-linkou.json`. For example:

```sh
.venv/bin/python tracker.py data/retail-security-test/songridan-gil.mp4 --config configs/class-confidence-seoul-songridan-gil.json --model yolo11s.pt --imgsz 1280 --conf 0.1 --device cpu --classes person car truck bus motorcycle bicycle --output outputs/runs --name seoul-class-cutoffs
```

`experiments/search_class_confidence.py` records the scoring references and candidate list before the sweep, replays identical cached detector proposals, then runs the selected profile through fresh inference. It refuses to overwrite a completed search. The cache's source/model hashes are verified; older reports and screenshots remain unchanged. Treat the selected thresholds as a development candidate until tested on independent views with more complete labels.

## Larger native-frame-rate sample and iterative tuning

**Saved user selection (2026-10-25): retain the pre-adjustment class-specific profile.** The later-window results for this version are recall **75.21%**, F1 **79.97%**, and precision **85.38%**. The adjusted candidate raised recall by only 0.07 percentage points while lowering F1 by 0.27 and precision by 0.70; it is rejected for use. Both saved camera profiles (`configs/class-confidence-seoul-songridan-gil.json` and `configs/class-confidence-taiwan-linkou.json`) explicitly retain person track initiation **0.55**, vehicle initiation **0.20**, and detector cutoffs **car 0.50, truck 0.40, bus 0.55, person/motorcycle 0.10**. They use the original `configs/bytetrack-low-fps.yaml`: **30-frame retention**, match threshold **0.9**, confidence fusion disabled. No seconds-based retention override is active. Existing duplicate-box filtering, camera counting geometry and application defaults are retained. The latest shared report records this choice in `user_decision`; its historical `selected_profile` field still describes the rejected experiment winner.

Latest local review: **`outputs/larger-confidence-recovery/review.html`**. The first three rounds, including their rejected high-recall result, remain in **`outputs/larger-confidence/review.html`**. Across the study, **1,760 distinct source frames and 27,263 publisher-labeled object appearances** were examined in six source recordings, at their original 25/30 FPS. Object appearances are repeated per-frame labels, not unique visitors. Sources are MOT17 pedestrian streets/shop approaches and UA-DETRAC roads. They appeared in older project benchmarks; this is a broader historical-data regression study, not independent new-camera validation.

Round 1 compared class cutoffs on two four-second segments and retained the existing cutoffs because lower ones restored the reported vehicle-fixture errors. Round 2 added two views and compared track-start thresholds. Round 3 extended all four development segments to eight seconds and adjusted association/retention. The recall-first result lowered person track initiation to 0.25, but its deferred check lost F1/precision and an additional legacy check found 16 roof and 29 fixture false-person boxes. **That version was rejected rather than promoted.**

A fourth round restored the legacy-person guard and tested finer track-start thresholds with retention changes. The best passing development candidate under the requested recall → F1 → precision ordering keeps detector cutoffs **car 0.50, truck 0.40, bus 0.55, person/motorcycle 0.10**. It changes person track initiation from 0.55 to **0.50** and sets track retention to **2.4 seconds**, retaining match threshold 0.9 and disabled confidence fusion. Detector cutoffs and the threshold to create a new track are different settings. The corrected profile keeps zero known roof/fixture false-person boxes, the standing person in 45 frames under one ID, and the prior fixture/cargo/scooter checks intact.

After freezing that correction, later source windows from seconds 8–16 of two recordings supplied **440 frames and 4,179 labeled appearances**:

| Later-window result | Initial class-specific profile | Corrected recall-first candidate |
| --- | --- | --- |
| Tracked-box recall | 75.21% | 75.28% |
| Tracked-box F1 | 79.97% | 79.71% |
| Tracked-box precision | 85.38% | 84.68% |
| Matched boxes | 3,143 | 3,146 |
| Unmatched predictions | 538 | 569 |

The extra three matches cost 31 extra unmatched predictions. This does **not** support a clear general improvement; application defaults remain unchanged. The corrected configuration is saved for investigation in `configs/larger-confidence-recovered-profile.json`. This is an experiment profile consumed by the helper, not a camera-line file for `--config`. The earlier rejected profile is retained separately in `configs/larger-confidence-candidate.json` as evidence, not a recommended setting.

Scores use official pinned TrackEval and published ignore rules. Person/vehicle boxes are grouped; UA-DETRAC vehicle types are not scored as car-versus-truck classification. These numbers are not the earlier doorway-crossing metrics or directly comparable to the four-frame foreground sample. Later windows share cameras and actors with earlier footage, and tracking resets at each segment; this is not continuous live or hourly validation. Motorcycle behavior is supported only by the old small spot checks. No weights, ground-truth labels, installed libraries or prediction-location exclusions were changed.

`experiments/iterate_larger_sample.py` records the first three rounds and blocks development after its deferred check begins. `experiments/recover_larger_sample.py` preserves that failure, runs the corrective round, freezes it before later-window inference and creates the latest review. Native detector results are cached only to replay identical inputs across candidate settings. Each output group retains comparison videos with one shared report and review page; source recordings and historical outputs remain in place.

Published evidence for the confidence studies: [review, rollback rationale, four output comparisons and original reports](docs/metrics/confidence-study/README.md). Historical `selected_profile` fields are preserved; `user_decision` records the adopted initial version.
