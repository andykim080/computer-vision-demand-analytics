# MEVA validation — 2026-10-24

Inputs (read only): `/Users/andykim/project/propsight/data/meva/demo-excerpts/forward-5fps-crf18.mp4` and `forward-5fps-crf38.mp4`. Both contain the same 10-second scene, 50 frames at 5 FPS, with different compression. This is one scene, not two independent samples.

## Initial baseline settings and references

YOLO11n at image size 640, detector confidence 0.1, IoU 0.7; unchanged ByteTrack configuration and counting logic. Separate counting runs for vehicles and people because the prototype supports one boundary per run. Vehicle allowlist: car/truck/bus; pedestrian allowlist: person. No training or model tuning on this clip.

Boundaries were copied from the existing project's manual demonstration references before inference:
- Vehicle: x=0.7, y=0.7–0.8, entry left to right; expected entry around 3.0 seconds.
- People: x=0.421875, y=0.80–0.98, entry right to left; expected entries around 1.1, 3.7, and 5.5 seconds. The old reference covered only the first seven seconds.
- After inference, full-source visual review confirmed an additional left-to-right crossing around 8.6 seconds. This fourth pedestrian event is a post-run observation, not a pre-frozen test label.
- Counter: 8-pixel deadband, three confirmation frames, maximum observation gap 0.5 seconds. Entry/exit are configured direction labels at artificial lines, not actual facility entry/exit.

## Results

| Version | Vehicle predictions | Person predictions | Timing |
| --- | --- | --- | --- |
| CRF18, lighter compression | 1 entry, 0 exits | 3 entries, 1 exit | vehicle 3.6s; people 1.8, 4.4, 6.2, 9.2s |
| CRF38, heavier compression | 1 entry, 0 exits | 3 entries, 1 exit | identical event times |

All four pre-existing reference events match within the pre-existing 1-second tolerance in each version. The extra pedestrian exit agrees with the post-run visual review. Confirmation adds roughly 0.6–0.7 seconds to these reported times at 5 FPS; timestamps mark confirmation rather than the exact physical crossing.

Initial baseline failure: the leading person loses ID continuity during a turn. Each person-only run creates five temporary IDs for the three foreground people, including short fragments. Equal event totals therefore do not mean stable identity tracking. No complete box-level precision/recall evaluation was performed; not every visible background person/vehicle is labeled. Some stationary vehicles may switch between car and truck labels.

All four annotated outputs were fully decoded: 50 frames each. Initial baseline runs took approximately 2.1–5.1 seconds per 10 seconds of input on this laptop, including setup/export. This is an offline timing check at 5 FPS, not a live-network, sustained throughput, or 4–5-camera capacity test. It does not establish performance on higher frame rates, other angles, nighttime footage, or weather.

## Outputs and reproduction

Results are consolidated in `outputs/meva/`:
- `people-light-compression.mp4` and `people-heavy-compression.mp4`.
- `vehicles-light-compression.mp4` and `vehicles-heavy-compression.mp4`.
- One `report.json` contains each run's summary, settings, events, original observations, reference matching, timing, and caveats.

The random-ID folders, duplicate summaries, individual CSV files, and inspection snapshots were consolidated/removed at the user's request. All four videos were moved without re-encoding and verified byte-for-byte. Temporary IDs remain inside records for traceability.

Run `.venv/bin/python experiments/validate_meva.py` from this project root. Source paths are explicit in that script. `configs/meva-vehicle.json` and `configs/meva-person.json` preserve the boundaries. New validation runs append to the same shared report, with numbered video names to preserve prior runs. Per-frame observations are not exported unless requested. Source repository files were not edited.

## Attribution

MEVA dataset, Kitware Inc. and IARPA, CC BY 4.0, as documented in the source project's `docs/datasets.md` and `data/meva/LICENSE-MEVA.txt`. These excerpts derive from source seconds 132–142 of `2018-03-13.17-05-00.17-10-00.school.G328.r13.avi`, re-encoded silently at 5 FPS, CRF18/CRF38. Our derivatives add tracking boxes, temporary IDs, counting lines, and totals. Citation: Kellie Corona, Katie Osterdahl, Roderic Collins, Anthony Hoogs, “MEVA: A Large-Scale Multiview, Multimodal Video Dataset for Activity Detection,” WACV 2021, pp. 1060–1068. Publisher: https://mevadata.org/ ; license: https://mevadata.org/resources/MEVA-data-license.txt

## Live footage options

TxDOT's official camera directory links live metro-area feeds: https://www.txdot.gov/discover/live-traffic-cameras.html . Houston TranStar's map also has camera/motion layers: https://traffic.houstontranstar.org/layers/ . Public browser viewing alone does not establish permission for automated ingestion or redistribution; verify the selected feed's access and reuse terms before connecting it.

This prototype still accepts local recorded files only. Live URL ingestion, reconnect handling, processing lag, and dropped-frame behavior have not been implemented here. An accessible authorized continuous stream (HLS/RTSP/MJPEG), or a permitted downloadable clip, is needed for a live test; periodic still images cannot validate trajectory continuity.

The original `propsight/AGENTS.md` says not to download YouTube footage. YouTube support exists technically in upstream tooling, but platform permissions and the project's source policy govern whether to use it. Use owner-supplied original footage or an authorized direct camera feed rather than extracting a YouTube stream.


## Background-person correction after user review

The user identified a red-shirted person near the parked vehicles who was absent from the person overlay. The earlier crossing validation only covered the foreground boundary and did not establish detection coverage for background people.

At the original 1920-pixel width this person's box is about 50 pixels tall; reducing the image to width 640 leaves about 17 pixels. Five approximate boxes were marked in source frames 0, 10, 20, 30, and 49 before comparing detector settings. IoU >=0.3 was used for these coarse spot checks. YOLO11n/640 produced no target detections at confidence >=0.1 in those samples. YOLO11s/1280 detected the target in all five samples. This is targeted development data, not an independent accuracy benchmark.

Full-clip comparisons in both compression versions:
- YOLO11s/1280 plus original ByteTrack settings: target region tracked in 44/50 frames, two IDs per target, missed frames 15–20.
- YOLO11s/1280 plus the low-FPS association settings: 45/50 frames, **one ID before and after the gap**, missed frames 15–19 (3.0–3.8 seconds). Five manually marked keyframes have a matching target track.
- Detector confidence remains 0.1; detector weights were not trained or edited. `configs/bytetrack-low-fps.yaml` raises match_thresh from 0.8 to 0.9 and sets fuse_score=false; other ByteTrack settings remain unchanged. This more lenient association can also cause wrong matches in other scenes and must be validated per camera. The general `configs/bytetrack.yaml` is unchanged.
- Coverage is based on matching the source region x=1300–1430, bottom Y=700–820 across the clip, with five coarse box spot checks. It is not a densely labeled recall/precision score. Visibility and partial occlusion during frames 15–19 were not independently annotated frame by frame.
- Foreground counts remain **3 entries, 1 exit** in both versions. The background person does not cross the foreground counting line, so detecting them should not increase those counts.
- Updated runs took about 9.9 seconds (light compression) and 7.7 seconds (heavy compression) for 10 seconds of 5-FPS footage on this laptop, including setup and export; substantially slower than the old nano baseline. This leaves little margin for live operation and is not a multi-camera capacity result.

The same `people-light-compression.mp4` and `people-heavy-compression.mp4` filenames now contain the improved runs. Their original metrics and observation rows are retained under `previous_evaluations` in the existing shared `outputs/meva/report.json`; no old video copies or extra output files are retained. The report also contains the detector comparison and target checks. `experiments/validate_meva.py` now reproduces the improved person profile, while vehicle runs keep YOLO11n/640 and the original tracker settings. All four videos remain fully decodable; each updated person video has 50 frames. Larger detector / association changes have not been validated on held-out scenes or for general false positives.


## False stationary-person tracks reported by the user

The improved detector also proposed a pickup roof fixture and a roadside fixture as people. These were **false class predictions**, not evidence that stationary detections should be removed: real people can stand still.

Tracked confidence ranges in the pre-filter run: pickup roof up to 0.302 (light-compression clip); roadside fixture up to 0.494 (light) / 0.422 (heavy). The intended background person starts at 0.668 / 0.577. In the MEVA-only low-FPS profile, `new_track_thresh` was raised from 0.25 to **0.55**, leaving `track_high_thresh=0.25`, `track_low_thresh=0.1`, `match_thresh=0.9`, `fuse_score=false`, and `track_buffer=30` unchanged. This prevents those weak proposals from starting person tracks while retaining low-confidence recovery for already established tracks. No motion filter, hard-coded exclusion region, model training, or library modification was introduced. Inspection regions were used only to evaluate the two known false objects.

Both full clips were rerun. Confirmed observation rows at both manually identified fixture locations fell to zero. The real background person remains tracked in 45/50 frames with one ID across the gap, including the standing portion at the beginning. Foreground crossing counts remain 3 entries and 1 exit. Updated run times were about 11.6 seconds (light) and 8.7 seconds (heavy), including setup/export; this is not a stable real-time capacity measurement.

The two existing person videos were replaced after full 50-frame decode checks. Baseline and pre-filter metrics/observations remain in `previous_evaluations` in the same shared report. No additional outputs were created. Fourteen tests passed.

**Limit:** this higher starting threshold can delay or miss genuine people whose confidence never reaches 0.55. It suppresses confirmed tracks; it does not correct YOLO's underlying false proposals. These two negative objects and the standing-person portion are same-scene regression checks, not independent precision/recall validation.
