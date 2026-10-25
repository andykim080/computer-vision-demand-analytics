# Confidence study: detailed development history

This archived narrative supplements the [published review and saved decision](README.md). Local `outputs/` paths refer to the repository root. The adopted version is the initial class-specific profile; historical candidate selections remain preserved.

## Class-specific confidence development test

The optional `class_confidence` camera setting applies detector cutoffs by class before ByteTrack. Keep the base `--conf` at or below the smallest class cutoff so lower-confidence people/motorcycles remain available. Existing profiles without this field retain their previous behavior; global defaults and the installed YOLO/ByteTrack libraries are unchanged.

On the same Seoul and Taiwan clips, cars/trucks at 0.55 with people/motorcycles at 0.10 restores the two checked scooters, but a lamp can acquire the unchanged low-confidence bus label. A development search of 50 grid combinations, plus baseline/global/requested controls, ranks by **recall, then F1, then precision**. Exact ties use fewer reported fixture/cargo false-box frames and then lower cutoffs. The best tested combination is **car 0.50, truck 0.40, bus 0.55, person 0.10, motorcycle 0.10**, with bicycle unchanged at 0.10.

| Cutoffs | Foreground box recall | F1 | Precision |
| --- | --- | --- | --- |
| Original 0.10 for all classes | 67.6% | 71.9% | 76.7% |
| Global 0.55 | 50.0% | 65.4% | **94.4%** |
| Requested car/truck 0.55; others 0.10 | 67.6% | 76.7% | 88.5% |
| Selected class cutoffs | **70.6%** | **78.7%** | 88.9% |

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
| Tracked-box recall | 75.21% | **75.28%** |
| Tracked-box F1 | **79.97%** | 79.71% |
| Tracked-box precision | **85.38%** | 84.68% |
| Matched boxes | 3,143 | **3,146** |
| Unmatched predictions | **538** | 569 |

The extra three matches cost 31 extra unmatched predictions. This does **not** support a clear general improvement; application defaults remain unchanged. The corrected configuration is saved for investigation in `configs/larger-confidence-recovered-profile.json`. This is an experiment profile consumed by the helper, not a camera-line file for `--config`. The earlier rejected profile is retained separately in `configs/larger-confidence-candidate.json` as evidence, not a recommended setting.

Scores use official pinned TrackEval and published ignore rules. Person/vehicle boxes are grouped; UA-DETRAC vehicle types are not scored as car-versus-truck classification. These numbers are not the earlier doorway-crossing metrics or directly comparable to the four-frame foreground sample. Later windows share cameras and actors with earlier footage, and tracking resets at each segment; this is not continuous live or hourly validation. Motorcycle behavior is supported only by the old small spot checks. No weights, ground-truth labels, installed libraries or prediction-location exclusions were changed.

`experiments/iterate_larger_sample.py` records the first three rounds and blocks development after its deferred check begins. `experiments/recover_larger_sample.py` preserves that failure, runs the corrective round, freezes it before later-window inference and creates the latest review. Native detector results are cached only to replay identical inputs across candidate settings. Each output group retains comparison videos with one shared report and review page; source recordings and historical outputs remain in place.

Published evidence for the confidence studies: [review, rollback rationale, four output comparisons and original reports](README.md). Historical `selected_profile` fields are preserved; `user_decision` records the adopted initial version.
