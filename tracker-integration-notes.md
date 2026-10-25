# Property traffic tracker — integration notes

Updated: 2026-10-24

## Baseline and sequence
- Prototype implemented in this folder: `tracker.py`, with `counting.py`, configurable ByteTrack/entrance-line settings, readably named annotated videos and a shared JSON report (per-frame observations CSV is opt-in). See `README.md` for usage and labeled-data resources. No detector fine-tuning has been performed.
- Start with a pretrained small YOLO person detector and ByteTrack on recorded footage.
- Tune and validate detection and tracking before live streaming or a dashboard.
- Revisit NVIDIA DeepStream for multi-camera processing and integration: https://github.com/nvidia/deepstream
- Planned camera-to-floor-plan calibration: **NVIDIA Metropolis Camera Calibration Toolkit**, distinct from detector/tracker tuning. References: [NVIDIA technical article](https://developer.nvidia.com/blog/simplifying-camera-calibration-to-enhance-ai-powered-multi-camera-tracking/) and [toolkit user guide](https://docs.nvidia.com/mms/text/MDX_Camera_Calibration_User_Guide.html). Integration is not implemented.
- Plan for 4–5 cameras per property. Camera specifications and placement are not yet available. Good image quality is expected; sub-4K resolution and readable license plates are assumptions to validate, not requirements already met.

## Counting and cross-camera integration
- Property analytics must prevent duplicate counts when cameras see the same person.
- ByteTrack IDs are local to a camera; a separate property-level association layer is needed for shared occupancy and journeys.
- Initially assign one primary camera to each entrance/counting boundary; do not sum overlapping camera counts.
- Preserve camera ID, synchronized timestamp, local track ID, confidence, and estimated position. Keep raw observations separate from entry/exit/zone/dwell events.
- For overlapping views, calibrate cameras to a shared floor plan and match by position, time, direction, and movement; associate matches with temporary property-wide visit IDs.
- Clothing color and visible accessories may support matching but are insufficient alone. Lighting and similar outfits cause errors.
- Non-overlapping views require travel-time and zone constraints; any appearance-assisted associations remain uncertain.
- Distinguish entries, occupancy, zone visits, and unique daily visitors. Returning visitors and broken tracks make unique visitor totals estimates until validated.

## Camera requirements to revisit
- Validate distance, lens, angle, pixel density, exposure, motion blur, lighting, timestamps, and stream access once camera specifications arrive.
- License plate recognition is optional and may require a dedicated camera/view and settings distinct from wide-area people tracking.
- Test synchronized footage from all cameras, including night scenes, crowds, returning people, and overlapping views; measure missed counts, duplicate counts, and incorrect merges separately.

## Privacy and legal scope
- Intended baseline: traffic analytics with temporary visit IDs, without recognizing people across days.
- Do not assume gait identification is non-biometric or Texas-compliant. Exclude gait identification from the baseline pending review of the exact implementation and applicable law.
- Do not treat clothing matching, temporary IDs, or license plates as a blanket exemption from privacy obligations; retention and use need defined policies before deployment.
- Texas statutes referenced in discussion: https://statutes.capitol.texas.gov/GetStatute.aspx?Code=BC&Value=503.001 and https://statutes.capitol.texas.gov/GetStatute.aspx?Code=BC&Value=541.001
- Camera guidance: https://whitepapers.axis.com/en-us/license-plate-capture
