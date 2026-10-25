# Screenshot provenance

Each before/after pair uses the same zero-based frame index, decoded sequentially from existing local videos and saved as JPEG at quality 93 without resizing or added annotations. The after images preserve the tracker overlays already present in the output video.

| Image prefix | Source video | Processed video | Frame | Time |
| --- | --- | --- | --- | --- |
| `retail-entrance` | `data/retail-entrance/holdout/TwoEnterShop3front.mpg` | `outputs/metal-accuracy/full-20261025T184647Z-hjppfg_5/mps/holdout/TwoEnterShop3front/twoentershop3front.mp4` | 450 | 18.00 s |
| `retail-crossings` | `data/retail-entrance/holdout/OneShopOneWait2front.mpg` | `outputs/metal-accuracy/full-20261025T184647Z-hjppfg_5/mps/holdout/OneShopOneWait2front/oneshoponewait2front.mp4` | 700 | 28.00 s |

Both clips are CAVIAR shopping-centre recordings at 25 FPS; output uses the existing YOLO11s/640 Apple Metal replay. Source and output videos are local artifacts excluded by the project ignore rules. These small screenshots live under `docs/` so they can be included in the repository and rendered on GitHub. Consult the source dataset terms before redistribution. Screenshots illustrate detection and tracking, not an accuracy improvement or a demand forecast.

## Location test replays

The six location screenshots are new replays of saved test clips, generated for the README. Historical annotated outputs had been removed. Each pair uses zero-based frame 40 (8.00 seconds at 5 FPS), decoded sequentially with no resizing; original resolution is 1280×720. JPEG quality is 93. Each run processed the first 61 frames with pretrained YOLO11s, image size 1280, confidence 0.1, IoU 0.7, CPU inference, and `configs/bytetrack-low-fps.yaml`. Classes include person, car, truck, bus, motorcycle, and bicycle. Site lines come from the saved five-site plan or retail-scene plan; these are illustrative boundaries.

| Image prefix | Saved source clip | Publisher attribution |
| --- | --- | --- |
| `new-york-times-square` | `data/test-five-sites/times-square/05-plus-08m.mp4` | [EarthCam: Times Square, New York](https://www.earthcam.com/cams/newyork/timessquare/?cam=tsrobo1) |
| `seoul-songridan-gil` | `data/retail-security-test/songridan-gil.mp4` | [EarthCam: Songridan-gil, Seoul](https://www.earthcam.com/world/southkorea/seoul/songridangil/) |
| `taiwan-linkou` | `data/retail-security-test/linkou-old-street.mp4` | [EarthCam: Linkou, New Taipei City](https://www.earthcam.com/world/taiwan/newtaipeicity/linkoudistrict/?cam=linkoudistrict) |

Attribution is copied from the saved acquisition manifests. Acquisition time is not verified scene time. These are illustrative model outputs, including mistakes; they do not establish detection or crossing accuracy. The original retail walkthrough is now shown as a limitation because of the user-identified mannequin false positive. No validated mitigation is claimed.

[Replay provenance](location-replays.json) records source hashes and actual run settings. Source videos, weights, and generated MP4s remain local.
