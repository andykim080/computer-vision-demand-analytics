# Screenshot provenance

Each before/after pair uses the same zero-based frame index, decoded sequentially from existing local videos and saved as JPEG at quality 93 without resizing or added annotations. The after images preserve the tracker overlays already present in the output video.

| Image prefix | Source video | Processed video | Frame | Time |
| --- | --- | --- | --- | --- |
| `retail-entrance` | `data/retail-entrance/holdout/TwoEnterShop3front.mpg` | `outputs/metal-accuracy/full-20261025T184647Z-hjppfg_5/mps/holdout/TwoEnterShop3front/twoentershop3front.mp4` | 450 | 18.00 s |
| `retail-crossings` | `data/retail-entrance/holdout/OneShopOneWait2front.mpg` | `outputs/metal-accuracy/full-20261025T184647Z-hjppfg_5/mps/holdout/OneShopOneWait2front/oneshoponewait2front.mp4` | 700 | 28.00 s |

Both clips are CAVIAR shopping-centre recordings at 25 FPS; output uses the existing YOLO11s/640 Apple Metal replay. Source and output videos are local artifacts excluded by the project ignore rules. These small screenshots live under `docs/` so they can be included in the repository and rendered on GitHub. Consult the source dataset terms before redistribution. Screenshots illustrate detection and tracking, not an accuracy improvement or a demand forecast.
