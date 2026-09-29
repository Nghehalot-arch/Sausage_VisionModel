# Baseline results - 2026-09-29

Two sausage instance-segmentation baselines completed 100 training epochs on the RTX 3080. Training used 15 images / 55 instances, with 4 images / 20 instances for validation. These photos show sausage pieces on a plate; the split is a pilot layout split, not an independent production test.

| Model | Mask mAP@0.5 | Mask mAP@0.5:0.95 |
| --- | ---: | ---: |
| YOLOv8n-seg | 0.9321 | 0.5697 |
| YOLO11n-seg | 0.9301 | 0.4893 |

YOLOv8 is selected only as the stronger **pilot validation baseline**. These numbers are not percentages of successful picks and do not demonstrate production readiness.

## Production footage result: not ready

At confidence 0.25 and input size 640, YOLOv8 produced zero detections on all 12 sampled frames across the three supplied site videos. YOLO11 produced 12 detections in total across those frames, but those counts are not correct-instance counts. Visual review of a bin frame showed a false positive on equipment and missed the visible pile. The model has not learned reliable production-bin segmentation.

There are no ground-truth site labels yet, so site precision, recall and mAP cannot be computed. The entire clips are also processed into local diagnostic videos, not used to train the models. Camera viewpoint, object size, pile density, background and occlusion differ substantially from the training photos.

## Local deliverables

- `weights/sausage_yolov8n_pilot.pt` and `weights/sausage_yolo11n_pilot.pt`: trained models.
- `weights/sausage_baseline_best.pt`: copy of the selected YOLOv8 pilot.
- `runs/yolov8n_pilot_02/` and `runs/yolo11n_pilot_02/`: training options, environment snapshots, logs, validation plots and original best/last checkpoints.
- `runs/baseline_comparison/comparison.json`: fresh checkpoint validation and frame-level predictions.
- `runs/baseline_comparison/review.html`: side-by-side site review.
- `runs/site_video_bin/`, `runs/site_video_transfer/`, `runs/site_video_packaging/`: full-clip overlays and frame-level JSONL records.
- `data/site_review/manifest.json`: timestamp provenance for the 12 review frames.

Weights, footage and generated outputs are local and excluded from Git. The repository contains code, configuration and this result summary.

## Training adjustments

The first `_pilot_01` runs stopped early with poor learning. The completed `_pilot_02` runs use explicit AdamW at 0.001, batch and nominal batch 4, full precision, no warm-up, no early stopping, seed 42 and the same augmentation settings for both models. Multiple settings changed together, so the improvement cannot be attributed to a single cause. CuBLAS deterministic workspace configuration was set for the second runs; this is not a cross-platform reproducibility guarantee.

## Next annotation session

1. Correct the missing IMG_1434 instance and review disconnected-instance masks.
2. Prioritize actual bin images, including touching sausages, occlusion, gloves, reflective equipment and empty/background regions.
3. Use SAM 3 proposals after access approval, then manually correct every instance. Manual polygons remain an option while access is pending.
4. Group by capture session and pile arrangement before splitting. Nearby video frames must stay together. Capture new independent scenes for final evaluation.
5. Retrain both architectures on reviewed site annotations and measure held-out site masks. Only then evaluate calibrated depth and grasp selection.

Today's completed milestone is **trainable, tested baseline models and a demonstrated production-data gap**, not a finished robot-ready vision model.
