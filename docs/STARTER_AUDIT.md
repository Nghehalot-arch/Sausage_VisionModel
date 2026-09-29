# Starter audit

- `sausage_lab_handoff_2.0.zip`: original images and COCO annotations. Retain as authoritative source; it is not application code.
- `sausage-vision.zip`: older scripts plus prepared data. Contains literal brace-expanded directory names and includes the known incomplete IMG_1434 label in validation. Not selected as the base.
- `sausage_vision_pipeline_v1.zip`: selected code base. Includes explicit split/exclusion configuration, EXIF correction, COCO source retention, multipart conversion audits, CLI and tests.

The selected code is a pilot, not a production system. Its split groups are inferred layouts, not verified independent recording sessions. Multipart polygons use retraced bridges and require visual review; Pillow raster agreement alone does not establish Ultralytics training-mask agreement. SAM mask/RLE conversion is not implemented by this polygon-only converter. The 20 photos cannot establish production-bin accuracy.

Setup changes add YOLOv8 checkpoint support, WSL/VS Code configuration and reproducible setup instructions. Keep private imagery, model weights and generated runs out of Git.
