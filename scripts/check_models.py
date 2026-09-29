"""Exercise model imports and YOLO CUDA inference; no training or accuracy claim."""
import json
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO


def main():
    assert torch.cuda.is_available(), "CUDA unavailable"
    weights = Path("weights")
    weights.mkdir(exist_ok=True)
    for name in ("yolov8n-seg.pt", "yolo11n-seg.pt"):
        model = YOLO(str(weights / name))
        result = model.predict(np.zeros((320, 320, 3), dtype=np.uint8), device=0, imgsz=320, verbose=False)
        assert model.task == "segment" and len(result) == 1
        print(json.dumps({"model": name, "cuda_inference": "passed", "note": "generic weights, not sausage-trained"}))
    from sam3.model_builder import build_sam3_image_model
    print("SAM 3 image builder import passed; weights/login checked separately")


if __name__ == "__main__":
    main()
