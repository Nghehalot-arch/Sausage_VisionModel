"""YOLO operations are imported only when training, evaluating or predicting."""
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import yaml


def doctor(require_gpu=False):
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": {}}
    for package in ("sausage-vision", "ultralytics", "torch", "torchvision", "numpy", "Pillow"):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = "not installed"
    info["cuda_available"] = False
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_runtime"] = torch.version.cuda
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            # Exercise the GPU; enumeration alone does not prove CUDA kernels work.
            sample = torch.ones((8, 8), device="cuda")
            info["cuda_operation_ok"] = float((sample @ sample).sum().item()) == 512.0
    except Exception as exc:
        info["gpu_check_error"] = str(exc)
    print(json.dumps(info, indent=2))
    if require_gpu and (not info["cuda_available"] or not info.get("cuda_operation_ok")):
        raise ValueError("GPU check failed. Confirm nvidia-smi in WSL and the CUDA-enabled PyTorch installation.")
    return info


def require_device(device):
    import torch
    if device != "cpu" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Run sausage-vision doctor --require-gpu first.")


def training(args):
    from ultralytics import YOLO
    require_device(args.device)
    data = Path(args.data).resolve()
    if not data.is_file():
        raise ValueError("Prepared data.yaml not found; run the prepare command first.")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config.update({key: getattr(args, key) for key in ("epochs", "batch", "imgsz", "workers", "model") if getattr(args, key) is not None})
    run = Path(args.out).resolve()
    if run.exists():
        raise ValueError("Training output already exists. Choose a new --out for the new experiment.")
    checkpoint = config.pop("model")
    run.mkdir(parents=True)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit, git_dirty = None, None
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "model": checkpoint,
                "data": str(data), "options": config, "device": args.device,
                "git_commit": git_commit, "git_dirty": git_dirty, "environment": doctor()}
    (run / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (run / "environment.txt").write_text(freeze, encoding="utf-8")
    # Core checkpoint is pretrained; it is downloaded on first use.
    model = YOLO(checkpoint)
    model.train(data=str(data), device=args.device, project=str(run.parent), name=run.name,
                exist_ok=True, plots=True, **config)
    checkpoint_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print("Training finished. Checkpoint:", checkpoint_path)
    print("Validation plots and logs:", model.trainer.save_dir)


def predict(args):
    import numpy as np
    from PIL import Image, ImageOps
    from ultralytics import YOLO
    require_device(args.device)
    weights, source, output = Path(args.weights), Path(args.source), Path(args.out).resolve()
    if not weights.is_file():
        raise ValueError("Trained weights not found. Run training first.")
    if output.exists():
        raise ValueError("Prediction output already exists. Choose a new --out directory.")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    if not files:
        raise ValueError("No images found at --source.")
    model = YOLO(str(weights))
    if model.task != "segment" or list(model.names.values()) != ["sausage"]:
        raise ValueError("Use the trained, one-class sausage segmentation checkpoint.")
    output.mkdir(parents=True)
    for frame_index, file in enumerate(files):
        with Image.open(file) as raw:
            original_orientation = raw.getexif().get(274, 1)
            image = ImageOps.exif_transpose(raw).convert("RGB")
        result = model.predict(source=image, device=args.device, imgsz=args.imgsz,
                               conf=args.conf, max_det=args.max_det, retina_masks=True, verbose=False)[0]
        folder = output / f"{frame_index:04d}_{file.stem}"
        folder.mkdir()
        result.save(filename=str(folder / "overlay.jpg"))
        instances = []
        if result.masks is not None:
            for index, mask in enumerate(result.masks.data.cpu().numpy()):
                if mask.shape != (image.height, image.width):
                    raise ValueError("Mask size differs from the oriented source image; refusing a misaligned export.")
                filename = f"mask_{index:03d}.png"
                Image.fromarray((mask > 0.5).astype(np.uint8) * 255).save(folder / filename)
                instances.append({"instance_index": index, "class": "sausage", "mask": filename,
                                  "confidence": float(result.boxes.conf[index].item()),
                                  "box_xyxy": result.boxes.xyxy[index].cpu().tolist()})
        record = {"source": str(file.resolve()), "width": image.width, "height": image.height,
                  "source_exif_orientation": original_orientation,
                  "coordinate_frame": "EXIF-oriented colour image pixels",
                  "instances": instances,
                  "note": "Per-image instance indices, not tracking IDs. Detection confidence is not graspability. No 3D pose."}
        (folder / "instances.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"{file.name}: {len(instances)} detected instances -> {folder}")


def validate(args):
    from ultralytics import YOLO
    require_device(args.device)
    if not Path(args.weights).is_file():
        raise ValueError("Trained checkpoint not found.")
    data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    if args.split not in data:
        raise ValueError(f"No {args.split} split. The pilot contains train/val only; collect independent test scenes later.")
    output = Path(args.out).resolve()
    if output.exists():
        raise ValueError("Validation output already exists. Choose a new --out directory.")
    model = YOLO(args.weights)
    metrics = model.val(data=str(Path(args.data).resolve()), split=args.split,
                        device=args.device, imgsz=args.imgsz, plots=True,
                        project=str(output.parent), name=output.name)
    print("Mask mAP50-95:", metrics.seg.map)
    print("Mask mAP50:", metrics.seg.map50)
    print("Pilot validation scores are not a production performance estimate.")
