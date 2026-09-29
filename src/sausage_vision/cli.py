import argparse


def parser():
    root = argparse.ArgumentParser(description="Sausage instance segmentation pilot")
    sub = root.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check Python, packages and GPU access")
    doctor.add_argument("--require-gpu", action="store_true")
    prepare = sub.add_parser("prepare", help="Convert the COCO handoff ZIP and create label previews")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--config", default="configs/pilot.yaml")
    prepare.add_argument("--out", default="data/prepared/pilot")
    train = sub.add_parser("train", help="Fine-tune the pretrained segmentation model")
    train.add_argument("--data", default="data/prepared/pilot/data.yaml")
    train.add_argument("--config", default="configs/train.yaml")
    train.add_argument("--out", default="runs/baseline")
    train.add_argument("--model", help="Segmentation checkpoint, e.g. yolov8n-seg.pt or yolo11n-seg.pt")
    for key in ("epochs", "batch", "imgsz", "workers"):
        train.add_argument("--" + key, type=int)
    train.add_argument("--device", default="0")
    predict = sub.add_parser("predict", help="Save overlays, individual masks and metadata for photos")
    predict.add_argument("--weights", default="runs/baseline/weights/best.pt")
    predict.add_argument("--source", required=True)
    predict.add_argument("--out", default="runs/predictions")
    predict.add_argument("--device", default="0")
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--max-det", type=int, default=300)
    validate = sub.add_parser("validate", help="Measure segmentation metrics on a labelled split")
    validate.add_argument("--weights", default="runs/baseline/weights/best.pt")
    validate.add_argument("--data", default="data/prepared/pilot/data.yaml")
    validate.add_argument("--out", default="runs/validation")
    validate.add_argument("--split", choices=["val", "test"], default="val")
    validate.add_argument("--device", default="0")
    validate.add_argument("--imgsz", type=int, default=640)
    return root


def main():
    args = parser().parse_args()
    try:
        if args.command == "prepare":
            from .prepare import prepare
            import json
            report = prepare(args.input, args.config, args.out)
            print(json.dumps({"counts": report["counts"], "excluded": report["excluded"],
                              "multipart_instances": report["multipart_instances"]}, indent=2))
            print("Review labels:", args.out + "/review/index.html")
        else:
            from . import runtime
            if args.command == "doctor":
                runtime.doctor(args.require_gpu)
            else:
                {"train": runtime.training, "predict": runtime.predict, "validate": runtime.validate}[args.command](args)
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}") from None
