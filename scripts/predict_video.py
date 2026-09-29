"""Write a sausage mask-overlay video and per-frame detection metadata."""
import argparse
import json
from pathlib import Path
import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path('weights/sausage_baseline_best.pt'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output exists. Select a new directory.')
    model = YOLO(str(args.weights))
    if model.task != 'segment' or list(model.names.values()) != ['sausage']:
        parser.error('Use a one-class sausage segmentation checkpoint.')
    cap = cv2.VideoCapture(str(args.source))
    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not ok or fps <= 0:
        parser.error('Cannot decode source video or frame rate.')
    args.out.mkdir(parents=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(args.out / 'predictions.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Cannot open MP4 writer')
    index = 0
    try:
        with (args.out / 'frames.jsonl').open('w') as log:
            while ok:
                result = model.predict(frame, device=0, imgsz=640, conf=args.conf, max_det=300, verbose=False)[0]
                writer.write(result.plot())
                log.write(json.dumps({'frame': index, 'seconds': index/fps, 'count': len(result.boxes),
                                      'boxes_xyxy': result.boxes.xyxy.cpu().tolist(),
                                      'confidence': result.boxes.conf.cpu().tolist()}) + '\n')
                index += 1
                ok, frame = cap.read()
    finally:
        cap.release()
        writer.release()
    (args.out / 'summary.json').write_text(json.dumps({'source': str(args.source), 'weights': str(args.weights),
        'frames': index, 'fps': fps, 'confidence_threshold': args.conf,
        'note': 'Qualitative prediction only. No ground-truth labels, tracking IDs, 3D geometry or graspability.'}, indent=2))
    print(f'Wrote {index} frames to {args.out}')


if __name__ == '__main__':
    main()
