"""Validate trained baselines and save qualitative site-frame predictions."""
import json
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO


def main():
    root = Path.cwd()
    output = root / 'runs/baseline_comparison'
    output.mkdir(parents=True, exist_ok=False)
    records = json.loads((root / 'data/site_review/manifest.json').read_text())
    summary = {'validation_images': 4, 'validation_instances': 20,
               'site_review_frames': len(records),
               'limitation': 'Pilot validation only. Site frames are unlabelled; no site accuracy can be computed.', 'models': {}}
    for name in ('yolo11n', 'yolov8n'):
        checkpoint = root / f'runs/{name}_pilot_02/weights/best.pt'
        model = YOLO(str(checkpoint))
        metrics = model.val(data=str(root / 'data/prepared/pilot/data.yaml'), device=0,
                            imgsz=640, batch=4, workers=0, plots=True,
                            project=str(output), name=name + '_validation')
        scores = {k: float(v) for k, v in metrics.results_dict.items()}
        frame_results = []
        folder = output / name
        folder.mkdir()
        for record in records:
            frame = cv2.imread(str(root / 'data/site_review' / record['file']))
            result = model.predict(frame, device=0, imgsz=640, conf=0.25,
                                   max_det=300, retina_masks=True, verbose=False)[0]
            result.save(filename=str(folder / record['file']))
            frame_results.append({'file': record['file'], 'detections': len(result.boxes),
                                  'confidence': result.boxes.conf.cpu().tolist(), 'speed_ms': result.speed})
        target = root / f'weights/sausage_{name}_pilot.pt'
        shutil.copy2(checkpoint, target)
        summary['models'][name] = {'checkpoint': str(target.relative_to(root)),
                                   'validation': scores, 'site_frames': frame_results}
    winner = max(summary['models'], key=lambda name: summary['models'][name]['validation']['metrics/mAP50-95(M)'])
    summary['selected_baseline'] = winner
    summary['selection_basis'] = 'Highest mask mAP50-95 on four pilot validation images; not production readiness.'
    shutil.copy2(root / summary['models'][winner]['checkpoint'], root / 'weights/sausage_baseline_best.pt')
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    cards = []
    for record in records:
        cards.append(f'<h2>{record["video"]} - {record["seconds"]:.1f}s</h2><div class="row">' +
                     ''.join(f'<figure><figcaption>{name}</figcaption><img src="{name}/{record["file"]}"></figure>' for name in summary['models']) + '</div>')
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Site footage baseline review</title><style>body{font:16px system-ui;margin:24px}.row{display:flex}figure{width:48%;margin:1%}img{max-width:100%;max-height:700px}h2{font-size:16px}</style><h1>Site footage: baseline predictions</h1><p>Unlabelled frames, confidence threshold 0.25. Compare masks visually; counts are predictions, not ground truth. Models were trained on plate photos.</p>' + ''.join(cards), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
