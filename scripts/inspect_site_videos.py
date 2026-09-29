"""Extract time-spaced site frames with provenance for qualitative review only."""
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

SOURCE = Path('/mnt/c/Users/n.ghehalot/Downloads')
NAMES = ['16123e62-41e7-4678-8a49-aa7d03f6d828', 'd8e8bd2c-2627-4766-a618-a26c80cd7d8a', '6f55ded6-db1f-40ec-99cc-019156ef247f']
output = Path('data/site_review')
output.mkdir(parents=True, exist_ok=True)
records = []
sheet = Image.new('RGB', (1200, 900), 'white')
draw = ImageDraw.Draw(sheet)
for row, name in enumerate(NAMES):
    cap = cv2.VideoCapture(str(SOURCE / (name + '.mp4')))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot decode {name}')
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for col, fraction in enumerate((0.1, 0.35, 0.6, 0.85)):
        index = min(count - 1, round(count * fraction))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot read {name}:{index}')
        filename = f'{name}_{index:06d}.jpg'
        cv2.imwrite(str(output / filename), frame)
        records.append({'file': filename, 'video': name + '.mp4', 'frame_index': index, 'seconds': index / fps, 'fps': fps, 'total_frames': count, 'split': 'unlabelled_review_only'})
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((300, 270))
        sheet.paste(image, (col * 300, row * 300 + 30))
        draw.text((col * 300 + 4, row * 300 + 5), f'Video {row+1}, {index/fps:.1f}s', fill='black')
    cap.release()
(output / 'manifest.json').write_text(json.dumps(records, indent=2))
sheet.save(output / 'contact_sheet.jpg')
print(json.dumps(records, indent=2))
