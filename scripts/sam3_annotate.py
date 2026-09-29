"""Generate review-only instance-mask proposals for one image using SAM 3."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", default="sausage")
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; use a new directory to preserve prior annotations.")
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this annotation starter.")
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.confidence)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        result = processor.set_text_prompt(state=state, prompt=args.prompt)
    args.out.mkdir(parents=True)
    image.save(args.out / "image.png")
    instances = []
    for index, (mask, score) in enumerate(zip(result["masks"], result["scores"])):
        array = mask.detach().cpu().numpy().squeeze()
        if array.shape != (image.height, image.width):
            raise ValueError(f"Unexpected mask shape: {array.shape}")
        filename = f"mask_{index:04d}.png"
        Image.fromarray((array > 0).astype(np.uint8) * 255).save(args.out / filename)
        instances.append({"id": index, "mask": filename, "score": float(score.item())})
    (args.out / "proposals.json").write_text(json.dumps({
        "source": str(args.image.resolve()), "prompt": args.prompt,
        "review_status": "unreviewed", "instances": instances,
        "note": "Review and correct every instance before adding to training data."
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(instances)} unreviewed instance masks to {args.out}")


if __name__ == "__main__":
    main()
