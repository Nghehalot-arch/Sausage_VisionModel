# Sausage Vision Model

Instance segmentation for sausage bin picking. Compare YOLOv8-seg and YOLO11-seg; use SAM 3 to generate annotation proposals for manual correction. Detectron2 is installed separately for research. No sausage-trained model or production accuracy is claimed yet.

## Open and start coding

Open this folder in **VS Code: WSL / Ubuntu-24.04**. The selected interpreter is `/home/nghehalot/.venvs/sausage-vision/bin/python`.

In the WSL terminal:

```bash
source ~/.venvs/sausage-vision/bin/activate
export GIT_CONFIG_GLOBAL=/dev/null
export YOLO_CONFIG_DIR="$HOME/.local/share/sausage-vision/ultralytics"
python -m sausage_vision doctor --require-gpu
python -m unittest discover -s tests -v
```

The Git environment override bypasses this machine's pre-existing `.gitconfig` directory. It does not replace local repository settings. VS Code terminal settings include it.

## Installed setup

Ubuntu 24.04 / Python 3.12.3 / RTX 3080 10 GB. Main environment: PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126, Ultralytics 8.4.157 and SAM 3. NumPy and image-processing packages are pinned for compatibility.

Detectron2 uses `~/.venvs/sausage-detectron2` because its iopath requirement conflicts with SAM 3. Its compiled extension is CPU-only; CUDA tensor operations and standard ROIAlign forward/backward passed. Custom CUDA operators such as rotated/deformable operations need a compatible CUDA toolkit and a rebuild. Full Detectron2 model training has not been tested.

Recreate the setup after installing Ubuntu packages `python3-venv python3-dev build-essential ninja-build libgl1 libglib2.0-0`:

```bash
bash scripts/setup_wsl.sh
bash scripts/setup_research_models.sh
python scripts/check_models.py
~/.venvs/sausage-detectron2/bin/python scripts/check_detectron.py
```

The scripts pin upstream revisions. Installed package snapshots are in `docs/environment-*.txt`; these are audit records, not portable lockfiles.

## Dataset and annotation

The code is based on `sausage_vision_pipeline_v1.zip`; see [starter audit](docs/STARTER_AUDIT.md). Original data comes from `sausage_lab_handoff_2.0.zip`. Private data and weights stay outside Git.

The pilot is already prepared locally in `data/prepared/pilot`: 15 training images / 55 instances, 4 validation images / 20 instances. IMG_1434 is excluded pending its missing label. Inspect `data/prepared/pilot/review/index.html` before training. These layout-based groups are only a debugging split, not an independent production evaluation.

For a new checkout:

```bash
python -m sausage_vision prepare --input /path/to/sausage_lab_handoff_2.0.zip --out data/prepared/pilot
```

SAM 3 access was approved on the user's Hugging Face account; local login and model-weight inference remain pending:

```bash
hf auth login
python scripts/sam3_annotate.py data/prepared/pilot/images/train/IMG_1429.jpg --out data/annotations/first-sam3
```

Enter the token privately in the terminal. This command downloads the gated weights on first use and writes one PNG mask per proposed instance plus `proposals.json`. All proposals are explicitly unreviewed. The command's imports and syntax were verified; end-to-end inference requires login and has not yet been verified. The 10 GB GPU's peak SAM 3 memory use remains to be measured. Optional video decoding dependencies are omitted; start with still frames. The existing COCO converter accepts polygons, not SAM mask/RLE outputs; reviewed-mask export is a later pipeline step.

## Train the two baselines

After reviewing labels, run separate matched experiments:

```bash
python -m sausage_vision train --model weights/yolov8n-seg.pt --out runs/yolov8n_pilot --batch 4
python -m sausage_vision train --model weights/yolo11n-seg.pt --out runs/yolo11n_pilot --batch 4
```

Both pretrained checkpoints passed CUDA inference on this machine; they are generic models, not sausage-trained weights. F5 configurations are also supplied. Capture-session separation and held-out production footage are required before comparing accuracy for the site.

## Repository and verification

Remote: `Nghehalot-arch/Sausage_VisionModel`. Setup branch: `codex/vision-setup`, based on the original main commit. Existing MIT LICENSE is preserved; dependency and model licenses remain separate.

Verified: three dataset unit tests, actual COCO conversion, YOLOv8/YOLO11 CUDA inference, SAM 3 image-builder import, Detectron2 CUDA ROIAlign forward/backward, and dependency consistency in both environments. No full model training, robot integration or production benchmark has run.

Official installation references: [Ultralytics](https://docs.ultralytics.com/quickstart/), [SAM 3](https://github.com/facebookresearch/sam3), [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
