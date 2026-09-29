#!/usr/bin/env bash
set -euo pipefail
export GIT_CONFIG_GLOBAL=/dev/null
PY="${VISION_ENV:-$HOME/.venvs/sausage-vision}/bin/python"
SOURCE_ROOT="$HOME/.local/share/sausage-vision"
mkdir -p "$SOURCE_ROOT"
for repo in sam3 detectron2; do
  if [ ! -d "$SOURCE_ROOT/$repo/.git" ]; then
    git clone "https://github.com/facebookresearch/$repo.git" "$SOURCE_ROOT/$repo"
  fi
done
git -C "$SOURCE_ROOT/sam3" checkout 2345a4ad109ac29c569da749c91d84f10dc08c40
git -C "$SOURCE_ROOT/detectron2" checkout fc3b7a1e658db27cdb52ee94ef5dcec7cc7eb1e7
"$PY" -m pip install -e "$SOURCE_ROOT/sam3" einops pycocotools
"$PY" -m pip check
DETECTRON_ENV="$HOME/.venvs/sausage-detectron2"
python3 -m venv "$DETECTRON_ENV"
D2PY="$DETECTRON_ENV/bin/python"
"$D2PY" -m pip install --upgrade pip 'setuptools<81' wheel
"$D2PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
MAX_JOBS=2 "$D2PY" -m pip install --no-build-isolation -e "$SOURCE_ROOT/detectron2"
"$D2PY" -m pip check
