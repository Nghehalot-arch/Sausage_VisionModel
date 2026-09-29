#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_PATH="${VISION_ENV:-$HOME/.venvs/sausage-vision}"
python3 -m venv "$ENV_PATH"
PY="$ENV_PATH/bin/python"
"$PY" -m pip install --upgrade pip 'setuptools<81' wheel
"$PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
"$PY" -m pip install -e '.[train]' ipykernel
"$PY" -m sausage_vision doctor --require-gpu
