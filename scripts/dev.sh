#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

export PYTHONPATH=src
mkdir -p recordings

echo "Launching demo recorder. Press 'r' to start/stop, 'q' to quit."
python -m world2data_cam record --camera 0 --out recordings/demo.mp4 --fps 30 --width 1280 --height 720
