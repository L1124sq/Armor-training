#!/usr/bin/env bash
# Fetch the pinned Ultralytics YOLOv5 source and overlay this repo's armor-specific files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

UPSTREAM_COMMIT="${UPSTREAM_COMMIT:-9777cd0}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/ultralytics/yolov5.git}"

if [ ! -d upstream/.git ]; then
  git clone "$UPSTREAM_URL" upstream
fi

git -C upstream fetch --all --tags
git -C upstream checkout -f "$UPSTREAM_COMMIT"
cp -r overrides/. upstream/

echo "Ready: cd upstream"
echo "Next: pip install -r ../requirements.txt"
echo "      download yolov5s.pt, prepare datasets/data.yaml, then run train_kpt.py"
