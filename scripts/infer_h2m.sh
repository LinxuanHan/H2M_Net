#!/usr/bin/env bash
set -euo pipefail
python -m src.infer_h2m --config configs/h2m_infer_mouse.yaml --scale 2 "$@"
python -m src.infer_h2m --config configs/h2m_infer_mouse.yaml --scale 4 "$@"
