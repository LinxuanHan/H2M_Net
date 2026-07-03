#!/usr/bin/env bash
set -euo pipefail
python -m src.train_h2m --config configs/h2m_train_x2.yaml "$@"
