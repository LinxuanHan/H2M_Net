#!/usr/bin/env bash
set -euo pipefail

bash scripts/server_train.sh
bash scripts/server_infer.sh
