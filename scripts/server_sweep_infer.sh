#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export KMP_DUPLICATE_LIB_OK=TRUE

SCALE=${SCALE:-2}
SAMPLING_STEPS=${SAMPLING_STEPS:-50}
LIMIT=${LIMIT:-0}

run_one () {
  local gpu=$1
  local guidance=$2
  local noise=$3
  local anchor=$4
  local residual=$5
  local blend=$6
  local limit_arg=()
  if [ "$LIMIT" != "0" ]; then
    limit_arg=(--limit "$LIMIT")
  fi
  CUDA_VISIBLE_DEVICES=$gpu python -m src.inference \
    --config configs/infer_mouse.yaml \
    --scale "$SCALE" \
    --guidance_scale "$guidance" \
    --omega "$guidance" \
    --sampling_steps "$SAMPLING_STEPS" \
    --noise_strength "$noise" \
    --anchor_weight "$anchor" \
    --residual_scale "$residual" \
    --bicubic_blend "$blend" \
    --data_consistency_weight 1.0 \
    "${limit_arg[@]}"
}

jobs=0
gpu=0
for guidance in 0.8 1.0 1.2; do
  for noise in 0.10 0.20 0.30; do
    for anchor in 0.25 0.40 0.55; do
      for residual in 0.30 0.45 0.60; do
        for blend in 0.0 0.10 0.20 0.35; do
          run_one "$gpu" "$guidance" "$noise" "$anchor" "$residual" "$blend" &
        jobs=$((jobs + 1))
        gpu=$(((gpu + 1) % 8))
        if [ "$jobs" -ge 8 ]; then
          wait
          jobs=0
        fi
        done
      done
    done
  done
done
wait

python - <<'PY'
import csv
from pathlib import Path

metric_dir = Path("experiments/steered_diffusion/metrics")
rows = []
for path in metric_dir.glob("steered_diffusion_x*_summary.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f), None)
    if row and "psnr_mean" in row:
        row["file"] = str(path)
        rows.append(row)
rows.sort(key=lambda r: float(r["psnr_mean"]), reverse=True)
for row in rows[:20]:
    print(row["psnr_mean"], row.get("ssim_mean", ""), row["file"])
PY
