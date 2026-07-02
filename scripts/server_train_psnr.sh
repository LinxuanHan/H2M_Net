#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export KMP_DUPLICATE_LIB_OK=TRUE

torchrun --standalone --nproc_per_node=8 -m src.train \
  --config configs/train_x2_server.yaml \
  --resume experiments/steered_diffusion/x2/checkpoints/last_x2.pt \
  --epochs 700 \
  --no_preview

torchrun --standalone --nproc_per_node=8 -m src.train \
  --config configs/train_x4_server.yaml \
  --resume experiments/steered_diffusion/x4/checkpoints/last_x4.pt \
  --epochs 700 \
  --no_preview
