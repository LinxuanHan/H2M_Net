#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export KMP_DUPLICATE_LIB_OK=TRUE

GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.2}
OMEGA=${OMEGA:-3.0}
SAMPLING_STEPS=${SAMPLING_STEPS:-50}
NUM_SHARDS=${NUM_SHARDS:-8}
NOISE_STRENGTH=${NOISE_STRENGTH:-0.25}
ANCHOR_WEIGHT=${ANCHOR_WEIGHT:-0.35}
RESIDUAL_SCALE=${RESIDUAL_SCALE:-0.55}
BICUBIC_BLEND=${BICUBIC_BLEND:-0.15}
SELF_ENSEMBLE=${SELF_ENSEMBLE:-0}

fmt_float () {
  python -c 'import sys; print(f"{float(sys.argv[1]):g}")' "$1"
}

run_scale () {
  local scale=$1
  local omega_tag noise_tag anchor_tag residual_tag blend_tag
  omega_tag=$(fmt_float "$OMEGA")
  noise_tag=$(fmt_float "$NOISE_STRENGTH")
  anchor_tag=$(fmt_float "$ANCHOR_WEIGHT")
  residual_tag=$(fmt_float "$RESIDUAL_SCALE")
  blend_tag=$(fmt_float "$BICUBIC_BLEND")
  local tag="x${scale}_om${omega_tag}_ns${noise_tag}_aw${anchor_tag}_rs${residual_tag}_bb${blend_tag}"
  rm -f experiments/steered_diffusion/metrics/steered_diffusion_${tag}_shard*of*_per_image.csv
  rm -f experiments/steered_diffusion/metrics/steered_diffusion_${tag}_shard*of*_summary.csv
  rm -f experiments/steered_diffusion/metrics/steered_diffusion_${tag}_shard*of*_vdp_vhi.csv
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    if [ "$SELF_ENSEMBLE" = "1" ]; then
      CUDA_VISIBLE_DEVICES=$shard python -m src.inference \
        --config configs/infer_mouse.yaml \
        --scale "$scale" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --omega "$OMEGA" \
        --sampling_steps "$SAMPLING_STEPS" \
        --noise_strength "$NOISE_STRENGTH" \
        --anchor_weight "$ANCHOR_WEIGHT" \
        --residual_scale "$RESIDUAL_SCALE" \
        --bicubic_blend "$BICUBIC_BLEND" \
        --data_consistency_weight 1.0 \
        --num_shards "$NUM_SHARDS" \
        --shard_id "$shard" \
        --self_ensemble &
    else
      CUDA_VISIBLE_DEVICES=$shard python -m src.inference \
        --config configs/infer_mouse.yaml \
        --scale "$scale" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --omega "$OMEGA" \
        --sampling_steps "$SAMPLING_STEPS" \
        --noise_strength "$NOISE_STRENGTH" \
        --anchor_weight "$ANCHOR_WEIGHT" \
        --residual_scale "$RESIDUAL_SCALE" \
        --bicubic_blend "$BICUBIC_BLEND" \
        --data_consistency_weight 1.0 \
        --num_shards "$NUM_SHARDS" \
        --shard_id "$shard" &
    fi
  done
  wait
  python -m src.merge_metrics --scale "$scale" --guidance_scale "$GUIDANCE_SCALE" --tag "$tag"
}

run_scale 2
run_scale 4
