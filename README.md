# H2M-Net

Target-free zero-shot cross-species super-resolution for mouse Xenon-129 lung MRI via self-supervised elastic deformation, orthogonal residual steering, pathology-aware soft gating, and low-resolution anchoring.

## Core Method

H2M-Net has two stages:

1. **Target-free self-supervised elastic manifold training**
   - Uses only human HR images during training.
   - Applies smooth elastic deformation to human HR first.
   - Then applies degradation to create physically consistent LR-up conditions.
   - Trains three condition states:
     - `c=0`: Anatomy Follower
     - `c=1`: Anatomy Corrector
     - `c=2`: Human HR Prior

2. **Zero-shot mouse inference**
   - Uses unseen mouse LR image only at inference.
   - Computes three conditional noise predictions in one batched forward pass.
   - Applies ORS, PSG, and LRA to obtain faithful SR output.

## Core Files

- `src/h2m_net.py`: ElasticDeformer, H2MSteeredDiffusion, ORS, PSG, LRA.
- `src/model.py`: Conditional U-Net backbone with three condition states.
- `src/train_h2m.py`: H2M-Net training entrypoint.
- `src/infer_h2m.py`: H2M-Net inference entrypoint.
- `configs/h2m_train_x2.yaml`: x2 training config.
- `configs/h2m_train_x4.yaml`: x4 training config.
- `configs/h2m_infer_mouse.yaml`: mouse inference config.

## Expected Data Layout

This GitHub package does **not** include datasets or trained weights. Put files in the following paths:

```text
data/
  human/
    train/
      raw/                 # human HR 2D slices for training
  mouse/
    test/
      down_2x_bilinear/    # mouse LR x2 slices
      down_4x_bilinear/    # mouse LR x4 slices
      raw/                 # mouse HR reference, only for evaluation
```

## Training

```bash
python -m src.train_h2m --config configs/h2m_train_x2.yaml
python -m src.train_h2m --config configs/h2m_train_x4.yaml
```

Or on Linux/server:

```bash
bash scripts/train_h2m_x2.sh
bash scripts/train_h2m_x4.sh
```

## Inference

```bash
python -m src.infer_h2m --config configs/h2m_infer_mouse.yaml --scale 2
python -m src.infer_h2m --config configs/h2m_infer_mouse.yaml --scale 4
```

Or:

```bash
bash scripts/infer_h2m.sh
```

## Outputs

By default, checkpoints and results are written under:

```text
experiments/h2m_net/
  x2/checkpoints/
  x4/checkpoints/
  results/
  metrics/
```

These folders are ignored by Git to avoid uploading large model weights or generated results.
