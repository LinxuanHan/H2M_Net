# Zero-shot Cross-species Steered Diffusion Model for Mouse $^{129}\text{Xe}$ Lung MRI Super-resolution

This project trains a human-prior conditional diffusion model on 2D human `129Xe` lung MR slices and performs zero-shot mouse super-resolution inference.

## Data Layout

The code follows the current 2D folder layout:

- `data/human/train/raw`: human HR training slices
- `data/mouse/test/down_2x_bilinear`: mouse LR x2 test slices
- `data/mouse/test/down_4x_bilinear`: mouse LR x4 test slices
- `data/mouse/test/raw`: mouse HR reference slices for evaluation only

Mouse data is never used for training or fine-tuning.

## Train

```bash
conda run -n sr python -m src.train --config configs/train_x2.yaml
conda run -n sr python -m src.train --config configs/train_x4.yaml
```

For a quick smoke run:

```bash
conda run -n sr python -m src.train --config configs/train_x2.yaml --epochs 1 --batch_size 2 --limit 8 --sampling_steps 5
```

## Inference

```bash
conda run -n sr python -m src.inference --config configs/infer_mouse.yaml --scale 2 --guidance_scale 1.5
conda run -n sr python -m src.inference --config configs/infer_mouse.yaml --scale 4 --guidance_scale 1.5
```

Outputs are saved under `experiments/steered_diffusion/`.

## 8-GPU Server Run

Copy the whole project folder to the Linux server, enter the project root, then run:

```bash
conda create -n sr python=3.10 -y
conda activate sr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
chmod +x scripts/server_*.sh
bash scripts/server_run_all.sh
```

Training uses DDP:

```bash
torchrun --standalone --nproc_per_node=8 -m src.train --config configs/train_x2_server.yaml --no_preview
torchrun --standalone --nproc_per_node=8 -m src.train --config configs/train_x4_server.yaml --no_preview
```

Inference uses 8 independent GPU shards and then merges metric CSV files:

```bash
bash scripts/server_infer.sh
```

If the server has a different environment name, activate that environment instead of `sr`.

## Steering Components

- Human-prior steering: the denoiser is trained only from human HR slices.
- LR-content steering: LR-upsampled mouse slices are encoded as conditional input.
- Scale steering: x2/x4 is injected with a scale embedding.
- Classifier-free steering: inference uses `eps_uncond + guidance_scale * (eps_cond - eps_uncond)`.
- Data consistency steering: low-frequency projection keeps sampled SR aligned with observed LR content.
