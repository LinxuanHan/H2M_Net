import argparse
import csv
from pathlib import Path

from src.metrics import save_metric_tables
from src.utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Merge sharded inference metric CSV files.")
    parser.add_argument("--metric_dir", default="./experiments/steered_diffusion/metrics")
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    metric_dir = Path(args.metric_dir)
    tag = args.tag or f"x{args.scale}_g{args.guidance_scale:g}"
    pattern = f"steered_diffusion_{tag}_shard*of*_per_image.csv"
    files = sorted(metric_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No sharded metric files found: {metric_dir / pattern}")
    rows = []
    for path in files:
        with open(path, "r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                rows.append(row)
    ensure_dir(metric_dir)
    summary = save_metric_tables(
        rows,
        metric_dir / f"steered_diffusion_{tag}_merged_per_image.csv",
        metric_dir / f"steered_diffusion_{tag}_merged_summary.csv",
    )
    print(summary)


if __name__ == "__main__":
    main()
