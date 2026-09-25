"""Plot one or more runs' metrics.csv side by side.

    python -m part2_gpt.plot runs/baseline [runs/other ...] [--out plot.png]
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [("train_loss", "linear"), ("val_loss", "linear"), ("grad_norm", "log"), ("update_ratio", "log"), ("lr", "linear")]


def read(run: Path) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {k: [] for k, _ in PANELS}
    with open(run / "metrics.csv") as f:
        for row in csv.DictReader(f):
            for k in series:
                if row.get(k) not in (None, ""):
                    series[k].append((int(row["step"]), float(row[k])))
    return series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4 * len(PANELS), 3.2))
    for run in args.runs:
        s = read(run)
        for ax, (k, scale) in zip(axes, PANELS):
            if s[k]:
                xs, ys = zip(*s[k])
                ax.plot(xs, ys, label=run.name)
    for ax, (k, scale) in zip(axes, PANELS):
        ax.set_title(k)
        ax.set_yscale(scale)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = args.out or args.runs[0] / "metrics.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
