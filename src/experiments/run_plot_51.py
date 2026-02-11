# src/experiments/run_plot_51.py
from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def spearmanr_approx(x, y) -> float:
    x = np.asarray(x); y = np.asarray(y)
    rx = x.argsort().argsort().astype(np.float32)
    ry = y.argsort().argsort().astype(np.float32)
    rx = (rx - rx.mean()) / (rx.std() + 1e-8)
    ry = (ry - ry.mean()) / (ry.std() + 1e-8)
    return float((rx * ry).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)  # delete_retrain output dir
    args = p.parse_args()

    cum_path = os.path.join(args.out_dir, "cum_influence.npy")
    acc_path = os.path.join(args.out_dir, "acc_change.npy")
    res_path = os.path.join(args.out_dir, "results.json")

    cum = np.load(cum_path)
    acc = np.load(acc_path)
    obj = None
    try:
        import json
        with open(res_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        row_names = obj["row_names"]
    except Exception:
        row_names = [f"row{i}" for i in range(cum.shape[0])]

    # Heatmap: cum influence
    plt.figure()
    plt.imshow(cum, aspect="auto")
    plt.yticks(range(len(row_names)), row_names)
    plt.xticks(range(cum.shape[1]), list(range(cum.shape[1])))
    plt.colorbar()
    plt.title("Cumulative influence of removed samples")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "heatmap_cum_influence.png"), dpi=200)
    plt.show()

    # Heatmap: acc change
    plt.figure()
    plt.imshow(acc, aspect="auto")
    plt.yticks(range(len(row_names)), row_names)
    plt.xticks(range(acc.shape[1]), list(range(acc.shape[1])))
    plt.colorbar()
    plt.title("Per-class accuracy change after delete+retrain")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "heatmap_acc_change.png"), dpi=200)
    plt.show()

    xs = cum.reshape(-1)
    ys = acc.reshape(-1)
    rho = spearmanr_approx(xs, ys)

    plt.figure()
    plt.scatter(xs, ys, s=10)
    plt.title(f"Scatter: cumulative influence vs acc change (Spearman ~ {rho:.3f})")
    plt.xlabel("Cumulative influence")
    plt.ylabel("Accuracy change")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "scatter_spearman.png"), dpi=200)
    plt.show()

    print("Spearman approx:", rho)


if __name__ == "__main__":
    main()
