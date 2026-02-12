# src/experiments/run_plot_51.py
from __future__ import annotations

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def spearmanr_approx(x, y) -> float:
    x = np.asarray(x); y = np.asarray(y)
    rx = x.argsort().argsort().astype(np.float32)
    ry = y.argsort().argsort().astype(np.float32)
    rx = (rx - rx.mean()) / (rx.std() + 1e-8)
    ry = (ry - ry.mean()) / (ry.std() + 1e-8)
    return float((rx * ry).mean())


def _load_results(out_dir: str) -> dict:
    """
    Prefer results.json; fallback to results_partial.json.
    """
    res_path = os.path.join(out_dir, "results.json")
    part_path = os.path.join(out_dir, "results_partial.json")

    if os.path.exists(res_path):
        with open(res_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if os.path.exists(part_path):
        with open(part_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # wrap into same shape as results.json
        row_names = [r["key"] for r in obj.get("results", [])]
        return {"row_names": row_names, "results": obj.get("results", [])}

    raise FileNotFoundError("Neither results.json nor results_partial.json exists in out_dir.")


def _rebuild_mats(obj: dict, K: int = 10):
    """
    Build:
      acc_change: [R, K]
      cum_influence: [R, K]
      row_names: length R
    from the records (robust to resume / missing npy).
    """
    row_names = obj.get("row_names", [])
    results = obj.get("results", [])

    # If row_names missing, infer from results order
    if not row_names:
        row_names = [r.get("key", f"row{i}") for i, r in enumerate(results)]

    # map key -> record
    rec_map = {r.get("key"): r for r in results if "key" in r}

    R = len(row_names)
    acc_change = np.full((R, K), np.nan, dtype=np.float32)
    cum_infl = np.full((R, K), np.nan, dtype=np.float32)

    for i, key in enumerate(row_names):
        r = rec_map.get(key, None)
        if r is None:
            continue

        # acc change (delta) is safest source
        base = np.array(r.get("baseline_per_class", [np.nan]*K), dtype=np.float32)
        new = np.array(r.get("per_class_acc", [np.nan]*K), dtype=np.float32)
        if np.isfinite(base).all() and np.isfinite(new).all():
            acc_change[i] = new - base

        # cum influence: if stored, use it; else leave nan
        # your run_delete_retrain currently doesn't persist per-row cum vector in json,
        # so we cannot reconstruct it reliably here. We'll use cum_influence.npy if exists.
    return row_names, acc_change, cum_infl


def _load_cum_if_available(out_dir: str, row_names: list[str], K: int = 10):
    """
    Try to load cum_influence.npy and align rows if possible.
    If it has correct shape, return it; else return None.
    """
    cum_path = os.path.join(out_dir, "cum_influence.npy")
    if not os.path.exists(cum_path):
        return None
    cum = np.load(cum_path)
    if cum.ndim == 2 and cum.shape[1] == K:
        # If row count matches, assume aligned
        if cum.shape[0] == len(row_names):
            return cum.astype(np.float32)
    return None


def _plot_heatmap(mat, row_names, title, save_path, vlim=None):
    """
    mat: [R, K] with possible NaNs
    Diverging colormap with center at 0: negative blue, positive red.
    """
    K = mat.shape[1]
    # mask NaN rows/entries
    m = np.array(mat, copy=True)
    # choose limits
    finite = np.isfinite(m)
    if vlim is None:
        if finite.any():
            vmax = float(np.nanmax(np.abs(m[finite])))
            vmax = max(vmax, 1e-6)
        else:
            vmax = 1.0
        vlim = (-vmax, vmax)

    norm = TwoSlopeNorm(vmin=vlim[0], vcenter=0.0, vmax=vlim[1])

    plt.figure(figsize=(10, max(4, 0.35 * len(row_names))))
    im = plt.imshow(m, aspect="auto", cmap="RdBu_r", norm=norm)  # RdBu_r: negative->blue, positive->red
    plt.yticks(range(len(row_names)), row_names)
    plt.xticks(range(K), list(range(K)))
    plt.colorbar(im)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--only", type=str, default="all", choices=["all", "beneficial", "detrimental"])
    args = p.parse_args()

    out_dir = args.out_dir
    K = args.K

    obj = _load_results(out_dir)
    row_names, acc_change, _ = _rebuild_mats(obj, K=K)

    # If cum_influence.npy exists and aligned, load it; else skip cum heatmap
    cum = _load_cum_if_available(out_dir, row_names, K=K)

    # split rows
    idx_all = np.arange(len(row_names))
    idx_ben = np.array([i for i, n in enumerate(row_names) if "beneficial" in n], dtype=int)
    idx_det = np.array([i for i, n in enumerate(row_names) if "detrimental" in n], dtype=int)

    def do_block(tag, idx):
        rn = [row_names[i] for i in idx]
        acc_blk = acc_change[idx]
        _plot_heatmap(
            acc_blk,
            rn,
            title=f"Per-class accuracy change after delete+retrain ({tag})",
            save_path=os.path.join(out_dir, f"heatmap_acc_change_{tag}.png"),
        )
        if cum is not None:
            cum_blk = cum[idx]
            _plot_heatmap(
                cum_blk,
                rn,
                title=f"Cumulative influence of removed samples ({tag})",
                save_path=os.path.join(out_dir, f"heatmap_cum_influence_{tag}.png"),
            )

    if args.only == "all":
        # 1) beneficial-only
        do_block("beneficial", idx_ben)
        # 2) detrimental-only
        do_block("detrimental", idx_det)
        # 3) combined
        do_block("all", idx_all)
    elif args.only == "beneficial":
        do_block("beneficial", idx_ben)
    else:
        do_block("detrimental", idx_det)

    # scatter & spearman (use valid entries only)
    if cum is not None:
        xs = cum.reshape(-1)
        ys = acc_change.reshape(-1)
        m = np.isfinite(xs) & np.isfinite(ys)
        if m.any():
            rho = spearmanr_approx(xs[m], ys[m])
            plt.figure()
            plt.scatter(xs[m], ys[m], s=10)
            plt.title(f"Scatter: cumulative influence vs acc change (Spearman ~ {rho:.3f})")
            plt.xlabel("Cumulative influence")
            plt.ylabel("Accuracy change")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "scatter_spearman.png"), dpi=200)
            plt.show()
            print("Spearman approx:", rho)
        else:
            print("No finite points for scatter/spearman.")
    else:
        print("cum_influence.npy not available/aligned; skip scatter/spearman.")


if __name__ == "__main__":
    main()
