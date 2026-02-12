# src/experiments/run_plot_51.py
from __future__ import annotations

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def compute_cum_from_ptrain(inf_dir: str, row_names: list[str], K: int = 10) -> np.ndarray:
    """
    Recompute cum_influence[row, k] = sum_{i in removed(row)} P_train[i, k]
    using P_train.npy + top_lists/class{t}_{mode}.npy

    inf_dir: directory that contains P_train.npy and top_lists/.
            If your influence dir is different from out_dir, pass it via arg.
    """
    P_path = os.path.join(inf_dir, "P_train.npy")
    top_dir = os.path.join(inf_dir, "top_lists")
    if not os.path.exists(P_path):
        raise FileNotFoundError(f"Missing {P_path}")
    if not os.path.isdir(top_dir):
        raise FileNotFoundError(f"Missing {top_dir}")

    P = np.load(P_path)  # [N, K]
    cum = np.zeros((len(row_names), K), dtype=np.float32)

    for r, key in enumerate(row_names):
        # key looks like "t2_beneficial"
        if not key.startswith("t"):
            continue
        t = int(key.split("_")[0][1:])  # between 't' and '_'
        mode = "beneficial" if "beneficial" in key else "detrimental"
        idx_path = os.path.join(top_dir, f"class{t}_{mode}.npy")
        if not os.path.exists(idx_path):
            # if missing, keep zeros
            continue
        removed = np.load(idx_path)
        cum[r] = P[removed].sum(axis=0)
    return cum


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
    
    K = m.shape[1]
    R = len(row_names)
    size = max(6, 0.6 * max(K, R))
    plt.figure(figsize=(size, size))
    im = plt.imshow(m, aspect="equal", cmap="RdBu_r", norm=norm)  # RdBu_r: negative->blue, positive->red
    plt.gca().set_aspect('equal', adjustable='box')
    
    # plt.figure(figsize=(10, max(4, 0.35 * len(row_names))))
    # im = plt.imshow(m, aspect="auto", cmap="RdBu_r", norm=norm)  
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
    p.add_argument("--inf_dir", type=str, default="", help="Directory containing P_train.npy and top_lists/. If empty, use out_dir.")

    args = p.parse_args()

    out_dir = args.out_dir
    K = args.K

    obj = _load_results(out_dir)
    row_names, acc_change, _ = _rebuild_mats(obj, K=K)

    # If cum_influence.npy exists and aligned, load it; else skip cum heatmap
    # cum = _load_cum_if_available(out_dir, row_names, K=K)

    inf_dir = args.inf_dir.strip() if args.inf_dir.strip() else out_dir
    try:
        cum = compute_cum_from_ptrain(inf_dir, row_names, K=K)
        print("cum computed from:", inf_dir)
    except Exception as e:
        print("Failed to compute cum from P_train/top_lists, fallback to cum_influence.npy. Reason:", repr(e))
        cum = _load_cum_if_available(out_dir, row_names, K=K)
    
    if cum is None:
        print("cum is None (no P_train/top_lists and no aligned cum_influence.npy).")

    

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
    # if cum is not None:
    #     xs = cum.reshape(-1)
    #     ys = acc_change.reshape(-1)
    #     m = np.isfinite(xs) & np.isfinite(ys)
    #     if m.any():
    #         rho = spearmanr_approx(xs[m], ys[m])
    #         plt.figure()
    #         plt.scatter(xs[m], ys[m], s=10)
    #         plt.title(f"Scatter: cumulative influence vs acc change (Spearman ~ {rho:.3f})")
    #         plt.xlabel("Cumulative influence")
    #         plt.ylabel("Accuracy change")
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(out_dir, "scatter_spearman.png"), dpi=200)
    #         plt.show()
    #         print("Spearman approx:", rho)
    #     else:
    #         print("No finite points for scatter/spearman.")
    # else:
    #     print("cum_influence.npy not available/aligned; skip scatter/spearman.")

    # def plot_scatter_for_mode(mode: str, save_name: str):
    #     xs, ys = [], []
    #     for i, key in enumerate(row_names):
    #         if mode not in key:
    #             continue
    
    #         # key = "t2_beneficial" -> target = 2
    #         t = int(key.split("_")[0][1:])
    
    #         # 
    #         x = float(cum[i, t])          # cum influence on target dimension
    #         y = float(acc_change[i, t])   # delta acc on target class
    
    #         if np.isfinite(x) and np.isfinite(y):
    #             xs.append(x); ys.append(y)
    
    #     if len(xs) < 3:
    #         print(f"Not enough points for scatter ({mode}).")
    #         return
    
    #     rho = spearmanr_approx(np.array(xs), np.array(ys))
    
    #     plt.figure(figsize=(6, 6))
    #     plt.scatter(xs, ys, s=35)
    #     plt.axhline(0, linewidth=1)
    #     plt.axvline(0, linewidth=1)
    #     plt.title(f"{mode}: cum(target) vs delta(target)  (Spearman≈{rho:.3f})")
    #     plt.xlabel("cum influence on target dimension")
    #     plt.ylabel("accuracy change on target class")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(out_dir, save_name), dpi=300)
    #     plt.show()
    
    #     print(f"{mode} Spearman approx:", rho)
    
    
    # if cum is not None:
    #     plot_scatter_for_mode("beneficial", "scatter_target_beneficial.png")
    #     plot_scatter_for_mode("detrimental", "scatter_target_detrimental.png")
    # else:
    #     print("cum not available; skip scatter.")

def plot_scatter_target_only(mode: str, save_name: str):
    xs, ys = [], []
    for i, key in enumerate(row_names):
        if mode not in key:
            continue
        t = int(key.split("_")[0][1:])
        x = - float(cum[i, t])
        y = float(acc_change[i, t])
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x); ys.append(y)

    if len(xs) < 3:
        print(f"Not enough points (target-only) for {mode}.")
        return

    rho = spearmanr_approx(np.array(xs), np.array(ys))
    plt.figure(figsize=(6,6))
    plt.scatter(xs, ys, s=60)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.title(f"{mode} target-only (Spearman≈{rho:.3f})")
    plt.xlabel("cum influence on target dimension")
    plt.ylabel("accuracy change on target class")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, save_name), dpi=300)
    plt.show()
    print(mode, "target-only Spearman approx:", rho)


    def plot_scatter_classwise(mode: str, save_name: str):
        xs, ys = [], []
        for i, key in enumerate(row_names):
            if mode not in key:
                continue
            for k in range(K):
                x = - float(cum[i, k])
                y = float(acc_change[i, k])
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x); ys.append(y)
    
        xs = np.array(xs); ys = np.array(ys)
        if xs.size < 10:
            print(f"Not enough points (class-wise) for {mode}.")
            return
    
        rho = spearmanr_approx(xs, ys)
    
        a, b = np.polyfit(xs, ys, deg=1)
        xline = np.linspace(xs.min(), xs.max(), 200)
        yline = a * xline + b
    
        plt.figure(figsize=(6,6))
        plt.scatter(xs, ys, s=20)
        plt.plot(xline, yline, linewidth=2)
        plt.axhline(0, linewidth=1)
        plt.axvline(0, linewidth=1)
        plt.title(f"{mode} class-wise (Spearman≈{rho:.3f}, N={xs.size})")
        plt.xlabel("Class-wise influences (cum over removed set)")
        plt.ylabel("Per-class accuracy differences (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, save_name), dpi=300)
        plt.show()
        print(mode, "class-wise Spearman approx:", rho, "N=", xs.size)
    
    
    if cum is not None:
        plot_scatter_target_only("beneficial", "scatter_target_beneficial.png")
        plot_scatter_target_only("detrimental", "scatter_target_detrimental.png")
    
        plot_scatter_classwise("beneficial", "scatter_classwise_beneficial.png")
        plot_scatter_classwise("detrimental", "scatter_classwise_detrimental.png")
    else:
        print("cum not available; skip scatter.")



if __name__ == "__main__":
    main()
