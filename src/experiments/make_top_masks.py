# src/experiments/make_top_lists.py
import os
import argparse
import numpy as np
from torchvision import datasets
from src.data.transforms import cifar10_noaug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inf_dir", type=str, required=True, help="dir containing P_train.npy")
    ap.add_argument("--data_root", type=str, required=True, help="CIFAR10 root (same as training)")
    ap.add_argument("--top_frac", type=float, default=0.10, help="fraction to select (e.g., 0.10)")
    ap.add_argument("--out_subdir", type=str, default="top_lists")
    ap.add_argument("--within_class", type=int, default=1,
                    help="1: select within each class (recommended); 0: select over all samples")
    ap.add_argument("--sanity_print", type=int, default=1, help="1: print label hist for each saved list")
    args = ap.parse_args()

    # ---- load P_train ----
    P_path = os.path.join(args.inf_dir, "P_train.npy")
    assert os.path.exists(P_path), f"missing {P_path}"
    P = np.load(P_path)  # [N, K]
    N, K = P.shape

    # ---- load CIFAR10 train labels (noaug transform is fine; we only need targets) ----
    train_ds = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=cifar10_noaug(),
    )
    labels = np.array(train_ds.targets, dtype=np.int64)
    assert len(labels) == N, f"Label length {len(labels)} != P rows {N}. Check data_root / dataset split."

    out_dir = os.path.join(args.inf_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    top_frac = float(args.top_frac)
    assert 0.0 < top_frac < 1.0, "top_frac must be in (0,1)"

    for k in range(K):
        if args.within_class:
            idx_pool = np.where(labels == k)[0]          # global indices of class k samples
            scores = P[idx_pool, k]                      # influence of those samples on class k
            topn = int(round(len(idx_pool) * top_frac))  # e.g., 500 for CIFAR10 if top_frac=0.1
            topn = max(1, topn)
            order = np.argsort(scores)                   # ascending (most negative first)

            detrimental = idx_pool[order[:topn]]          # global indices
            beneficial  = idx_pool[order[-topn:]]         # global indices
        else:
            scores = P[:, k]
            topn = int(round(N * top_frac))
            topn = max(1, topn)
            order = np.argsort(scores)

            detrimental = order[:topn]                    # global indices already
            beneficial  = order[-topn:]

        # save
        np.save(os.path.join(out_dir, f"class{k}_beneficial.npy"), beneficial.astype(np.int64))
        np.save(os.path.join(out_dir, f"class{k}_detrimental.npy"), detrimental.astype(np.int64))

        # sanity
        if args.sanity_print:
            hb = np.bincount(labels[beneficial], minlength=K)
            hd = np.bincount(labels[detrimental], minlength=K)
            print(f"class{k}: topn={topn} within_class={bool(args.within_class)}")
            print(f"  beneficial  label_hist={hb}  target_ratio={hb[k]/max(1,hb.sum()):.4f}")
            print(f"  detrimental label_hist={hd}  target_ratio={hd[k]/max(1,hd.sum()):.4f}")

    print("[DONE] top_lists regenerated at:", out_dir)


if __name__ == "__main__":
    main()
