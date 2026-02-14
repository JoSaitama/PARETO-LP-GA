# src/experiments/make_top10_masks.py
import argparse
import os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_train", type=str, required=True, help="Path to P_train.npy (N,K)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for class{k}_{beneficial|detrimental}.npy")
    ap.add_argument("--top_frac", type=float, default=0.10, help="Top fraction (default 0.10)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    P = np.load(args.p_train)  # (N,K)
    N, K = P.shape
    m = int(np.ceil(args.top_frac * N))

    print(f"[INFO] P shape={P.shape}, top_frac={args.top_frac}, per-class m={m}")

    for k in range(K):
        col = P[:, k]

        # beneficial = largest
        idx_ben = np.argpartition(col, -m)[-m:]
        # sort descending for reproducibility
        idx_ben = idx_ben[np.argsort(col[idx_ben])[::-1]]

        # detrimental = smallest
        idx_det = np.argpartition(col, m)[:m]
        idx_det = idx_det[np.argsort(col[idx_det])]

        np.save(os.path.join(args.out_dir, f"class{k}_beneficial.npy"), idx_ben.astype(np.int64))
        np.save(os.path.join(args.out_dir, f"class{k}_detrimental.npy"), idx_det.astype(np.int64))

        print(f"[K={k}] ben range=({col[idx_ben].min():.3f},{col[idx_ben].max():.3f}) "
              f"det range=({col[idx_det].min():.3f},{col[idx_det].max():.3f})")

    print("[DONE] saved to:", args.out_dir)

if __name__ == "__main__":
    main()
