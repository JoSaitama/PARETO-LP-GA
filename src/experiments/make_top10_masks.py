# src/experiments/make_top_lists.py
import os, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inf_dir", type=str, required=True, help="dir containing P_train.npy")
    ap.add_argument("--top_frac", type=float, default=0.10)
    ap.add_argument("--out_subdir", type=str, default="top_lists")
    args = ap.parse_args()

    P_path = os.path.join(args.inf_dir, "P_train.npy")
    P = np.load(P_path)  # [N,K]
    N, K = P.shape
    topn = int(round(N * args.top_frac))

    out_dir = os.path.join(args.inf_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    for k in range(K):
        s = P[:, k]
        order = np.argsort(s)  # ascending

        detrimental = order[:topn]      # most negative
        beneficial  = order[-topn:]     # most positive

        np.save(os.path.join(out_dir, f"class{k}_beneficial.npy"), beneficial.astype(np.int64))
        np.save(os.path.join(out_dir, f"class{k}_detrimental.npy"), detrimental.astype(np.int64))

        # quick sanity: check label composition if you want (optional)
        print(f"class{k}: saved beneficial[{len(beneficial)}], detrimental[{len(detrimental)}]")

    print("[DONE] top_lists regenerated at:", out_dir)

if __name__ == "__main__":
    main()
