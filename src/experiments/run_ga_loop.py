# src/experiments/run_ga_loop.py
import argparse, os, json
import numpy as np

from src.utils.io import ensure_dir, save_json  # 你已有的话
# from src.experiments.cc_eval import evaluate_weights  # 推荐抽出来

def crossover(p1, p2, cx_rate=0.5):
    a = np.random.uniform(1-cx_rate, cx_rate)  # e.g. [0.5,0.5] -> fixed 0.5; you can widen
    return a*p1 + (1-a)*p2

def mutate(w, mut_rate, sigma, w_max):
    w = w.copy()
    n = w.shape[0]
    m = int(mut_rate * n)
    if m <= 0: 
        return w
    idx = np.random.choice(n, size=m, replace=False)
    w[idx] += np.random.normal(0.0, sigma, size=m)
    np.clip(w, 0.0, w_max, out=w)
    return w

def main():
    p = argparse.ArgumentParser()
    # 复用 CC 参数
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt_e", type=str, required=True)
    p.add_argument("--ckpt_orig_e1", type=str, required=True)
    p.add_argument("--P_train", type=str, required=True)
    p.add_argument("--targets", type=str, required=True)
    p.add_argument("--train_aug", type=int, default=0)
    p.add_argument("--w_max", type=float, default=10.0)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--lambda_shortfall", type=float, default=50.0)
    p.add_argument("--out_dir", type=str, required=True)

    # GA 参数
    p.add_argument("--pop", type=int, default=30)
    p.add_argument("--gens", type=int, default=15)
    p.add_argument("--elite", type=int, default=5)
    p.add_argument("--mut_rate", type=float, default=0.1)
    p.add_argument("--mut_sigma", type=float, default=0.5)
    p.add_argument("--cx_rate", type=float, default=0.5)
    p.add_argument("--seed_weights", type=str, default="")  # 可选：best_weights.npy
    args = p.parse_args()

    np.random.seed(args.seed)
    ensure_dir(args.out_dir)

    targets = [int(x) for x in args.targets.split(",")]

    # === load P_train to infer N ===
    P = np.load(args.P_train)      # 你自己的格式：可能是 [N,K] 或类似
    N = P.shape[0]

    # === init population ===
    pop = []
    if args.seed_weights:
        w0 = np.load(args.seed_weights)
        pop.append(np.clip(w0, 0.0, args.w_max))
    while len(pop) < args.pop:
        w = np.random.uniform(0.0, args.w_max, size=N).astype(np.float32)
        pop.append(w)

    history = []

    for gen in range(args.gens):
        scored = []
        for i, w in enumerate(pop):
            # TODO: 用你 CC 的“评估函数”
            # res = evaluate_weights(weights=w, args=args, targets=targets)
            # scored.append((res["fit"], res))

            pass

        # TODO: sort, select elites, breed next generation
        # 保存 best + history

    save_json(os.path.join(args.out_dir, "history.json"), {"history": history})

if __name__ == "__main__":
    main()
