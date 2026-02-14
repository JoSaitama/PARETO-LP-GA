# src/experiments/run_delete_retrain.py
from __future__ import annotations

"""
Delete & retrain experiment for Section 5.1:
- read P_train.npy + top_lists/ from influence output
- delete top beneficial / detrimental samples for each target class
- retrain for a short schedule
- record per-class accuracy changes and cumulative influence
- supports resume for Colab interruptions
"""

import argparse
import os
import hashlib
import numpy as np
import torch
import torch.optim as optim

from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json, load_json
from src.data.transforms import cifar10_noaug, cifar10_train_aug
from src.models.resnet9 import ResNet9
from src.train.trainer import TrainConfig, train_one_epoch, evaluate
from src.experiments.parse_file import add_common_args, exp_dirs, ckpt_path


def _sha256_file(path: str, max_mb: int = 16) -> str:
    """Lightweight hash: read first `max_mb` MB to avoid huge I/O on Drive."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        chunk = f.read(max_mb * 1024 * 1024)
        h.update(chunk)
    return h.hexdigest()


def retrain_once(
    train_ds_full,
    test_loader,
    device: str,
    removed_idx: np.ndarray,
    epochs: int,
    train_batch_size: int,
    num_workers: int,
    init_state_dict=None,
    # optimizer/scheduler params
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    scheduler_type: str = "cosine",  # "cosine" | "multistep" | "none"
    ms_milestones: str = "50,75",    # used if multistep
    ms_gamma: float = 0.1,
    print_every: int = 10,
):
    N = len(train_ds_full)
    removed_set = set(removed_idx.tolist())
    keep_idx = [i for i in range(N) if i not in removed_set]

    train_ds = Subset(train_ds_full, keep_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    model = ResNet9(num_classes=10).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    cfg = TrainConfig(
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        device=device,
        num_classes=10,
        use_amp=(device == "cuda"),
    )

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    # scheduler
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif scheduler_type == "multistep":
        if ms_milestones.strip() == "":
            milestones = []
        else:
            milestones = [int(x) for x in ms_milestones.split(",") if x.strip() != ""]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=ms_gamma)
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler_type={scheduler_type}")

    scaler = GradScaler("cuda", enabled=cfg.use_amp)

    best = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, cfg, scaler)
        te = evaluate(model, test_loader, cfg)

        if scheduler is not None:
            scheduler.step()

        if te["acc"] > best:
            best = float(te["acc"])
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}


        if ep == 1 or ep % print_every == 0 or ep == epochs:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  ep {ep:02d}/{epochs} lr={cur_lr:.5f} test_acc={te['acc']:.2f} best={best:.2f}")

    model.load_state_dict(best_state)
    final = evaluate(model, test_loader, cfg)
    return {
        "best_acc": float(final["acc"]),
        "per_class_acc": np.array(final["per_class_acc"], dtype=np.float32),
        "keep_size": len(keep_idx),
    }


def parse_targets(s: str, num_classes: int = 10):
    s = s.strip().lower()
    if s == "all":
        return list(range(num_classes))
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    p = argparse.ArgumentParser()

    # 公共默认参数：data_root/output_root/seed/batch_size/num_workers/epochs/aug...
    add_common_args(p)

    # ===== delete-retrain 专属参数 =====
    p.add_argument("--inf_dir", type=str, default="", help="Optional. If empty, auto infer from output_root.")
    p.add_argument("--base_ckpt", type=str, required=True, help="Required. If empty, please set the baseline best.pt.")
    p.add_argument("--out_dir", type=str, default="", help="Optional. If empty, auto infer from output_root.")

    p.add_argument("--targets", type=str, default="all", help='"all" or "0,1,2"')

    p.add_argument("--resume", type=int, default=1)
    p.add_argument("--init_from_ckpt", type=int, default=0, help="1: retrain initializes from baseline ckpt; 0: from scratch")

    # retrain data augmentation
    p.add_argument("--noaug", type=int, default=1, help="1: retrain uses no-aug; 0: uses train aug")

    # ===== NEW: retrain hyperparams (CLI) =====
    p.add_argument("--train_bs", type=int, default=256, help="Retrain batch size (train loader)")
    p.add_argument("--eval_bs", type=int, default=256, help="Eval batch size (test loader)")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)

    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "multistep", "none"])
    p.add_argument("--ms_milestones", type=str, default="50,75", help='e.g. "30,45"')
    p.add_argument("--ms_gamma", type=float, default=0.1)

    p.add_argument("--print_every", type=int, default=10)

    # for sanity debug
    p.add_argument("--sanity_no_delete", type=int, default=0,
               help="1: run one retrain with removed_idx empty and exit")


    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== infer default dirs =====
    dirs = exp_dirs(args.output_root, "resnet9_cifar10", args.seed)

    baseline_dir = dirs["baseline"]

    # NOTE: 这里你现在默认是 headonly 命名；如果你 EKFAC 的 influence_dir 命名不同，
    # 建议运行时直接传 --inf_dir 来覆盖，避免找错文件夹。
    inf_default = os.path.join(
        args.output_root,
        f"influence_resnet9_cifar10_seed{args.seed}_headonly_" + ("noaug" if args.noaug else "aug"),
    )
    out_default = os.path.join(
        args.output_root,
        f"delete_retrain_resnet9_cifar10_seed{args.seed}_headonly_" + ("noaug" if args.noaug else "aug"),
    )

    inf_dir = args.inf_dir or inf_default
    out_dir = args.out_dir or out_default
    base_ckpt = args.base_ckpt or ckpt_path(baseline_dir)

    ensure_dir(out_dir)

    # ===== load influence =====
    P_path = os.path.join(inf_dir, "P_train.npy")
    top_dir = os.path.join(inf_dir, "top_lists")
    assert os.path.exists(P_path), f"missing {P_path}"
    assert os.path.isdir(top_dir), f"missing {top_dir}"

    P = np.load(P_path)  # [N, 10]
    N, K = P.shape

    targets = parse_targets(args.targets, num_classes=K)
    modes = ["beneficial", "detrimental"]

    # ===== datasets/loaders =====
    train_tf = cifar10_noaug() if args.noaug else cifar10_train_aug()
    test_tf = cifar10_noaug()

    train_ds_full = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_tf)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # ===== baseline eval =====
    base_model = ResNet9(num_classes=10).to(device)
    ckpt_obj = torch.load(base_ckpt, map_location=device)

    # Compatible with multiple ckpt formats
    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        base_state = ckpt_obj["model_state"]
    elif isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        base_state = ckpt_obj["state_dict"]
    else:
        base_state = ckpt_obj

    base_model.load_state_dict(base_state, strict=False)
    base_model.eval()

    base_cfg = TrainConfig(epochs=1, device=device, num_classes=10, use_amp=False)
    base_eval = evaluate(base_model, test_loader, base_cfg)
    base_acc = np.array(base_eval["per_class_acc"], dtype=np.float32)
    base_overall = float(base_eval["acc"])

    # ===== resume =====
    partial_path = os.path.join(out_dir, "results_partial.json")
    done = set()
    results = []
    if args.resume and os.path.exists(partial_path):
        obj = load_json(partial_path)
        results = obj.get("results", [])
        done = set(obj.get("done", []))
        print(f"Resume: loaded {len(done)} finished runs from {partial_path}")

    # ===== allocate outputs with stable row indexing =====
    row_names = [f"t{t}_{m}" for t in targets for m in modes]
    name2row = {name: i for i, name in enumerate(row_names)}

    cum_infl = np.zeros((len(row_names), K), dtype=np.float32)
    acc_change = np.zeros((len(row_names), K), dtype=np.float32)

    labels = np.array(train_ds_full.targets)

    # ===== meta / provenance =====
    meta = {
        "device": device,
        "seed": int(args.seed),
        "data_root": args.data_root,
        "inf_dir": inf_dir,
        "base_ckpt": base_ckpt,
        "base_ckpt_name": os.path.basename(base_ckpt),
        "base_ckpt_sha256_first16mb": _sha256_file(base_ckpt),
        "retrain": {
            "epochs": int(args.epochs),
            "train_bs": int(args.train_bs),
            "eval_bs": int(args.eval_bs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "momentum": float(args.momentum),
            "scheduler": args.scheduler,
            "ms_milestones": args.ms_milestones,
            "ms_gamma": float(args.ms_gamma),
            "init_from_ckpt": bool(args.init_from_ckpt),
            "noaug": bool(args.noaug),
        },
    }
    save_json(os.path.join(out_dir, "meta.json"), meta)

    if args.sanity_no_delete:
        print("\n[SANITY] Running retrain with NO deletion...")
        out = retrain_once(
            train_ds_full=train_ds_full,
            test_loader=test_loader,
            device=device,
            removed_idx=np.array([], dtype=np.int64),
            epochs=args.epochs,
            train_batch_size=args.train_bs,
            num_workers=args.num_workers,
            init_state_dict=(base_state if args.init_from_ckpt else None),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            scheduler_type=args.scheduler,
            ms_milestones=args.ms_milestones,
            ms_gamma=args.ms_gamma,
            print_every=args.print_every,
        )
        print("[SANITY] overall_acc:", out["best_acc"])
        print("[SANITY] per_class_acc:", out["per_class_acc"])
        return

    for t in targets:
        for mode in modes:
            key = f"t{t}_{mode}"
            row = name2row[key]

            if key in done:
                continue

            removed = np.load(os.path.join(top_dir, f"class{t}_{mode}.npy"))

            print(f"\n=== target={t}, mode={mode}, removed={len(removed)} ===")
            cum = P[removed].sum(axis=0).astype(np.float32)
            cum_infl[row] = cum

            hist = np.bincount(labels[removed], minlength=K)
            target_ratio = float(hist[t] / max(1, hist.sum()))
            print("cum influence on target class:", float(cum[t]))
            print("removed label hist:", hist)
            print("removed target ratio:", target_ratio)

            init_state = base_state if args.init_from_ckpt else None

            out = retrain_once(
                train_ds_full=train_ds_full,
                test_loader=test_loader,
                device=device,
                removed_idx=removed,
                epochs=args.epochs,
                train_batch_size=args.train_bs,
                num_workers=args.num_workers,
                init_state_dict=init_state,
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
                scheduler_type=args.scheduler,
                ms_milestones=args.ms_milestones,
                ms_gamma=args.ms_gamma,
                print_every=args.print_every,
            )

            new_acc = out["per_class_acc"]
            delta = new_acc - base_acc
            acc_change[row] = delta

            rec = {
                "key": key,
                "target": t,
                "mode": mode,
                "removed": int(len(removed)),
                "keep_size": int(out["keep_size"]),
                "baseline_overall": base_overall,
                "baseline_per_class": base_acc.tolist(),
                "overall_acc": float(out["best_acc"]),
                "per_class_acc": new_acc.tolist(),
                "delta_target": float(delta[t]),
                "removed_label_hist": hist.tolist(),
                "removed_target_ratio": target_ratio,
                "noaug_retrain": bool(args.noaug),
                "seed": int(args.seed),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "momentum": float(args.momentum),
                "scheduler": args.scheduler,
                "init_from_ckpt": bool(args.init_from_ckpt),
            }

            results.append(rec)
            done.add(key)


            save_json(partial_path, {"done": sorted(list(done)), "results": results})

            print("overall_acc:", out["best_acc"])
            print("delta(target class):", float(delta[t]))

    # ===== final save =====
    np.save(os.path.join(out_dir, "cum_influence.npy"), cum_infl)
    np.save(os.path.join(out_dir, "acc_change.npy"), acc_change)
    save_json(os.path.join(out_dir, "results.json"), {"row_names": row_names, "results": results})

    print("\n=== Delete & Retrain DONE ===")
    print("device:", device)
    print("data_root:", args.data_root)
    print("baseline_ckpt:", base_ckpt)
    print("inf_dir:", inf_dir)
    print("out_dir:", out_dir)
    print("noaug_retrain:", bool(args.noaug))
    print("Saved:", os.path.join(out_dir, "results.json"))


if __name__ == "__main__":
    main()
