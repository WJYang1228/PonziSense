#!/usr/bin/env python3
"""
自动化超参搜索（Optuna），仅调 **本仓库已实现** 的可训练超参，主目标为验证集 **F1**。

论文方法论备注（为何不全按 λ1/λ2/τ 搜索）::

1. L_con、L_clu 仍为占位 0，搜 λ1/λ2 **无梯度意义**；需先实现对比/聚类损失再扩空间。
2. 多损失若尺度不一，应做 loss 归一化或分阶段训练，否则 HPO 易偏向梯度大的项。
3. μ（explain_loss_weight）与 lr、wd 强耦合，**联合 HPO 比手抄论文单点更稳**。

用法（仓库根目录）::

    pip install optuna
    python hpo/run_optuna.py --n-trials 20 --epochs-per-trial 10

结果：``outputs/hpo_study/hpo_best.json``；各 trial 权重在 ``outputs/hpo_study/trial_<id>/``。
正式训练请把最优参数写回 ``configs/config.py``，并把 ``OUTPUT_DIR`` 设回 ``./outputs``、``EPOCHS`` 设回全量轮数。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

try:
    import requests
except ImportError:
    requests = None  # type: ignore

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from configs.config import Config
from train import run_training


def build_cfg(trial: optuna.Trial, base: Config, epochs: int, out_root: str) -> Config:
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.06)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.12)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.02, 0.12)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 2.0)
    explain_w = trial.suggest_float("explain_loss_weight", 0.03, 0.35)
    accum = trial.suggest_int("gradient_accumulation_steps", 1, 4)

    trial_dir = os.path.join(out_root, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    return replace(
        base,
        LR=lr,
        WEIGHT_DECAY=weight_decay,
        LABEL_SMOOTHING=label_smoothing,
        WARMUP_RATIO=warmup_ratio,
        MAX_GRAD_NORM=max_grad_norm,
        EXPLAIN_LOSS_WEIGHT=explain_w,
        GRADIENT_ACCUMULATION_STEPS=accum,
        EPOCHS=epochs,
        OUTPUT_DIR=trial_dir,
        EARLY_STOPPING_PATIENCE=3,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--epochs-per-trial", type=int, default=10)
    ap.add_argument("--out-root", type=str, default="./outputs/hpo_study")
    ap.add_argument("--study-name", type=str, default="ponzi_graphcodebert")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    storage = f"sqlite:///{os.path.join(args.out_root, 'optuna.db')}"

    base = Config()
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = build_cfg(trial, base, args.epochs_per_trial, args.out_root)
        try:
            result = run_training(cfg, run_test=False, quiet=True, optuna_trial=trial)
        except optuna.TrialPruned:
            raise
        return result["best_val_f1"]

    # 单次 trial 若因 Hugging Face 下载超时等网络问题失败，不终止整次 study
    _catch: tuple = (OSError, TimeoutError, ConnectionError)
    if requests is not None:
        _catch = _catch + (requests.exceptions.RequestException,)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True, catch=_catch)

    try:
        best = study.best_trial
    except ValueError:
        print(
            "\n无已完成 trial（可能全部因网络/缓存失败）。"
            "请先在本机缓存模型：HF_HOME 可指向本地，或提前运行一次 python train.py 完成下载。"
        )
        return
    best_params = dict(best.params)
    best_params["_best_val_f1"] = best.value
    best_params["_epochs_per_trial"] = args.epochs_per_trial

    out_json = os.path.join(args.out_root, "hpo_best.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    print("\n=== Best trial (validation F1) ===")
    print(json.dumps(best_params, indent=2, ensure_ascii=False))
    print("\nSaved:", out_json)
    print(
        "\n下一步：将 lr, weight_decay, label_smoothing, warmup_ratio, max_grad_norm, "
        "explain_loss_weight, gradient_accumulation_steps 写入 configs/config.py；"
        "将 EPOCHS 设为正式轮数，OUTPUT_DIR 设为 ./outputs，再运行 python train.py"
    )


if __name__ == "__main__":
    main()
