#!/usr/bin/env python3
"""
RQ1: 合同级检测主表指标（与论文 Table 主分类一致：P/R/F1/AUC/AUPRC/MCC）。

用法（在仓库根目录）:
  python experiments/rq1/run_main_metrics.py

依赖: 已训练 best_model.pt；数据划分与 configs/config.py 一致。
论文中的 Ridge/SVM 等基线需单独实现，本脚本仅输出本工程（GraphCodeBERT+PonziModel）结果。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 须先于 ``import experiments``：直接 python experiments/rq1/xxx.py 时 cwd 不在包根上
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from torch.utils.data import DataLoader

from experiments.common.project import ensure_repo_importable

ensure_repo_importable()

from experiments.common.classification_metrics import binary_metrics_from_scores
from experiments.common.forward_dataset import collect_ponzi_probs_and_labels
from experiments.common.load_model import load_trained_ponzimodel

from configs.config import Config  # noqa: E402
from data.collate import collate_fn  # noqa: E402
from data.dataset import load_datasets  # noqa: E402
from utils.io import ensure_dir  # noqa: E402
from experiments.common.output_paths import figure_dir, table_dir  # noqa: E402
from experiments.common.plot_pdf import save_bar_pdf, save_table_pdf  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5, help="二分类阈值（论文中可扫 RQ4）")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    args = parser.parse_args()

    cfg = Config()
    model, tokenizer, _, device = load_trained_ponzimodel()

    _, val_set, test_set = load_datasets(tokenizer, cfg)
    ds = test_set if args.split == "test" else val_set
    loader = DataLoader(
        ds,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    y_true, y_score = collect_ponzi_probs_and_labels(model, tokenizer, loader, device, cfg)
    metrics = binary_metrics_from_scores(y_true, y_score, threshold=args.threshold)
    out = {
        "rq": "RQ1",
        "split": args.split,
        "threshold": args.threshold,
        "n": len(y_true),
        "metrics": metrics,
        "note": "Baselines (Ridge-NC, SVM-NC, ...) are not in this repo; fill paper table separately.",
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    out_dir = os.path.join(cfg.OUTPUT_DIR, "logs", "experiments")
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"rq1_main_metrics_{args.split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved:", path)

    m = metrics
    fig_dir = figure_dir(cfg, "rq1")
    pdf_bar = os.path.join(fig_dir, f"main_metrics_bar_{args.split}.pdf")
    save_bar_pdf(
        pdf_bar,
        labels=["Precision", "Recall", "F1", "AUC", "AUPRC", "MCC"],
        values=[
            m["precision"],
            m["recall"],
            m["f1"],
            m["auc"],
            m["auprc"],
            m["mcc"],
        ],
        title=f"RQ1 classification (split={args.split}, τ={args.threshold})",
        ylabel="Score",
    )
    tbl = os.path.join(table_dir(cfg), f"rq1_main_metrics_{args.split}.pdf")
    save_table_pdf(
        tbl,
        headers=["Metric", "Value"],
        rows=[
            ["Precision", f"{m['precision']:.4f}"],
            ["Recall", f"{m['recall']:.4f}"],
            ["F1", f"{m['f1']:.4f}"],
            ["AUC", f"{m['auc']:.4f}"],
            ["AUPRC", f"{m['auprc']:.4f}"],
            ["MCC", f"{m['mcc']:.4f}"],
        ],
        title=f"RQ1 main metrics ({args.split})",
    )
    print("PDF:", pdf_bar, tbl)


if __name__ == "__main__":
    main()
