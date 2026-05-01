#!/usr/bin/env python3
"""
RQ4: 决策阈值敏感性 — 在多个阈值下报告 P/R/F1/AUC/AUPRC/MCC（AUC/AUPRC 与阈值无关）。

用法:
  python experiments/rq4/run_threshold_sweep.py --thresholds 0.3,0.5,0.7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.common.project import ensure_repo_importable

ensure_repo_importable()

from experiments.common.classification_metrics import binary_metrics_from_scores
from experiments.common.forward_dataset import collect_ponzi_probs_and_labels
from experiments.common.load_model import load_trained_ponzimodel

from configs.config import Config  # noqa: E402
from data.collate import collate_fn  # noqa: E402
from data.dataset import load_datasets  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from utils.io import ensure_dir  # noqa: E402
from experiments.common.output_paths import figure_dir, table_dir  # noqa: E402
from experiments.common.plot_pdf import save_lines_pdf, save_table_pdf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--thresholds", type=str, default="0.3,0.5,0.7")
    args = ap.parse_args()
    th_list = [float(x.strip()) for x in args.thresholds.split(",")]

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
    rows = []
    for t in th_list:
        m = binary_metrics_from_scores(y_true, y_score, threshold=t)
        rows.append({"threshold": t, **m})

    out = {"rq": "RQ4", "name": "threshold_sweep", "split": args.split, "n": len(y_true), "rows": rows}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    out_dir = os.path.join(cfg.OUTPUT_DIR, "logs", "experiments")
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"rq4_threshold_sweep_{args.split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved:", path)

    ths = [r["threshold"] for r in rows]
    fig_dir = figure_dir(cfg, "rq4")
    pdf_curve = os.path.join(fig_dir, f"threshold_curve_{args.split}.pdf")
    save_lines_pdf(
        pdf_curve,
        xs=ths,
        series={
            "Precision": [r["precision"] for r in rows],
            "Recall": [r["recall"] for r in rows],
            "F1": [r["f1"] for r in rows],
        },
        title=f"RQ4 threshold sensitivity ({args.split})",
        xlabel="Threshold",
        ylabel="Score",
    )
    tbl_rows = [[str(r["threshold"]), f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", f"{r['auc']:.4f}", f"{r['auprc']:.4f}", f"{r['mcc']:.4f}"] for r in rows]
    tbl = os.path.join(table_dir(cfg), f"rq4_threshold_{args.split}.pdf")
    save_table_pdf(
        tbl,
        headers=["τ", "P", "R", "F1", "AUC", "AUPRC", "MCC"],
        rows=tbl_rows,
        title=f"RQ4 threshold sweep ({args.split})",
        figsize=(12, 0.5 * (len(rows) + 2)),
    )
    print("PDF:", pdf_curve, tbl)


if __name__ == "__main__":
    main()
