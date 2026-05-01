#!/usr/bin/env python3
"""
RQ2: MSP / MSR / MIoU（论文公式与 utils/explain_metrics 一致）。

用法:
  python experiments/rq2/run_overlap_metrics.py [--split test]
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

import torch
from torch.utils.data import DataLoader

from experiments.common.project import ensure_repo_importable

ensure_repo_importable()

from experiments.common.load_model import load_trained_ponzimodel

from configs.config import Config  # noqa: E402
from data.collate import collate_fn  # noqa: E402
from data.dataset import load_datasets  # noqa: E402
from utils.explain_metrics import compute_explainability_macro  # noqa: E402
from utils.io import ensure_dir  # noqa: E402
from experiments.common.output_paths import figure_dir, table_dir  # noqa: E402
from experiments.common.plot_pdf import save_bar_pdf, save_table_pdf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], default="test")
    args = ap.parse_args()

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

    ex = compute_explainability_macro(
        model,
        tokenizer,
        loader,
        device,
        cfg,
        desc=f"RQ2 overlap {args.split}",
    )
    out = {"rq": "RQ2", "name": "overlap_metrics", "split": args.split, **ex}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    out_dir = os.path.join(cfg.OUTPUT_DIR, "logs", "experiments")
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"rq2_overlap_{args.split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved:", path)

    fig_dir = figure_dir(cfg, "rq2")
    pdf_bar = os.path.join(fig_dir, f"overlap_metrics_{args.split}.pdf")
    save_bar_pdf(
        pdf_bar,
        labels=["MIoU", "MSP", "MSR"],
        values=[ex["MIoU"], ex["MSP"], ex["MSR"]],
        title=f"RQ2 overlap metrics ({args.split})",
        ylabel="Score",
    )
    tbl = os.path.join(table_dir(cfg), f"rq2_overlap_{args.split}.pdf")
    save_table_pdf(
        tbl,
        headers=["Metric", "Value"],
        rows=[
            ["MIoU", f"{ex['MIoU']:.4f}"],
            ["MSP", f"{ex['MSP']:.4f}"],
            ["MSR", f"{ex['MSR']:.4f}"],
        ],
        title=f"RQ2 overlap ({args.split})",
    )
    print("PDF:", pdf_bar, tbl)


if __name__ == "__main__":
    main()
