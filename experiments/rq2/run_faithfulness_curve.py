#!/usr/bin/env python3
"""
RQ2：论文 Figure faithfulness curve — 随 Top-k 掩码语句数变化的平均置信度下降（FD 近似）。

用法:
  python experiments/rq2/run_faithfulness_curve.py [--max-samples 100] [--k-max 10]
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

from tqdm import tqdm

from experiments.common.project import ensure_repo_importable

ensure_repo_importable()

from experiments.common.infer_one import ponzi_probability  # noqa: E402
from experiments.common.load_model import load_trained_ponzimodel  # noqa: E402
from experiments.common.output_paths import figure_dir, table_dir  # noqa: E402
from experiments.common.plot_pdf import save_lines_pdf, save_table_pdf  # noqa: E402
from experiments.rq2.run_fd_arl import simplified_top_k_ids  # noqa: E402
from utils.mask_code import remove_statement_blocks_by_id  # noqa: E402

from configs.config import Config  # noqa: E402
from data.dataset import load_datasets  # noqa: E402
from utils.io import ensure_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--max-samples", type=int, default=80)
    ap.add_argument("--k-max", type=int, default=10)
    args = ap.parse_args()

    cfg = Config()
    model, tokenizer, _, device = load_trained_ponzimodel()
    _, val_set, test_set = load_datasets(tokenizer, cfg)
    ds = test_set if args.split == "test" else val_set
    n = min(args.max_samples, len(ds))

    ks = list(range(1, args.k_max + 1))
    curve = {k: [] for k in ks}

    for i in tqdm(range(n), desc="faithfulness-curve"):
        row = ds.df.iloc[i]
        code = str(row["code"])
        p_orig = ponzi_probability(model, tokenizer, code, cfg, device)
        for k in ks:
            pred_ids, _, _ = simplified_top_k_ids(model, tokenizer, code, cfg, device, k)
            if not pred_ids:
                curve[k].append(0.0)
                continue
            masked = remove_statement_blocks_by_id(code, pred_ids)
            if not masked.strip():
                p_mask = p_orig
            else:
                p_mask = ponzi_probability(model, tokenizer, masked, cfg, device)
            curve[k].append(p_orig - p_mask)

    means = [sum(curve[k]) / max(1, len(curve[k])) for k in ks]
    out = {"rq": "RQ2", "name": "faithfulness_curve", "split": args.split, "k": ks, "mean_fd": means}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    out_dir = os.path.join(cfg.OUTPUT_DIR, "logs", "experiments")
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"rq2_faithfulness_curve_{args.split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved:", path)

    pdf = os.path.join(figure_dir(cfg, "rq2"), f"faithfulness_curve_{args.split}.pdf")
    save_lines_pdf(
        pdf,
        xs=ks,
        series={"Mean FD": means},
        title=f"RQ2 faithfulness vs Top-k ({args.split})",
        xlabel="Top-k",
        ylabel="Mean confidence drop",
    )
    rows = [[str(k), f"{m:.4f}"] for k, m in zip(ks, means)]
    tbl = os.path.join(table_dir(cfg), f"rq2_faithfulness_curve_{args.split}.pdf")
    save_table_pdf(tbl, headers=["k", "Mean FD"], rows=rows, title=f"RQ2 faithfulness curve ({args.split})")
    print("PDF:", pdf, tbl)


if __name__ == "__main__":
    main()
