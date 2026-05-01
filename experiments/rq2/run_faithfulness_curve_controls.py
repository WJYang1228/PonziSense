#!/usr/bin/env python3
"""
RQ2：Faithfulness 对照实验 — Top-k / Random-k / Bottom-k 掩码下的 FD 曲线。

仅统计 **测试集中真实为 Ponzi 且模型正确预测为 Ponzi** 的样本（可按阈值判定预测）。

FD@k ≈ (1/N) Σ_i [ p_i - p̃_i(k) ]，p̃ 为遮蔽后 Ponzi 概率；Random-k 对多次随机采样取平均后再对样本平均。

用法（仓库根目录）::

  python experiments/rq2/run_faithfulness_curve_controls.py --split test \\
    --max-samples 80 --k-list 1,2,3,5,8,10 --random-repeats 8
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

import numpy as np
from tqdm import tqdm

from experiments.common.project import ensure_repo_importable

ensure_repo_importable()

from experiments.common.infer_one import ponzi_probability  # noqa: E402
from experiments.common.load_model import load_trained_ponzimodel  # noqa: E402
from experiments.common.output_paths import figure_dir, table_dir  # noqa: E402
from experiments.common.plot_pdf import save_lines_pdf, save_table_pdf  # noqa: E402
from experiments.rq2.run_fd_arl import statement_importance_scores  # noqa: E402
from utils.mask_code import remove_statement_blocks_by_id  # noqa: E402

from configs.config import Config  # noqa: E402
from data.dataset import load_datasets  # noqa: E402
from utils.io import ensure_dir  # noqa: E402


def _fd_after_mask(
    model, tokenizer, code: str, cfg, device, stmt_ids: set[int], p_orig: float
) -> float:
    if not stmt_ids:
        return 0.0
    masked = remove_statement_blocks_by_id(code, stmt_ids)
    if not masked.strip():
        p_mask = p_orig
    else:
        p_mask = ponzi_probability(model, tokenizer, masked, cfg, device)
    return p_orig - p_mask


def _stmt_ids_top_k(blocks, scores, k: int) -> set[int]:
    n = len(scores)
    if n == 0:
        return set()
    kk = min(k, n)
    order = sorted(range(n), key=lambda j: scores[j], reverse=True)[:kk]
    return {blocks[j].stmt_id for j in order}


def _stmt_ids_bottom_k(blocks, scores, k: int) -> set[int]:
    n = len(scores)
    if n == 0:
        return set()
    kk = min(k, n)
    order = sorted(range(n), key=lambda j: scores[j])[:kk]
    return {blocks[j].stmt_id for j in order}


def _mean_fd_random_k(
    model,
    tokenizer,
    code: str,
    cfg,
    device,
    blocks,
    scores,
    p_orig: float,
    k: int,
    rng: np.random.Generator,
    n_repeat: int,
) -> float:
    n = len(scores)
    if n == 0:
        return 0.0
    kk = min(k, n)
    fds = []
    idx_all = np.arange(n)
    for _ in range(n_repeat):
        chosen = rng.choice(idx_all, size=kk, replace=False)
        stmt_ids = {blocks[int(j)].stmt_id for j in chosen}
        fds.append(_fd_after_mask(model, tokenizer, code, cfg, device, stmt_ids, p_orig))
    return float(np.mean(fds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=80,
        help="最多纳入的真实 Ponzi 且预测正确的样本数（在数据集中顺序扫描直至凑满）",
    )
    ap.add_argument(
        "--k-list",
        type=str,
        default="1,2,3,5,8,10",
        help="逗号分隔的 k 列表",
    )
    ap.add_argument("--threshold", type=float, default=0.5, help="判定预测为 Ponzi 的概率阈值")
    ap.add_argument("--random-repeats", type=int, default=8, help="Random-k 每条样本重复采样次数")
    ap.add_argument("--seed", type=int, default=-1, help="随机种子，默认使用 cfg.RANDOM_SEED")
    args = ap.parse_args()

    ks = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    if not ks or any(k < 1 for k in ks):
        raise SystemExit("k-list 须为非空正整数列表")

    cfg = Config()
    seed = cfg.RANDOM_SEED if args.seed < 0 else args.seed
    model, tokenizer, _, device = load_trained_ponzimodel()
    _, val_set, test_set = load_datasets(tokenizer, cfg)
    ds = test_set if args.split == "test" else val_set

    curve_top: dict[int, list[float]] = {k: [] for k in ks}
    curve_rand: dict[int, list[float]] = {k: [] for k in ks}
    curve_bot: dict[int, list[float]] = {k: [] for k in ks}

    used = 0
    scanned = 0
    pbar = tqdm(total=args.max_samples, desc="faithfulness-controls (TP-Ponzi)")
    for i in range(len(ds.df)):
        if used >= args.max_samples:
            break
        scanned += 1
        row = ds.df.iloc[i]
        if int(row["label"]) != cfg.POSITIVE_LABEL:
            continue
        code = str(row["code"])
        blocks, scores, p_orig = statement_importance_scores(model, tokenizer, code, cfg, device)
        if p_orig < args.threshold or not blocks:
            continue

        rng = np.random.default_rng(seed + used * 100_003 + i)
        for k in ks:
            top_ids = _stmt_ids_top_k(blocks, scores, k)
            bot_ids = _stmt_ids_bottom_k(blocks, scores, k)
            curve_top[k].append(_fd_after_mask(model, tokenizer, code, cfg, device, top_ids, p_orig))
            curve_bot[k].append(_fd_after_mask(model, tokenizer, code, cfg, device, bot_ids, p_orig))
            curve_rand[k].append(
                _mean_fd_random_k(
                    model,
                    tokenizer,
                    code,
                    cfg,
                    device,
                    blocks,
                    scores,
                    p_orig,
                    k,
                    rng,
                    args.random_repeats,
                )
            )

        used += 1
        pbar.update(1)
    pbar.close()

    def _means(curve: dict[int, list[float]]) -> list[float]:
        return [sum(curve[k]) / max(1, len(curve[k])) for k in ks]

    means_top = _means(curve_top)
    means_rand = _means(curve_rand)
    means_bot = _means(curve_bot)

    out = {
        "rq": "RQ2",
        "name": "faithfulness_curve_controls",
        "split": args.split,
        "filter": "true_ponzi_and_pred_ponzi",
        "threshold": args.threshold,
        "n_samples": used,
        "rows_scanned": scanned,
        "k": ks,
        "random_repeats": args.random_repeats,
        "seed": seed,
        "mean_fd_top_k": means_top,
        "mean_fd_random_k": means_rand,
        "mean_fd_bottom_k": means_bot,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    out_dir = os.path.join(cfg.OUTPUT_DIR, "logs", "experiments")
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"rq2_faithfulness_curve_controls_{args.split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved:", path)

    pdf = os.path.join(figure_dir(cfg, "rq2"), f"faithfulness_curve_controls_{args.split}.pdf")
    save_lines_pdf(
        pdf,
        xs=ks,
        series={
            "Top-k (PonziSense)": means_top,
            f"Random-k (×{args.random_repeats})": means_rand,
            "Bottom-k (low score)": means_bot,
        },
        title=f"RQ2 faithfulness: Top vs Random vs Bottom-k ({args.split}, N={used})",
        xlabel="k (masked statements)",
        ylabel="Mean confidence drop (FD@k)",
    )
    rows = []
    for i, k in enumerate(ks):
        rows.append([str(k), f"{means_top[i]:.4f}", f"{means_rand[i]:.4f}", f"{means_bot[i]:.4f}"])
    tbl = os.path.join(table_dir(cfg), f"rq2_faithfulness_curve_controls_{args.split}.pdf")
    save_table_pdf(
        tbl,
        headers=["k", "FD Top-k", "FD Random-k", "FD Bottom-k"],
        rows=rows,
        title=f"RQ2 faithfulness controls ({args.split}, TP-Ponzi N={used})",
        figsize=(7.5, 2.8),
    )
    print("PDF:", pdf, tbl)


if __name__ == "__main__":
    main()
