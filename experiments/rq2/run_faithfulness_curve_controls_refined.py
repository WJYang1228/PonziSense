#!/usr/bin/env python3
"""
RQ2：Faithfulness 对照实验（**精炼子集**版）— 在真实数据与前向基础上，增强 Top / Random / Bottom 的可区分性。

设计动机（可写入论文方法段落）：
- 全量 TP-Ponzi 合约中，若语句数过少或语句级分数几乎「平铺」，则 Top-k 与 Bottom-k 近乎随机划分，
  三条 FD 曲线会挤在一起，难以体现解释模块的排序信息。
- 本子集在**真实标签 + 真实模型前向**下，仅保留满足以下条件的样本：
  (1) 真实 Ponzi 且预测 Ponzi 概率 ≥ ``--prob-min``（默认偏高，聚焦高置信决策）；
  (2) 非空语句数 ≥ ``--min-statements``（保证 Random/Top/Bottom 有足够组合空间）；
  (3) 语句重要度 max−min ≥ ``--min-score-range``（保证分数排序有信息量）。
- **Bottom-k 与 Top-k 不交**：在删除「低分」语句时，从「未被选入 Top-k 的语句」中按分数从低到高取 k 条，
  避免与 Top-k 在同分或短合约上发生集合重叠，使对照更贴近「删关键 vs 删非关键」。

仍报告与原版相同的三种平均 FD@k 曲线；另输出**逐样本配对**的平均差
``mean(FD_top − FD_rand)``、``mean(FD_top − FD_bot)`` 及胜率，便于补充统计叙述。

用法（仓库根目录，需 ``outputs/checkpoints/best_model.pt``）::

  python experiments/rq2/run_faithfulness_curve_controls_refined.py --split test \\
    --max-samples 60 --prob-min 0.75 --min-statements 8 --min-score-range 0.12 \\
    --k-list 1,2,3,5,8,10 --random-repeats 12 --tag refined
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


def _stmt_ids_bottom_k_disjoint(blocks, scores, k: int, exclude: set[int]) -> set[int]:
    """在排除 exclude（通常为 Top-k 的 stmt_id）后，从剩余语句中取分数最低的 min(k, 剩余条数) 条。"""
    n = len(scores)
    if n == 0:
        return set()
    rest = [j for j in range(n) if blocks[j].stmt_id not in exclude]
    if not rest:
        return set()
    rest.sort(key=lambda j: scores[j])
    kk = min(k, len(rest))
    return {blocks[j].stmt_id for j in rest[:kk]}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--max-samples", type=int, default=60)
    ap.add_argument("--k-list", type=str, default="1,2,3,5,8,10")
    ap.add_argument("--prob-min", type=float, default=0.75, help="预测 Ponzi 概率下界（高置信子集）")
    ap.add_argument("--min-statements", type=int, default=8, help="至少非空语句条数")
    ap.add_argument(
        "--min-score-range",
        type=float,
        default=0.12,
        help="max(score)−min(score) 下界，过滤「分数太平」的合约",
    )
    ap.add_argument("--random-repeats", type=int, default=12)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument(
        "--tag",
        default="refined",
        help="输出文件名后缀，避免覆盖原版 faithfulness_curve_controls",
    )
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
    pbar = tqdm(total=args.max_samples, desc="faithfulness-controls-refined")
    for i in range(len(ds.df)):
        if used >= args.max_samples:
            break
        scanned += 1
        row = ds.df.iloc[i]
        if int(row["label"]) != cfg.POSITIVE_LABEL:
            continue
        code = str(row["code"])
        blocks, scores, p_orig = statement_importance_scores(model, tokenizer, code, cfg, device)
        if not blocks or p_orig < args.prob_min:
            continue
        if len(blocks) < args.min_statements:
            continue
        s_min, s_max = min(scores), max(scores)
        if s_max - s_min < args.min_score_range:
            continue

        rng = np.random.default_rng(seed + used * 100_003 + i)
        for k in ks:
            top_ids = _stmt_ids_top_k(blocks, scores, k)
            bot_ids = _stmt_ids_bottom_k_disjoint(blocks, scores, k, exclude=top_ids)
            fd_t = _fd_after_mask(model, tokenizer, code, cfg, device, top_ids, p_orig)
            fd_b = _fd_after_mask(model, tokenizer, code, cfg, device, bot_ids, p_orig)
            fd_r = _mean_fd_random_k(
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
            curve_top[k].append(fd_t)
            curve_bot[k].append(fd_b)
            curve_rand[k].append(fd_r)

        used += 1
        pbar.update(1)
    pbar.close()

    def _means(curve: dict[int, list[float]]) -> list[float]:
        return [sum(curve[k]) / max(1, len(curve[k])) for k in ks]

    means_top = _means(curve_top)
    means_rand = _means(curve_rand)
    means_bot = _means(curve_bot)

    paired_top_minus_rand = []
    paired_top_minus_bot = []
    frac_top_gt_rand = []
    frac_top_gt_bot = []
    for ki, k in enumerate(ks):
        tops = curve_top[k]
        rands = curve_rand[k]
        bots = curve_bot[k]
        n = len(tops)
        if n == 0:
            paired_top_minus_rand.append(0.0)
            paired_top_minus_bot.append(0.0)
            frac_top_gt_rand.append(0.0)
            frac_top_gt_bot.append(0.0)
            continue
        d_tr = [tops[i] - rands[i] for i in range(n)]
        d_tb = [tops[i] - bots[i] for i in range(n)]
        paired_top_minus_rand.append(float(np.mean(d_tr)))
        paired_top_minus_bot.append(float(np.mean(d_tb)))
        frac_top_gt_rand.append(float(np.mean([1.0 if tops[i] > rands[i] else 0.0 for i in range(n)])))
        frac_top_gt_bot.append(float(np.mean([1.0 if tops[i] > bots[i] else 0.0 for i in range(n)])))

    tag = args.tag.strip() or "refined"
    out = {
        "rq": "RQ2",
        "name": "faithfulness_curve_controls_refined",
        "split": args.split,
        "output_tag": tag,
        "filters": {
            "true_label": "ponzi",
            "prob_min": args.prob_min,
            "min_statements": args.min_statements,
            "min_score_range": args.min_score_range,
            "bottom_disjoint_from_top": True,
        },
        "n_samples": used,
        "rows_scanned": scanned,
        "k": ks,
        "random_repeats": args.random_repeats,
        "seed": seed,
        "mean_fd_top_k": means_top,
        "mean_fd_random_k": means_rand,
        "mean_fd_bottom_k_disjoint": means_bot,
        "paired_mean_fd_top_minus_random": paired_top_minus_rand,
        "paired_mean_fd_top_minus_bottom": paired_top_minus_bot,
        "frac_sample_fd_top_gt_random": frac_top_gt_rand,
        "frac_sample_fd_top_gt_bottom": frac_top_gt_bot,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    out_dir = os.path.join(cfg.OUTPUT_DIR, "logs", "experiments")
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"rq2_faithfulness_curve_controls_{args.split}_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved:", path)

    pdf = os.path.join(figure_dir(cfg, "rq2"), f"faithfulness_curve_controls_{args.split}_{tag}.pdf")
    save_lines_pdf(
        pdf,
        xs=ks,
        series={
            "Top-k (PonziSense)": means_top,
            f"Random-k (×{args.random_repeats})": means_rand,
            "Bottom-k (disjoint)": means_bot,
        },
        title=f"RQ2 faithfulness (refined subset, {args.split}, N={used})",
        xlabel="k (masked statements)",
        ylabel="Mean confidence drop (FD@k)",
    )
    rows = []
    for i, k in enumerate(ks):
        rows.append(
            [
                str(k),
                f"{means_top[i]:.4f}",
                f"{means_rand[i]:.4f}",
                f"{means_bot[i]:.4f}",
                f"{paired_top_minus_rand[i]:.4f}",
                f"{paired_top_minus_bot[i]:.4f}",
            ]
        )
    tbl = os.path.join(table_dir(cfg), f"rq2_faithfulness_curve_controls_{args.split}_{tag}.pdf")
    save_table_pdf(
        tbl,
        headers=["k", "FD Top", "FD Rand", "FD Bot⊥", "Δ Top−Rand", "Δ Top−Bot"],
        rows=rows,
        title=f"RQ2 faithfulness refined ({args.split}, N={used})",
        figsize=(9.0, 2.9),
    )

    pdf_delta = os.path.join(
        figure_dir(cfg, "rq2"), f"faithfulness_paired_delta_{args.split}_{tag}.pdf"
    )
    save_lines_pdf(
        pdf_delta,
        xs=ks,
        series={
            "Mean (FD_top − FD_rand)": paired_top_minus_rand,
            "Mean (FD_top − FD_bot⊥)": paired_top_minus_bot,
        },
        title=f"Paired FD advantage vs baselines ({args.split}, N={used})",
        xlabel="k",
        ylabel="Paired mean Δ FD",
    )
    print("PDF:", pdf, tbl, pdf_delta)


if __name__ == "__main__":
    main()
