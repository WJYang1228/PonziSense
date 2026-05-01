#!/usr/bin/env python3
"""
RQ2: Faithfulness Drop (FD) 与 Average Rationale Length (ARL)。

FD ≈ (1/N) Σ [ f(x_i) - f(x_i \\ S_e) ]，S_e 为模型 Top-K 语句集合；
f 为 Ponzi 概率；x_i \\ S_e 由 remove_statement_blocks_by_id 近似。

ARL = (1/N) Σ |S_e|，此处 |S_e| = min(K, 非空语句数)。

用法:
  python experiments/rq2/run_fd_arl.py [--max-samples 200] [--split test]
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
from tqdm import tqdm

from experiments.common.project import ensure_repo_importable

ensure_repo_importable()

from experiments.common.infer_one import ponzi_probability
from experiments.common.load_model import load_trained_ponzimodel
from experiments.common.mask_rationale import remove_statement_blocks_by_id

from configs.config import Config  # noqa: E402
from data.collate import collate_fn  # noqa: E402
from data.dataset import load_datasets  # noqa: E402
from data.feature_extractor import build_attention_mask, convert_code_to_features  # noqa: E402
from graph.statement_graph import build_statement_graph_tensors  # noqa: E402
from utils.io import ensure_dir  # noqa: E402
from utils.statements import build_statement_labels  # noqa: E402
from experiments.common.output_paths import figure_dir, table_dir  # noqa: E402
from experiments.common.plot_pdf import save_bar_pdf, save_table_pdf  # noqa: E402


@torch.no_grad()
def statement_importance_scores(model, tokenizer, code, cfg, device):
    """
    返回与 ``build_statement_labels(..., \"\")`` 过滤后语句一一对应的 PonziSense 语句重要度分数，
    以及原始合约 Ponzi 概率 p_orig。
    """
    statements, _, blocks = build_statement_labels(code, explain="")
    if not statements:
        return [], [], 0.0
    feature = convert_code_to_features(code, 0, "", tokenizer, cfg)
    attn_mask = build_attention_mask(feature, cfg, tokenizer)
    input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0).to(device)
    position_idx = torch.tensor(feature.position_idx, dtype=torch.long).unsqueeze(0).to(device)
    attn_mask_t = torch.tensor(attn_mask, dtype=torch.bool).unsqueeze(0).to(device)
    use_amp = cfg.USE_AMP and device.type == "cuda"
    stmts0, _, _ = build_statement_labels(code, "")
    ga, gm, _ = build_statement_graph_tensors(code, cfg)
    graph_adj = torch.tensor(ga, dtype=torch.float32, device=device).unsqueeze(0)
    graph_mask = torch.tensor(gm, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.autocast(device_type="cuda", enabled=use_amp):
        _, _, outputs = model.forward_contract(
            input_ids,
            position_idx,
            attn_mask_t,
            labels=None,
            graph_adj=graph_adj,
            graph_mask=graph_mask,
            statements=[stmts0],
            codes=[code],
            tokenizer=tokenizer,
        )
        contract_cls = outputs[:, 0, :]
        n = len(statements)
        cexp = contract_cls.expand(n, -1)
        _, stmt_probs = model.forward_statements(
            statements,
            tokenizer=tokenizer,
            device=device,
            max_len=cfg.EXPLAIN_STMT_MAX_LEN,
            contract_cls_emb=cexp,
        )
    scores = stmt_probs.float().cpu().tolist()
    p_orig = ponzi_probability(model, tokenizer, code, cfg, device)
    return blocks, scores, p_orig


@torch.no_grad()
def simplified_top_k_ids(model, tokenizer, code, cfg, device, k):
    """返回 top-k stmt_id 集合、p_orig、实际 K。"""
    blocks, scores, p_orig = statement_importance_scores(model, tokenizer, code, cfg, device)
    if not blocks:
        return set(), p_orig, 0
    kk = min(k, len(scores))
    order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:kk]
    pred_ids = {blocks[j].stmt_id for j in order}
    return pred_ids, p_orig, kk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--max-samples", type=int, default=0, help="0 表示全量")
    args = ap.parse_args()

    cfg = Config()
    k = cfg.EXPLAIN_EVAL_TOP_K
    model, tokenizer, _, device = load_trained_ponzimodel()
    _, val_set, test_set = load_datasets(tokenizer, cfg)
    ds = test_set if args.split == "test" else val_set
    n = len(ds) if args.max_samples <= 0 else min(args.max_samples, len(ds))

    fds = []
    arls = []
    for i in tqdm(range(n), desc="FD/ARL"):
        row = ds.df.iloc[i]
        code = str(row["code"])
        pred_ids, p_orig, k_used = simplified_top_k_ids(model, tokenizer, code, cfg, device, k)
        arls.append(k_used)
        if not pred_ids:
            fds.append(0.0)
            continue
        masked = remove_statement_blocks_by_id(code, pred_ids)
        if not masked.strip():
            p_mask = p_orig
        else:
            p_mask = ponzi_probability(model, tokenizer, masked, cfg, device)
        fds.append(p_orig - p_mask)

    out = {
        "rq": "RQ2",
        "name": "fd_arl",
        "split": args.split,
        "top_k": k,
        "n": n,
        "FD": sum(fds) / max(1, len(fds)),
        "ARL": sum(arls) / max(1, len(arls)),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    out_dir = os.path.join(cfg.OUTPUT_DIR, "logs", "experiments")
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"rq2_fd_arl_{args.split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved:", path)

    fd_mean = out["FD"]
    arl_mean = out["ARL"]
    pdf_bar = os.path.join(figure_dir(cfg, "rq2"), f"fd_arl_{args.split}.pdf")
    save_bar_pdf(
        pdf_bar,
        labels=["FD", "ARL"],
        values=[fd_mean, arl_mean],
        title=f"RQ2 FD & ARL (split={args.split}, top_k={k})",
        ylabel="Value",
    )
    tbl = os.path.join(table_dir(cfg), f"rq2_fd_arl_{args.split}.pdf")
    save_table_pdf(
        tbl,
        headers=["Metric", "Value"],
        rows=[["FD", f"{fd_mean:.4f}"], ["ARL", f"{arl_mean:.4f}"]],
        title=f"RQ2 FD / ARL ({args.split})",
        figsize=(6, 2.5),
    )
    print("PDF:", pdf_bar, tbl)


if __name__ == "__main__":
    main()
