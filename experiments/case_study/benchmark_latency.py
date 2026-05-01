#!/usr/bin/env python3
"""
案例/效率：单合约「分类前向」与「语句解释前向」耗时（与论文 inference / explanation 分项对应）。

用法:
  python experiments/case_study/benchmark_latency.py [--n 50]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from experiments.common.project import ensure_repo_importable

ensure_repo_importable()

from experiments.common.load_model import load_trained_ponzimodel

from configs.config import Config  # noqa: E402
from data.dataset import load_datasets  # noqa: E402
from data.feature_extractor import build_attention_mask, convert_code_to_features  # noqa: E402
from graph.statement_graph import build_statement_graph_tensors  # noqa: E402
from utils.io import ensure_dir  # noqa: E402
from utils.statements import build_statement_labels  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    cfg = Config()
    model, tokenizer, _, device = load_trained_ponzimodel()
    _, _, test_set = load_datasets(tokenizer, cfg)
    n = min(args.n, len(test_set))

    use_amp = cfg.USE_AMP and device.type == "cuda"
    autocast = torch.autocast(device_type="cuda", enabled=use_amp)

    t_cls = []
    t_exp = []
    for i in range(n):
        code = str(test_set.df.iloc[i]["code"])
        feature = convert_code_to_features(code, 0, "", tokenizer, cfg)
        attn_mask = build_attention_mask(feature, cfg, tokenizer)
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0).to(device)
        position_idx = torch.tensor(feature.position_idx, dtype=torch.long).unsqueeze(0).to(device)
        attn_mask_t = torch.tensor(attn_mask, dtype=torch.bool).unsqueeze(0).to(device)

        statements, _, _ = build_statement_labels(code, explain="")
        ga, gm, _ = build_statement_graph_tensors(code, cfg)
        graph_adj = torch.tensor(ga, dtype=torch.float32, device=device).unsqueeze(0)
        graph_mask = torch.tensor(gm, dtype=torch.float32, device=device).unsqueeze(0)

        t0 = time.perf_counter()
        with autocast:
            _, _, outputs = model.forward_contract(
                input_ids,
                position_idx,
                attn_mask_t,
                labels=None,
                graph_adj=graph_adj,
                graph_mask=graph_mask,
                statements=[statements],
                codes=[code],
                tokenizer=tokenizer,
            )
        t1 = time.perf_counter()
        t_cls.append(t1 - t0)

        if not statements:
            t_exp.append(0.0)
            continue
        contract_cls = outputs[:, 0, :]
        m = len(statements)
        cexp = contract_cls.expand(m, -1)
        t2 = time.perf_counter()
        with autocast:
            _, _ = model.forward_statements(
                statements,
                tokenizer=tokenizer,
                device=device,
                max_len=cfg.EXPLAIN_STMT_MAX_LEN,
                contract_cls_emb=cexp,
            )
        t3 = time.perf_counter()
        t_exp.append(t3 - t2)

    def mean(xs):
        return sum(xs) / max(1, len(xs))

    out = {
        "n": n,
        "inference_time_ms": mean(t_cls) * 1000,
        "explanation_time_ms": mean(t_exp) * 1000,
        "total_ms": (mean(t_cls) + mean(t_exp)) * 1000,
    }
    print(json.dumps(out, indent=2))
    out_dir = os.path.join(cfg.OUTPUT_DIR, "logs", "experiments")
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "case_study_latency.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved:", path)


if __name__ == "__main__":
    main()
