"""单合约 Ponzi 概率（与 predict.py 逻辑一致，供实验脚本复用）。"""
from __future__ import annotations

import torch

from data.feature_extractor import convert_code_to_features, build_attention_mask
from graph.statement_graph import build_statement_graph_tensors
from utils.statements import build_statement_labels


@torch.inference_mode()
def ponzi_probability(model, tokenizer, code: str, cfg, device) -> float:
    feature = convert_code_to_features(
        code=code,
        label=0,
        explain="",
        tokenizer=tokenizer,
        cfg=cfg,
    )
    attn_mask = build_attention_mask(feature, cfg, tokenizer)
    input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0).to(device)
    position_idx = torch.tensor(feature.position_idx, dtype=torch.long).unsqueeze(0).to(device)
    attn_mask_t = torch.tensor(attn_mask, dtype=torch.bool).unsqueeze(0).to(device)
    stmts, _, _ = build_statement_labels(code, "")
    ga, gm, _ = build_statement_graph_tensors(code, cfg)
    graph_adj = torch.tensor(ga, dtype=torch.float32, device=device).unsqueeze(0)
    graph_mask = torch.tensor(gm, dtype=torch.float32, device=device).unsqueeze(0)
    probs, _, _ = model.forward_contract(
        input_ids,
        position_idx,
        attn_mask_t,
        labels=None,
        graph_adj=graph_adj,
        graph_mask=graph_mask,
        statements=[stmts],
        codes=[code],
        tokenizer=tokenizer,
    )
    return float(probs[0, 1].item())
