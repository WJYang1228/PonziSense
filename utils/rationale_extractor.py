"""Node-aware rationale extraction for PonziSense inference and evaluation."""
from __future__ import annotations

from typing import Sequence

import torch

from utils.statements import StatementBlock, is_trivial_statement


@torch.inference_mode()
def _ponzi_probability_from_graph(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    position_idx: torch.Tensor,
    attn_mask: torch.Tensor,
    graph_adj: torch.Tensor,
    graph_mask: torch.Tensor,
    statements: Sequence[str],
    code: str,
) -> float:
    probs, _, _ = model.forward_contract(
        input_ids,
        position_idx,
        attn_mask,
        labels=None,
        graph_adj=graph_adj,
        graph_mask=graph_mask,
        statements=[list(statements)],
        codes=[code],
        tokenizer=tokenizer,
    )
    return float(probs[0, 1].detach().float().cpu().item())


@torch.inference_mode()
def node_perturbation_rationales(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    position_idx: torch.Tensor,
    attn_mask: torch.Tensor,
    graph_adj: torch.Tensor,
    graph_mask: torch.Tensor,
    statements: Sequence[str],
    blocks: Sequence[StatementBlock],
    code: str,
    *,
    base_ponzi_prob: float | None = None,
    top_k: int | None = 5,
) -> list[dict]:
    """
    Rank source statements by the paper's node perturbation score.

    For node v_i, suppress incoming edges, outgoing edges, and one-hop
    neighborhood propagation around v_i. The importance score is
    r_i=max(0, p - p_i^-), where p is the original Ponzi confidence.
    """
    n_eff = min(len(statements), len(blocks), int(graph_mask[0].sum().item()), graph_adj.size(1))
    if n_eff <= 0:
        return []

    base = base_ponzi_prob
    if base is None:
        base = _ponzi_probability_from_graph(
            model, tokenizer, input_ids, position_idx, attn_mask, graph_adj, graph_mask, statements, code
        )

    items: list[dict] = []
    for idx in range(n_eff):
        block = blocks[idx]
        text = block.text
        if is_trivial_statement(text):
            continue

        pert_adj = graph_adj.clone()
        row = pert_adj[0, idx, :n_eff]
        col = pert_adj[0, :n_eff, idx]
        neighbors = ((row > 0) | (col > 0)).nonzero(as_tuple=False).flatten().tolist()

        # Incoming to idx and outgoing from idx are removed.
        pert_adj[0, idx, :n_eff] = 0.0
        pert_adj[0, :n_eff, idx] = 0.0

        # One-hop propagation through the local neighborhood is dampened.
        for nb in neighbors:
            if nb == idx:
                continue
            pert_adj[0, nb, :n_eff] *= 0.5
            pert_adj[0, :n_eff, nb] *= 0.5

        perturbed = _ponzi_probability_from_graph(
            model, tokenizer, input_ids, position_idx, attn_mask, pert_adj, graph_mask, statements, code
        )
        score = max(0.0, float(base) - float(perturbed))
        items.append(
            {
                "stmt_id": block.stmt_id,
                "start_line": block.start_line,
                "end_line": block.end_line,
                "text": text,
                "score": score,
                "confidence_drop": score,
                "perturbed_ponzi_prob": float(perturbed),
            }
        )

    items.sort(key=lambda x: (-x["score"], x["start_line"], x["stmt_id"]))
    return items if top_k is None else items[:top_k]
