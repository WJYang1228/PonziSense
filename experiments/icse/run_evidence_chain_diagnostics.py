from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.icse.icse_common import (
    ensure_dir,
    get_device,
    load_cfg,
    load_model_tokenizer,
    make_dataset_from_csv,
    make_loader,
    save_json,
    set_reproducible_seed,
)
from experiments.icse.run_mechanism_role_coverage import CORE_ROLE_SET, classify_roles, prf
from utils.rationale_extractor import node_perturbation_rationales


TRANSFER_PATTERNS = (".transfer", ".send", ".call", "selfdestruct")


@torch.inference_mode()
def ponzi_prob_with_graph(model, tokenizer, batch, bi: int, device, graph_adj=None) -> float:
    graph_adj = graph_adj if graph_adj is not None else batch["graph_adj"][bi : bi + 1].to(device)
    probs, _, _ = model.forward_contract(
        batch["input_ids"][bi : bi + 1].to(device),
        batch["position_idx"][bi : bi + 1].to(device),
        batch["attn_mask"][bi : bi + 1].to(device),
        labels=None,
        graph_adj=graph_adj,
        graph_mask=batch["graph_mask"][bi : bi + 1].to(device),
        statements=[batch["statements"][bi]],
        codes=[batch["codes"][bi]],
        tokenizer=tokenizer,
    )
    return float(probs[0, 1].detach().float().cpu().item())


def selected_indices_from_items(items: list[dict], metas, k: int | None) -> list[int]:
    stmt_to_idx = {int(meta.stmt_id): idx for idx, meta in enumerate(metas)}
    selected = [stmt_to_idx[int(item["stmt_id"])] for item in items if int(item["stmt_id"]) in stmt_to_idx]
    return selected if k is None else selected[: min(k, len(selected))]


def perturb_remove_nodes(graph_adj: torch.Tensor, selected: list[int], n_eff: int, mode: str) -> torch.Tensor:
    pert = graph_adj.clone()
    for idx in selected:
        if idx < 0 or idx >= n_eff:
            continue
        row = pert[0, idx, :n_eff]
        col = pert[0, :n_eff, idx]
        neighbors = ((row > 0) | (col > 0)).nonzero(as_tuple=False).flatten().tolist()
        pert[0, idx, :n_eff] = 0.0
        pert[0, :n_eff, idx] = 0.0
        if mode == "edge_dampen":
            for nb in neighbors:
                if nb == idx:
                    continue
                pert[0, nb, :n_eff] *= 0.5
                pert[0, :n_eff, nb] *= 0.5
        elif mode != "node_isolate":
            raise ValueError(f"Unknown perturbation mode: {mode}")
    return pert


def keep_only_selected_graph(graph_adj: torch.Tensor, selected: list[int], n_eff: int) -> torch.Tensor:
    kept = torch.zeros_like(graph_adj)
    selected_set = {idx for idx in selected if 0 <= idx < n_eff}
    if not selected_set:
        return kept
    ids = sorted(selected_set)
    for dst in ids:
        for src in ids:
            kept[0, dst, src] = graph_adj[0, dst, src]
        kept[0, dst, dst] = max(float(kept[0, dst, dst].item()), 1.0)
    return kept


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def degree_vector(graph_adj: torch.Tensor, n_eff: int) -> np.ndarray:
    g = graph_adj[0, :n_eff, :n_eff].detach().float().cpu().numpy()
    return (g > 0).sum(axis=0) + (g > 0).sum(axis=1)


def degree_matched_non_evidence(
    graph_adj: torch.Tensor,
    n_eff: int,
    forbidden: set[int],
    target_size: int,
    rng: random.Random,
) -> list[int]:
    if target_size <= 0:
        return []
    candidates = [idx for idx in range(n_eff) if idx not in forbidden]
    if not candidates:
        return []
    degrees = degree_vector(graph_adj, n_eff)
    target_degree = float(np.mean([degrees[i] for i in forbidden if i < n_eff])) if forbidden else float(np.mean(degrees))
    ranked = sorted(candidates, key=lambda idx: (abs(float(degrees[idx]) - target_degree), rng.random()))
    return ranked[: min(target_size, len(ranked))]


def random_indices(n_eff: int, forbidden: set[int], target_size: int, rng: random.Random) -> list[int]:
    pool = [idx for idx in range(n_eff) if idx not in forbidden]
    if not pool or target_size <= 0:
        return []
    return rng.sample(pool, min(target_size, len(pool)))


def local_transfer_indices(metas, max_nodes: int) -> list[int]:
    hits = []
    for idx, meta in enumerate(metas):
        low = str(meta.text).lower()
        if any(pat in low for pat in TRANSFER_PATTERNS):
            hits.append(idx)
    return hits[:max_nodes]


def role_metrics_for_indices(indices: list[int], gold_indices: set[int], metas) -> dict:
    pred_roles = set()
    gold_roles = set()
    for idx in indices:
        if 0 <= idx < len(metas):
            pred_roles.update(classify_roles(metas[idx].text))
    for idx in gold_indices:
        if 0 <= idx < len(metas):
            gold_roles.update(classify_roles(metas[idx].text))
    p, r, f = prf(pred_roles, gold_roles)
    _, core_r, _ = prf(pred_roles & CORE_ROLE_SET, gold_roles & CORE_ROLE_SET)
    return {"role_precision": p, "role_recall": r, "role_f1": f, "core_role_recall": core_r}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-chain necessity and sufficiency diagnostics for PonziSense.")
    parser.add_argument("--test-path", default="./datafiles/processed/test.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chain-nodes", type=int, default=8)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--mode", choices=["edge_dampen", "node_isolate"], default="edge_dampen")
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    rng = random.Random(args.seed)
    cfg = load_cfg(args)
    device = get_device(cfg, args.device)
    model, tokenizer = load_model_tokenizer(cfg, device, checkpoint=args.checkpoint)
    dataset = make_dataset_from_csv(args.test_path, tokenizer, cfg)
    loader = make_loader(dataset, cfg, device, shuffle=False)

    out_root = Path(args.output_dir) / "icse" / "evidence_chain_diagnostics"
    ensure_dir(out_root)

    necessity_rows = []
    sufficiency_rows = []
    sample_count = 0
    for batch_idx, batch in enumerate(loader):
        bs = int(batch["labels"].shape[0])
        for bi in range(bs):
            if args.max_samples and sample_count >= args.max_samples:
                break
            label = int(batch["labels"][bi].item())
            if label != 1:
                continue
            stmts = batch["statements"][bi]
            metas = batch["statement_meta"][bi]
            labs = batch["statement_labels"][bi].float().cpu().tolist()
            graph_adj = batch["graph_adj"][bi : bi + 1].to(device)
            graph_mask = batch["graph_mask"][bi : bi + 1].to(device)
            n_eff = min(len(stmts), len(metas), int(graph_mask[0].sum().item()), graph_adj.size(1))
            if n_eff <= 0:
                continue

            base_prob = ponzi_prob_with_graph(model, tokenizer, batch, bi, device, graph_adj=graph_adj)
            items = node_perturbation_rationales(
                model,
                tokenizer,
                batch["input_ids"][bi : bi + 1].to(device),
                batch["position_idx"][bi : bi + 1].to(device),
                batch["attn_mask"][bi : bi + 1].to(device),
                graph_adj,
                graph_mask,
                stmts,
                metas,
                batch["codes"][bi],
                base_ponzi_prob=base_prob,
                top_k=None,
            )
            if not items:
                continue

            gold_indices = {idx for idx, lab in enumerate(labs[:n_eff]) if lab >= 0.5}
            pred_indices = selected_indices_from_items(items, metas, args.top_k)
            gold_chain = sorted(gold_indices)[: args.max_chain_nodes]
            local_chain = local_transfer_indices(metas[:n_eff], args.top_k)
            forbidden = set(pred_indices) | gold_indices
            degree_chain = degree_matched_non_evidence(graph_adj, n_eff, forbidden, len(pred_indices), rng)

            strategy_sets = {
                "annotated_mechanism_nodes": gold_chain,
                "ponzisense_top_nodes": pred_indices,
                "local_transfer_line_only": local_chain,
                "degree_matched_non_evidence": degree_chain,
            }

            for repeat in range(args.random_repeats):
                strategy_sets[f"random_same_size_{repeat}"] = random_indices(n_eff, forbidden, len(pred_indices), rng)

            for strategy, selected in strategy_sets.items():
                if not selected:
                    continue
                removed_graph = perturb_remove_nodes(graph_adj, selected, n_eff, args.mode)
                removed_prob = ponzi_prob_with_graph(model, tokenizer, batch, bi, device, graph_adj=removed_graph)
                necessity_rows.append(
                    {
                        "sample_id": sample_count,
                        "batch_idx": batch_idx,
                        "sample_idx": bi,
                        "strategy": strategy,
                        "selected_count": len(selected),
                        "base_ponzi_prob": base_prob,
                        "perturbed_ponzi_prob": removed_prob,
                        "confidence_drop": base_prob - removed_prob,
                        "label_flip": int((base_prob >= cfg.PRED_THRESHOLD) and (removed_prob < cfg.PRED_THRESHOLD)),
                        "miou_with_gold": jaccard(set(selected), gold_indices),
                        "mode": args.mode,
                        **role_metrics_for_indices(selected, gold_indices, metas),
                    }
                )

                kept_graph = keep_only_selected_graph(graph_adj, selected, n_eff)
                kept_prob = ponzi_prob_with_graph(model, tokenizer, batch, bi, device, graph_adj=kept_graph)
                sufficiency_rows.append(
                    {
                        "sample_id": sample_count,
                        "batch_idx": batch_idx,
                        "sample_idx": bi,
                        "strategy": strategy,
                        "selected_count": len(selected),
                        "base_ponzi_prob": base_prob,
                        "kept_ponzi_prob": kept_prob,
                        "score_retained": kept_prob / base_prob if base_prob > 1e-12 else float("nan"),
                        "label_kept": int(kept_prob >= cfg.PRED_THRESHOLD),
                        "miou_with_gold": jaccard(set(selected), gold_indices),
                        **role_metrics_for_indices(selected, gold_indices, metas),
                    }
                )

            sample_count += 1
        if args.max_samples and sample_count >= args.max_samples:
            break

    necessity_df = pd.DataFrame(necessity_rows)
    sufficiency_df = pd.DataFrame(sufficiency_rows)

    def summarize(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        agg = {}
        for col in value_cols:
            agg[f"mean_{col}"] = (col, "mean")
            agg[f"median_{col}"] = (col, "median")
        agg["n"] = ("sample_id", "count")
        return df.groupby("strategy", as_index=False).agg(**agg).sort_values("strategy")

    necessity_summary = summarize(
        necessity_df,
        ["confidence_drop", "label_flip", "miou_with_gold", "role_f1", "core_role_recall"],
    )
    sufficiency_summary = summarize(
        sufficiency_df,
        ["score_retained", "label_kept", "miou_with_gold", "role_f1", "core_role_recall"],
    )

    necessity_df.to_csv(out_root / "necessity_samples.csv", index=False)
    sufficiency_df.to_csv(out_root / "sufficiency_samples.csv", index=False)
    necessity_summary.to_csv(out_root / "necessity_summary.csv", index=False)
    sufficiency_summary.to_csv(out_root / "sufficiency_summary.csv", index=False)
    summary = {
        "n_positive_samples": int(sample_count),
        "top_k": args.top_k,
        "max_chain_nodes": args.max_chain_nodes,
        "random_repeats": args.random_repeats,
        "mode": args.mode,
        "necessity_summary": necessity_summary.to_dict(orient="records"),
        "sufficiency_summary": sufficiency_summary.to_dict(orient="records"),
        "note": "The interventions keep source text fixed and manipulate graph propagation, so the diagnostic tests model evidence use rather than legal necessity of fraud.",
    }
    save_json(summary, out_root / "evidence_chain_diagnostics_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
