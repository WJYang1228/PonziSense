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
from utils.rationale_extractor import node_perturbation_rationales


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


def selected_indices_from_items(items: list[dict], metas, k: int, strategy: str, rng: random.Random) -> list[int]:
    stmt_to_idx = {int(meta.stmt_id): idx for idx, meta in enumerate(metas)}
    ordered = [stmt_to_idx[int(item["stmt_id"])] for item in items if int(item["stmt_id"]) in stmt_to_idx]
    if not ordered:
        return []
    k = min(k, len(ordered))
    if strategy == "top":
        return ordered[:k]
    if strategy == "bottom":
        return ordered[-k:]
    if strategy == "random":
        return rng.sample(ordered, k)
    raise ValueError(f"Unknown strategy: {strategy}")


def perturb_graph_preserve_syntax(graph_adj: torch.Tensor, selected: list[int], n_eff: int, mode: str) -> torch.Tensor:
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
        elif mode == "node_isolate":
            pass
        else:
            raise ValueError(f"Unknown perturbation mode: {mode}")
    return pert


def mean_or_nan(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Perturb graph evidence while keeping source syntax unchanged.")
    parser.add_argument("--test-path", default="./datafiles/processed/test.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 8, 10])
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--mode", choices=["edge_dampen", "node_isolate"], default="edge_dampen")
    parser.add_argument("--include-negatives", action="store_true")
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    rng = random.Random(args.seed)
    cfg = load_cfg(args)
    device = get_device(cfg, args.device)
    model, tokenizer = load_model_tokenizer(cfg, device, checkpoint=args.checkpoint)
    dataset = make_dataset_from_csv(args.test_path, tokenizer, cfg)
    loader = make_loader(dataset, cfg, device, shuffle=False)

    out_root = Path(args.output_dir) / "icse" / "syntax_preserving_faithfulness"
    ensure_dir(out_root)

    rows: list[dict] = []
    sample_count = 0
    for batch_idx, batch in enumerate(loader):
        bs = batch["labels"].shape[0]
        for bi in range(bs):
            label = int(batch["labels"][bi].item())
            if not args.include_negatives and label != 1:
                continue
            if args.max_samples and sample_count >= args.max_samples:
                break

            stmts = batch["statements"][bi]
            metas = batch["statement_meta"][bi]
            if not stmts or not metas:
                continue

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

            for k in args.k_values:
                for strategy in ["top", "bottom"]:
                    selected = selected_indices_from_items(items, metas, k, strategy, rng)
                    pert = perturb_graph_preserve_syntax(graph_adj, selected, n_eff, args.mode)
                    pert_prob = ponzi_prob_with_graph(model, tokenizer, batch, bi, device, graph_adj=pert)
                    rows.append(
                        {
                            "sample_id": sample_count,
                            "batch_idx": batch_idx,
                            "sample_idx": bi,
                            "label": label,
                            "k": k,
                            "strategy": strategy,
                            "repeat": 0,
                            "base_ponzi_prob": base_prob,
                            "perturbed_ponzi_prob": pert_prob,
                            "confidence_drop": base_prob - pert_prob,
                            "selected_count": len(selected),
                            "mode": args.mode,
                        }
                    )

                for repeat in range(args.random_repeats):
                    selected = selected_indices_from_items(items, metas, k, "random", rng)
                    pert = perturb_graph_preserve_syntax(graph_adj, selected, n_eff, args.mode)
                    pert_prob = ponzi_prob_with_graph(model, tokenizer, batch, bi, device, graph_adj=pert)
                    rows.append(
                        {
                            "sample_id": sample_count,
                            "batch_idx": batch_idx,
                            "sample_idx": bi,
                            "label": label,
                            "k": k,
                            "strategy": "random",
                            "repeat": repeat,
                            "base_ponzi_prob": base_prob,
                            "perturbed_ponzi_prob": pert_prob,
                            "confidence_drop": base_prob - pert_prob,
                            "selected_count": len(selected),
                            "mode": args.mode,
                        }
                    )
            sample_count += 1
        if args.max_samples and sample_count >= args.max_samples:
            break

    df = pd.DataFrame(rows)
    summary_rows = []
    for (strategy, k), sub in df.groupby(["strategy", "k"]):
        summary_rows.append(
            {
                "strategy": strategy,
                "k": int(k),
                "mean_confidence_drop": float(sub["confidence_drop"].mean()),
                "median_confidence_drop": float(sub["confidence_drop"].median()),
                "n": int(len(sub)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["strategy", "k"])

    aopc = {}
    for strategy, sub in summary_df.groupby("strategy"):
        ordered = sub.sort_values("k")
        aopc[strategy] = mean_or_nan(ordered["mean_confidence_drop"].tolist())

    df.to_csv(out_root / "syntax_preserving_faithfulness_samples.csv", index=False)
    summary_df.to_csv(out_root / "syntax_preserving_faithfulness_summary.csv", index=False)
    save_json(
        {
            "n_samples": int(sample_count),
            "mode": args.mode,
            "k_values": args.k_values,
            "random_repeats": args.random_repeats,
            "aopc": aopc,
        },
        out_root / "syntax_preserving_faithfulness_summary.json",
    )
    print(summary_df)
    print({"aopc": aopc, "n_samples": sample_count})


if __name__ == "__main__":
    main()
