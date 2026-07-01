from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data.dataset import read_one_csv
from experiments.icse.icse_common import (
    classification_from_scores,
    collect_scores,
    ensure_dir,
    get_device,
    load_cfg,
    load_model_tokenizer,
    make_dataset_from_frame,
    make_loader,
    save_json,
    set_reproducible_seed,
)
from experiments.icse.run_mechanism_role_coverage import CORE_ROLE_SET, classify_roles, prf
from utils.rationale_extractor import node_perturbation_rationales


VARIANTS = {
    "full": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "graph_branch": True},
    "cfg_only": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "graph_branch": True},
    "dfg_only": {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "graph_branch": True},
    "prop_only": {"alpha": 0.0, "beta": 0.0, "gamma": 1.0, "graph_branch": True},
    "no_cfg": {"alpha": 0.0, "beta": 1.0, "gamma": 1.0, "graph_branch": True},
    "no_dfg": {"alpha": 1.0, "beta": 0.0, "gamma": 1.0, "graph_branch": True},
    "no_prop": {"alpha": 1.0, "beta": 1.0, "gamma": 0.0, "graph_branch": True},
    "zero_edges": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "graph_branch": True},
    "source_only_no_graph_branch": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "graph_branch": False},
}


def configure_variant(base_cfg, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant}. Available: {sorted(VARIANTS)}")
    spec = VARIANTS[variant]
    cfg = copy.deepcopy(base_cfg)
    cfg.GRAPH_WEIGHT_ALPHA = float(spec["alpha"])
    cfg.GRAPH_WEIGHT_BETA = float(spec["beta"])
    cfg.GRAPH_WEIGHT_GAMMA = float(spec["gamma"])
    cfg.USE_GRAPH_BRANCH = bool(spec["graph_branch"])
    return cfg


@torch.inference_mode()
def role_coverage_for_variant(
    model,
    tokenizer,
    loader,
    cfg,
    device,
    *,
    top_k: int,
    max_samples: int,
) -> dict:
    rows = []
    seen = 0
    for batch in loader:
        bs = int(batch["labels"].shape[0])
        for bi in range(bs):
            if max_samples and seen >= max_samples:
                break
            label = int(batch["labels"][bi].item())
            if label != 1:
                continue
            stmts = batch["statements"][bi]
            metas = batch["statement_meta"][bi]
            labs = batch["statement_labels"][bi].float().cpu().tolist()
            if not stmts or not metas:
                continue

            pred_items = node_perturbation_rationales(
                model,
                tokenizer,
                batch["input_ids"][bi : bi + 1].to(device),
                batch["position_idx"][bi : bi + 1].to(device),
                batch["attn_mask"][bi : bi + 1].to(device),
                batch["graph_adj"][bi : bi + 1].to(device),
                batch["graph_mask"][bi : bi + 1].to(device),
                stmts,
                metas,
                batch["codes"][bi],
                top_k=top_k,
            )

            gold_roles = set()
            pred_roles = set()
            for j, meta in enumerate(metas):
                if j < len(labs) and labs[j] >= 0.5:
                    gold_roles.update(classify_roles(meta.text))
            for item in pred_items:
                pred_roles.update(classify_roles(str(item.get("text", ""))))

            p, r, f = prf(pred_roles, gold_roles)
            _, core_r, _ = prf(pred_roles & CORE_ROLE_SET, gold_roles & CORE_ROLE_SET)
            rows.append({"role_precision": p, "role_recall": r, "role_f1": f, "core_role_recall": core_r})
            seen += 1
        if max_samples and seen >= max_samples:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return {"role_precision": float("nan"), "role_recall": float("nan"), "role_f1": float("nan"), "core_role_recall": float("nan"), "role_samples": 0}
    return {
        "role_precision": float(df["role_precision"].mean()),
        "role_recall": float(df["role_recall"].mean()),
        "role_f1": float(df["role_f1"].mean()),
        "core_role_recall": float(df["core_role_recall"].mean()),
        "role_samples": int(len(df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PonziSense inference-time graph component ablations.")
    parser.add_argument("--test-path", default="./datafiles/processed/test.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument("--with-role-coverage", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-role-samples", type=int, default=120)
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    base_cfg = load_cfg(args)
    device = get_device(base_cfg, args.device)
    base_df = read_one_csv(args.test_path)

    out_root = Path(args.output_dir) / "icse" / "graph_component_ablation"
    ensure_dir(out_root)

    rows = []
    model_cache = {}
    for variant in args.variants:
        cfg = configure_variant(base_cfg, variant)
        cache_key = "graph" if cfg.USE_GRAPH_BRANCH else "source_only"
        if cache_key not in model_cache:
            model_cache[cache_key] = load_model_tokenizer(cfg, device, checkpoint=args.checkpoint)
        model, tokenizer = model_cache[cache_key]

        dataset = make_dataset_from_frame(base_df, tokenizer, cfg)
        loader = make_loader(dataset, cfg, device, shuffle=False)
        y_true, y_score = collect_scores(model, tokenizer, loader, cfg, device)
        metrics = classification_from_scores(y_true, y_score, threshold=cfg.PRED_THRESHOLD)
        metrics.update(
            {
                "variant": variant,
                "graph_branch": bool(cfg.USE_GRAPH_BRANCH),
                "graph_weight_alpha": float(cfg.GRAPH_WEIGHT_ALPHA),
                "graph_weight_beta": float(cfg.GRAPH_WEIGHT_BETA),
                "graph_weight_gamma": float(cfg.GRAPH_WEIGHT_GAMMA),
            }
        )

        if args.with_role_coverage:
            role_loader = make_loader(dataset, cfg, device, shuffle=False)
            metrics.update(
                role_coverage_for_variant(
                    model,
                    tokenizer,
                    role_loader,
                    cfg,
                    device,
                    top_k=args.top_k,
                    max_samples=args.max_role_samples,
                )
            )
        rows.append(metrics)
        print(f"[{variant}] {metrics}")

    full = next((r for r in rows if r["variant"] == "full"), None)
    if full:
        for row in rows:
            row["delta_f1_vs_full"] = float(row["f1"] - full["f1"])
            row["delta_auprc_vs_full"] = float(row.get("auprc", float("nan")) - full.get("auprc", float("nan")))
            if "role_f1" in row and "role_f1" in full:
                row["delta_role_f1_vs_full"] = float(row["role_f1"] - full["role_f1"])

    pd.DataFrame(rows).to_csv(out_root / "graph_component_ablation_metrics.csv", index=False)
    save_json(
        {
            "test_path": args.test_path,
            "checkpoint": args.checkpoint,
            "variants": rows,
            "note": "These are inference-time component diagnostics on the same checkpoint, not retrained ablation models.",
        },
        out_root / "graph_component_ablation_summary.json",
    )


if __name__ == "__main__":
    main()
