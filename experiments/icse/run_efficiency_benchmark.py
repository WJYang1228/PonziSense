from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    now_seconds,
    quantiles,
    save_json,
    set_reproducible_seed,
    synchronize_cuda,
)
from utils.rationale_extractor import node_perturbation_rationales


@torch.inference_mode()
def forward_batch(model, tokenizer, batch, device):
    probs, _, _ = model.forward_contract(
        batch["input_ids"].to(device),
        batch["position_idx"].to(device),
        batch["attn_mask"].to(device),
        labels=None,
        graph_adj=batch["graph_adj"].to(device),
        graph_mask=batch["graph_mask"].to(device),
        statements=batch["statements"],
        codes=batch["codes"],
        tokenizer=tokenizer,
    )
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure inference and explanation latency.")
    parser.add_argument("--test-path", default="./datafiles/processed/test.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-contracts", type=int, default=300)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--explain-samples", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    cfg = load_cfg(args)
    device = get_device(cfg, args.device)
    model, tokenizer = load_model_tokenizer(cfg, device, checkpoint=args.checkpoint)
    dataset = make_dataset_from_csv(args.test_path, tokenizer, cfg)
    loader = make_loader(dataset, cfg, device, shuffle=False)

    out_root = Path(args.output_dir) / "icse" / "efficiency"
    ensure_dir(out_root)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batch_latencies = []
    per_contract_forward_ms = []
    seen = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx < args.warmup_batches:
            forward_batch(model, tokenizer, batch, device)
            synchronize_cuda(device)
            continue
        bs = int(batch["labels"].shape[0])
        if args.max_contracts and seen >= args.max_contracts:
            break
        t0 = now_seconds(device)
        forward_batch(model, tokenizer, batch, device)
        t1 = now_seconds(device)
        latency_ms = (t1 - t0) * 1000.0
        batch_latencies.append(latency_ms)
        per_contract_forward_ms.extend([latency_ms / max(1, bs)] * bs)
        seen += bs

    explanation_ms = []
    explain_seen = 0
    for batch in loader:
        bs = int(batch["labels"].shape[0])
        for bi in range(bs):
            if args.explain_samples and explain_seen >= args.explain_samples:
                break
            stmts = batch["statements"][bi]
            metas = batch["statement_meta"][bi]
            if not stmts or not metas:
                continue
            t0 = now_seconds(device)
            node_perturbation_rationales(
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
                top_k=args.top_k,
            )
            t1 = now_seconds(device)
            explanation_ms.append((t1 - t0) * 1000.0)
            explain_seen += 1
        if args.explain_samples and explain_seen >= args.explain_samples:
            break

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    forward_q = quantiles(per_contract_forward_ms)
    summary = {
        "device": str(device),
        "batch_size": cfg.EVAL_BATCH_SIZE,
        "forward_contracts": int(len(per_contract_forward_ms)),
        "explanation_contracts": int(len(explanation_ms)),
        "forward_ms_per_contract": forward_q,
        "explanation_ms_per_contract": quantiles(explanation_ms),
        "throughput_contracts_per_second": float(1000.0 / forward_q["mean"])
        if per_contract_forward_ms and forward_q["mean"] > 0
        else float("nan"),
        "peak_cuda_memory_mb": peak_memory_mb,
    }

    pd.DataFrame({"batch_latency_ms": batch_latencies}).to_csv(out_root / "forward_batch_latencies.csv", index=False)
    pd.DataFrame({"explanation_latency_ms": explanation_ms}).to_csv(out_root / "explanation_latencies.csv", index=False)
    save_json(summary, out_root / "efficiency_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
