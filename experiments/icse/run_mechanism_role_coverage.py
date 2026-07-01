from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

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


ROLE_PATTERNS = {
    "fund_inflow": [
        r"\bmsg\.value\b", r"\bpayable\b", r"\bdeposit\b", r"\binvest\b",
        r"\bcontribute\b", r"\bbuy\b", r"\bjoin\b", r"\benter\b",
    ],
    "participant_registration": [
        r"\bparticipants?\b", r"\binvestors?\b", r"\busers?\b", r"\bmembers?\b",
        r"\bplayers?\b", r"\bqueue\b", r"\.push\s*\(", r"\bregister\b", r"\breferr?er\b",
    ],
    "state_accounting": [
        r"\bbalances?\b", r"\bdeposited\b", r"\bamount\b", r"\binvested\b",
        r"\btotal\b", r"\bshares?\b", r"\bcounts?\b", r"\+=", r"-=", r"=",
    ],
    "reward_computation": [
        r"\breward\b", r"\bprofit\b", r"\breturns?\b", r"\bdividend\b", r"\bbonus\b",
        r"\binterest\b", r"\bpercent\b", r"\brate\b", r"\bmultiplier\b", r"\*", r"/\s*100",
    ],
    "payout_condition": [
        r"\bif\s*\(", r"\brequire\s*\(", r"\bbalance\s*[><=]", r"\bcursor\b",
        r"\bindex\b", r"\bnext\b", r"\bwhile\s*\(",
    ],
    "fund_transfer": [
        r"\.transfer\s*\(", r"\.send\s*\(", r"\.call\.value\s*\(", r"\.call\s*\{",
        r"\bselfdestruct\s*\(",
    ],
    "owner_or_referral_fee": [
        r"\bowner\b", r"\badmin\b", r"\bfee\b", r"\bcommission\b", r"\breferr?al\b",
        r"\bsponsor\b", r"\bupline\b", r"\bparent\b",
    ],
}

CORE_ROLE_SET = {
    "fund_inflow",
    "participant_registration",
    "state_accounting",
    "reward_computation",
    "payout_condition",
    "fund_transfer",
}


def classify_roles(text: str) -> set[str]:
    low = text.lower()
    roles: set[str] = set()
    for role, patterns in ROLE_PATTERNS.items():
        if any(re.search(pat, low) for pat in patterns):
            roles.add(role)
    return roles


def prf(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    inter = len(pred & gold)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure whether rationales cover Ponzi mechanism roles.")
    parser.add_argument("--test-path", default="./datafiles/processed/test.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-negatives", action="store_true")
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    cfg = load_cfg(args)
    device = get_device(cfg, args.device)
    model, tokenizer = load_model_tokenizer(cfg, device, checkpoint=args.checkpoint)
    dataset = make_dataset_from_csv(args.test_path, tokenizer, cfg)
    loader = make_loader(dataset, cfg, device, shuffle=False)

    out_root = Path(args.output_dir) / "icse" / "mechanism_role_coverage"
    ensure_dir(out_root)

    rows: list[dict] = []
    role_gold_counter: Counter[str] = Counter()
    role_pred_counter: Counter[str] = Counter()
    role_hit_counter: Counter[str] = Counter()
    role_support_counter: Counter[str] = Counter()

    for batch_idx, batch in enumerate(loader):
        bs = batch["labels"].shape[0]
        for bi in range(bs):
            label = int(batch["labels"][bi].item())
            if not args.include_negatives and label != 1:
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
                top_k=args.top_k,
            )

            gold_roles: set[str] = set()
            pred_roles: set[str] = set()
            gold_stmt_ids: set[int] = set()
            pred_stmt_ids: set[int] = set()

            for j, meta in enumerate(metas):
                if j < len(labs) and labs[j] >= 0.5:
                    roles = classify_roles(meta.text)
                    gold_roles.update(roles)
                    gold_stmt_ids.add(meta.stmt_id)

            for item in pred_items:
                roles = classify_roles(item.get("text", ""))
                pred_roles.update(roles)
                pred_stmt_ids.add(int(item["stmt_id"]))

            p, r, f = prf(pred_roles, gold_roles)
            _, core_r, _ = prf(pred_roles & CORE_ROLE_SET, gold_roles & CORE_ROLE_SET)

            for role in gold_roles:
                role_gold_counter[role] += 1
                role_support_counter[role] += 1
            for role in pred_roles:
                role_pred_counter[role] += 1
            for role in gold_roles & pred_roles:
                role_hit_counter[role] += 1

            rows.append(
                {
                    "batch_idx": batch_idx,
                    "sample_idx": bi,
                    "label": label,
                    "gold_stmt_count": len(gold_stmt_ids),
                    "pred_stmt_count": len(pred_stmt_ids),
                    "gold_roles": ";".join(sorted(gold_roles)),
                    "pred_roles": ";".join(sorted(pred_roles)),
                    "role_precision": p,
                    "role_recall": r,
                    "role_f1": f,
                    "core_role_recall": core_r,
                }
            )

    df = pd.DataFrame(rows)
    summary = {
        "n_samples": int(len(df)),
        "top_k": args.top_k,
        "role_precision": float(df["role_precision"].mean()) if len(df) else float("nan"),
        "role_recall": float(df["role_recall"].mean()) if len(df) else float("nan"),
        "role_f1": float(df["role_f1"].mean()) if len(df) else float("nan"),
        "core_role_recall": float(df["core_role_recall"].mean()) if len(df) else float("nan"),
        "per_role_recall": {
            role: float(role_hit_counter[role] / role_support_counter[role])
            for role in sorted(role_support_counter)
            if role_support_counter[role] > 0
        },
        "gold_role_frequency": dict(sorted(role_gold_counter.items())),
        "pred_role_frequency": dict(sorted(role_pred_counter.items())),
    }

    df.to_csv(out_root / "mechanism_role_coverage_samples.csv", index=False)
    save_json(summary, out_root / "mechanism_role_coverage_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
