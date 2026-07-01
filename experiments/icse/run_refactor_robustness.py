from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.icse.icse_common import (
    ensure_dir,
    evaluate_classifier,
    get_device,
    load_cfg,
    load_model_tokenizer,
    make_dataset_from_frame,
    make_loader,
    save_json,
    set_reproducible_seed,
)


SOLIDITY_KEYWORDS = {
    "address", "bool", "bytes", "calldata", "constructor", "contract", "delete",
    "else", "emit", "enum", "event", "external", "false", "for", "function",
    "if", "immutable", "import", "int", "interface", "internal", "is", "library",
    "mapping", "memory", "modifier", "new", "override", "payable", "pragma",
    "private", "public", "pure", "receive", "return", "returns", "revert",
    "solidity", "storage", "string", "struct", "super", "this", "true", "try",
    "type", "uint", "uint8", "uint16", "uint32", "uint64", "uint128", "uint256",
    "unchecked", "using", "view", "virtual", "while",
}

SOLIDITY_BUILTINS = {
    "abi", "addmod", "assert", "balance", "block", "blockhash", "call",
    "callcode", "delegatecall", "ecrecover", "gasleft", "keccak256", "msg",
    "origin", "require", "ripemd160", "selfdestruct", "send", "sender", "sha256",
    "sig", "staticcall", "timestamp", "transfer", "tx", "value",
}


def strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    code = re.sub(r"//.*", " ", code)
    return code


def normalize_literals(code: str) -> str:
    code = re.sub(r"0x[a-fA-F0-9]{8,}", "0x0000000000000000000000000000000000000000", code)
    code = re.sub(r'"(?:\\.|[^"\\])*"', '"STR"', code)
    code = re.sub(r"'(?:\\.|[^'\\])*'", "'STR'", code)
    code = re.sub(r"\b\d+(?:\.\d+)?\b", "1", code)
    return code


def normalize_layout(code: str) -> str:
    out = []
    for line in code.splitlines():
        stripped = re.sub(r"\s+", " ", line.strip())
        if stripped:
            out.append(stripped)
    return "\n".join(out)


def normalize_identifiers(code: str) -> str:
    mapping: dict[str, str] = {}

    def repl(match: re.Match) -> str:
        token = match.group(0)
        low = token.lower()
        if low in SOLIDITY_KEYWORDS or low in SOLIDITY_BUILTINS:
            return token
        if token.startswith("0x"):
            return token
        if token not in mapping:
            mapping[token] = f"id_{len(mapping) + 1}"
        return mapping[token]

    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", repl, code)


def transform_code(code: str, variant: str) -> str:
    if variant == "original":
        return code
    if variant == "strip_comments":
        return strip_comments(code)
    if variant == "identifier_normalized":
        return normalize_identifiers(code)
    if variant == "literal_normalized":
        return normalize_literals(code)
    if variant == "layout_normalized":
        return normalize_layout(code)
    if variant == "all_refactor":
        return normalize_layout(normalize_literals(normalize_identifiers(strip_comments(code))))
    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate robustness under syntax-preserving shallow refactorings.")
    parser.add_argument("--test-path", default="./datafiles/processed/test.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "original",
            "strip_comments",
            "identifier_normalized",
            "literal_normalized",
            "layout_normalized",
            "all_refactor",
        ],
    )
    parser.add_argument("--save-transformed-csv", action="store_true")
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    cfg = load_cfg(args)
    device = get_device(cfg, args.device)
    model, tokenizer = load_model_tokenizer(cfg, device, checkpoint=args.checkpoint)

    base_df = pd.read_csv(args.test_path)
    base_df.columns = [c.strip().lower() for c in base_df.columns]
    missing = {"code", "label", "explain"} - set(base_df.columns)
    if missing:
        raise ValueError(f"{args.test_path} missing required columns: {sorted(missing)}")

    out_root = Path(args.output_dir) / "icse" / "refactor_robustness"
    ensure_dir(out_root)

    rows = []
    for variant in args.variants:
        df = base_df[["code", "label", "explain"]].copy()
        df["code"] = df["code"].astype(str).map(lambda x: transform_code(x, variant))
        if args.save_transformed_csv and variant != "original":
            df.to_csv(out_root / f"test_{variant}.csv", index=False, encoding="utf-8")
        dataset = make_dataset_from_frame(df, tokenizer, cfg)
        loader = make_loader(dataset, cfg, device, shuffle=False)
        metrics = evaluate_classifier(model, tokenizer, loader, cfg, device, threshold=cfg.PRED_THRESHOLD)
        metrics["variant"] = variant
        rows.append(metrics)
        print(f"[{variant}] {metrics}")

    original = next((r for r in rows if r["variant"] == "original"), None)
    if original is not None:
        for row in rows:
            row["delta_f1_vs_original"] = float(row["f1"] - original["f1"])
            row["delta_precision_vs_original"] = float(row["precision"] - original["precision"])
            row["delta_recall_vs_original"] = float(row["recall"] - original["recall"])

    pd.DataFrame(rows).to_csv(out_root / "refactor_robustness_metrics.csv", index=False)
    save_json(
        {"metrics": rows, "config": {"test_path": args.test_path, "checkpoint": args.checkpoint}},
        out_root / "refactor_robustness_metrics.json",
    )


if __name__ == "__main__":
    main()
