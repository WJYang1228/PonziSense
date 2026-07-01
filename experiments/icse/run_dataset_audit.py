from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.statements import build_statement_labels


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


def ensure_dir(path: str | Path) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def save_json(obj, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def coerce_code_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "code" not in df.columns:
        for candidate in ("source_code", "source", "contract_source"):
            if candidate in df.columns:
                df["code"] = df[candidate]
                break
    if "label" not in df.columns:
        for candidate in ("target", "is_ponzi", "ponzi"):
            if candidate in df.columns:
                df["label"] = df[candidate]
                break
    if "explain" not in df.columns:
        for candidate in ("explanation", "rationale", "evidence"):
            if candidate in df.columns:
                df["explain"] = df[candidate]
                break
        if "explain" not in df.columns:
            df["explain"] = ""
    if "code" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain code/source and label columns.")
    df["code"] = df["code"].fillna("").astype(str)
    df["explain"] = df["explain"].fillna("").astype(str)
    df["label"] = df["label"].fillna(0).astype(int)
    return df


def strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    return re.sub(r"//.*", " ", code)


def normalize_basic(code: str) -> str:
    return re.sub(r"\s+", " ", strip_comments(code)).strip().lower()


def normalize_template(code: str) -> str:
    code = strip_comments(code)
    code = re.sub(r"0x[a-fA-F0-9]{8,}", " HEXADDR ", code)
    code = re.sub(r'"(?:\\.|[^"\\])*"', " STR ", code)
    code = re.sub(r"'(?:\\.|[^'\\])*'", " STR ", code)
    code = re.sub(r"\b\d+(?:\.\d+)?\b", " NUM ", code)
    mapping: dict[str, str] = {}

    def repl(match: re.Match) -> str:
        token = match.group(0)
        if token.lower() in SOLIDITY_KEYWORDS:
            return token.lower()
        if token not in mapping:
            mapping[token] = f"id_{len(mapping) + 1}"
        return mapping[token]

    code = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", repl, code)
    return re.sub(r"\s+", " ", code).strip().lower()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def add_hash_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "raw_hash" not in df.columns:
        df["raw_hash"] = df["code"].map(digest)
    if "basic_hash" not in df.columns:
        df["basic_hash"] = df["code"].map(lambda x: digest(normalize_basic(x)))
    if "template_hash" not in df.columns:
        df["template_hash"] = df["code"].map(lambda x: digest(normalize_template(x)))
    return df


def split_stats(name: str, df: pd.DataFrame, max_annotation_samples: int) -> dict:
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    explain_nonempty = df["explain"].astype(str).str.strip().ne("").sum()
    pos_explain_nonempty = df[df["label"] == 1]["explain"].astype(str).str.strip().ne("").sum()
    code_lengths = df["code"].astype(str).str.len()

    rationale_sizes = []
    stmt_counts = []
    sampled = df[df["label"] == 1].head(max_annotation_samples) if max_annotation_samples else df.iloc[0:0]
    for _, row in sampled.iterrows():
        statements, labels, _ = build_statement_labels(str(row["code"]), str(row["explain"]))
        stmt_counts.append(len(statements))
        rationale_sizes.append(int(sum(labels)))

    return {
        "split": name,
        "size": int(len(df)),
        "positive": pos,
        "negative": neg,
        "positive_ratio": float(pos / len(df)) if len(df) else 0.0,
        "explain_nonempty": int(explain_nonempty),
        "positive_explain_nonempty": int(pos_explain_nonempty),
        "unique_raw_hash": int(df["raw_hash"].nunique()),
        "unique_basic_hash": int(df["basic_hash"].nunique()),
        "unique_template_hash": int(df["template_hash"].nunique()),
        "mean_code_chars": float(code_lengths.mean()) if len(df) else 0.0,
        "median_code_chars": float(code_lengths.median()) if len(df) else 0.0,
        "annotation_sampled_positive": int(len(sampled)),
        "mean_statement_count": float(pd.Series(stmt_counts).mean()) if stmt_counts else float("nan"),
        "mean_rationale_statement_count": float(pd.Series(rationale_sizes).mean()) if rationale_sizes else float("nan"),
    }


def pairwise_overlap(split_frames: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    keys = ["raw_hash", "basic_hash", "template_hash"]
    if any("address" in df.columns for df in split_frames.values()):
        keys.insert(0, "address")
    for left, right in combinations(split_frames.keys(), 2):
        a = split_frames[left]
        b = split_frames[right]
        for key in keys:
            if key not in a.columns or key not in b.columns:
                continue
            av = set(str(x) for x in a[key].dropna().tolist() if str(x).strip())
            bv = set(str(x) for x in b[key].dropna().tolist() if str(x).strip())
            inter = av & bv
            rows.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "key": key,
                    "left_unique": len(av),
                    "right_unique": len(bv),
                    "overlap": len(inter),
                    "overlap_over_left": float(len(inter) / len(av)) if av else 0.0,
                    "overlap_over_right": float(len(inter) / len(bv)) if bv else 0.0,
                }
            )
    return rows


def group_overlap(full_df: pd.DataFrame) -> list[dict]:
    if "group_id" not in full_df.columns or "split" not in full_df.columns:
        return []
    rows = []
    for group_id, sub in full_df.groupby("group_id"):
        splits = sorted(set(str(x) for x in sub["split"].dropna().tolist()))
        if len(splits) <= 1:
            continue
        rows.append(
            {
                "group_id": str(group_id),
                "splits": ";".join(splits),
                "size": int(len(sub)),
                "positive": int((sub["label"] == 1).sum()) if "label" in sub.columns else None,
            }
        )
    return rows


def maybe_read(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return add_hash_columns(coerce_code_column(read_csv_flexible(p)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PonziSense dataset splits and rationale fields.")
    parser.add_argument("--train-path", default="datafiles/processed/train.csv")
    parser.add_argument("--val-path", default="datafiles/processed/val.csv")
    parser.add_argument("--test-path", default="datafiles/processed/test.csv")
    parser.add_argument("--full-path", default="datafiles/processed/full_processed_with_groups.csv")
    parser.add_argument("--metadata-path", default="datafiles/ponzi_e_constructed/ponzi_e_real_contracts_metadata.csv")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--max-annotation-samples", type=int, default=1000)
    args = parser.parse_args()

    out_root = Path(args.output_dir) / "icse" / "dataset_audit"
    ensure_dir(out_root)

    split_frames = {}
    for name, path in {"train": args.train_path, "val": args.val_path, "test": args.test_path}.items():
        df = maybe_read(path)
        if df is not None:
            split_frames[name] = df

    if not split_frames:
        raise ValueError("No split files were found. Check --train-path/--val-path/--test-path.")

    stats = [split_stats(name, df, args.max_annotation_samples) for name, df in split_frames.items()]
    overlap_rows = pairwise_overlap(split_frames)

    full_df = maybe_read(args.full_path)
    group_rows = group_overlap(full_df) if full_df is not None else []

    metadata_summary = {}
    metadata_path = Path(args.metadata_path) if args.metadata_path else None
    if metadata_path and metadata_path.exists():
        meta = read_csv_flexible(metadata_path)
        metadata_summary = {
            "metadata_path": str(metadata_path),
            "rows": int(len(meta)),
            "columns": list(meta.columns),
            "address_nonempty": int(meta["address"].astype(str).str.strip().ne("").sum())
            if "address" in meta.columns
            else None,
            "quality_score_mean": float(pd.to_numeric(meta.get("quality_score"), errors="coerce").mean())
            if "quality_score" in meta.columns
            else None,
            "mechanism_score_mean": float(pd.to_numeric(meta.get("mechanism_score"), errors="coerce").mean())
            if "mechanism_score" in meta.columns
            else None,
        }

    pd.DataFrame(stats).to_csv(out_root / "split_stats.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(out_root / "pairwise_overlap.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out_root / "group_overlap.csv", index=False)

    summary = {
        "split_stats": stats,
        "pairwise_overlap": overlap_rows,
        "group_overlap_count": int(len(group_rows)),
        "metadata_summary": metadata_summary,
        "notes": [
            "raw_hash detects exact source duplicates.",
            "basic_hash ignores comments and whitespace.",
            "template_hash additionally normalizes identifiers and literals.",
            "Rationale statement counts are computed from the existing explain field and are not a substitute for a fresh human inter-annotator agreement study.",
        ],
    }
    save_json(summary, out_root / "dataset_audit_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
