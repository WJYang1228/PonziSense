from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("datafiles/ponzi_e_constructed/ponzi_e_real_contracts.csv")
DEFAULT_OUTPUT = Path("datafiles/processed_ponzi_e")
EXPECTED = {"total": 8233, "positive": 744, "negative": 7489}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_code_basic(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", " ", str(code), flags=re.DOTALL)
    code = re.sub(r"//.*?$", " ", code, flags=re.MULTILINE)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\n\s*\n+", "\n", code)
    return code.strip()


def normalize_code_template(code: str) -> str:
    code = normalize_code_basic(code).lower()
    code = re.sub(r"0x[a-f0-9]{40}", "ADDR", code)
    code = re.sub(r'"([^"\\]|\\.)*"', "STR", code)
    code = re.sub(r"'([^'\\]|\\.)*'", "STR", code)
    code = re.sub(r"\b\d+\b", "NUM", code)
    return re.sub(r"\s+", " ", code).strip()


def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"code", "label", "explain"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df[["code", "label", "explain"]].copy()
    df["code"] = df["code"].fillna("").astype(str)
    df["explain"] = df["explain"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df = df[df["code"].str.strip() != ""].reset_index(drop=True)
    return df


def add_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["raw_hash"] = out["code"].map(sha256_text)
    out["basic_hash"] = out["code"].map(lambda x: sha256_text(normalize_code_basic(x)))
    out["template_hash"] = out["code"].map(lambda x: sha256_text(normalize_code_template(x)))
    return out


def stratified_split(df: pd.DataFrame, seed: int, ratios=(0.6, 0.2, 0.2)):
    rng = random.Random(seed)
    parts = {"train": [], "val": [], "test": []}
    for label, sub in df.groupby("label"):
        idxs = list(sub.index)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        train_ids = idxs[:n_train]
        val_ids = idxs[n_train : n_train + n_val]
        test_ids = idxs[n_train + n_val :]
        parts["train"].extend(train_ids)
        parts["val"].extend(val_ids)
        parts["test"].extend(test_ids)
    return (
        df.loc[parts["train"]].sample(frac=1.0, random_state=seed).reset_index(drop=True),
        df.loc[parts["val"]].sample(frac=1.0, random_state=seed).reset_index(drop=True),
        df.loc[parts["test"]].sample(frac=1.0, random_state=seed).reset_index(drop=True),
    )


def summarize(df: pd.DataFrame, name: str) -> dict:
    label_dist = {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).to_dict().items()}
    return {
        "name": name,
        "size": int(len(df)),
        "label_distribution": label_dist,
        "unique_raw_hash": int(df["raw_hash"].nunique()) if "raw_hash" in df else None,
        "unique_basic_hash": int(df["basic_hash"].nunique()) if "basic_hash" in df else None,
        "unique_template_hash": int(df["template_hash"].nunique()) if "template_hash" in df else None,
    }


def write_split(train_df, val_df, test_df, output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = ["code", "label", "explain"]
    train_df[keep].to_csv(output_dir / "train.csv", index=False, encoding="utf-8-sig")
    val_df[keep].to_csv(output_dir / "val.csv", index=False, encoding="utf-8-sig")
    test_df[keep].to_csv(output_dir / "test.csv", index=False, encoding="utf-8-sig")
    full = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        axis=0,
    ).reset_index(drop=True)
    full.to_csv(output_dir / "full_processed_with_groups.csv", index=False, encoding="utf-8-sig")
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the paper-aligned Ponzi-E constructed dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--also-write-default-processed",
        action="store_true",
        help="Also write datafiles/processed so existing train/evaluate scripts use the Ponzi-E split.",
    )
    args = parser.parse_args()

    df = add_audit_columns(load_input(args.input))
    total = int(len(df))
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    warnings = []
    if total != EXPECTED["total"]:
        warnings.append(f"expected total={EXPECTED['total']}, observed={total}")
    if pos != EXPECTED["positive"]:
        warnings.append(f"expected positive={EXPECTED['positive']}, observed={pos}")
    if neg != EXPECTED["negative"]:
        warnings.append(f"expected negative={EXPECTED['negative']}, observed={neg}")

    train_df, val_df, test_df = stratified_split(df, args.seed)
    report = {
        "input_csv": str(args.input),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "expected": EXPECTED,
        "observed": {"total": total, "positive": pos, "negative": neg},
        "warnings": warnings,
        "split": {
            "train": summarize(train_df, "train"),
            "val": summarize(val_df, "val"),
            "test": summarize(test_df, "test"),
        },
        "note": "Default split is label-stratified to preserve the paper label distribution. Use full_processed_with_groups.csv for hash/template leakage checks.",
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    write_split(train_df, val_df, test_df, args.output_dir, report)
    if args.also_write_default_processed:
        write_split(train_df, val_df, test_df, Path("datafiles/processed"), report)


if __name__ == "__main__":
    main()
