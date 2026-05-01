import os
import re
import json
import math
import hashlib
import random
from collections import defaultdict

import pandas as pd


# =========================
# 配置区
# =========================
INPUT_CSV = r"./datafiles/PonziDataset_20221114_explain_augmented_negatives.csv"
OUTPUT_DIR = r"./datafiles/processed"

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

RANDOM_SEED = 42

PONZI_E_TOTAL_CONTRACTS = 8233
PONZI_E_POSITIVE_CONTRACTS = 744
PONZI_E_NEGATIVE_CONTRACTS = 7489
PONZI_E_POSITIVE_LABEL = 1

# Paper convention: Ponzi is the positive class; Config.POSITIVE_LABEL defaults to 1.
# 这里只是保留原始 label，不做标签翻转


# =========================
# 通用工具
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def remove_comments(code: str) -> str:
    """
    去掉 Solidity 中常见的 // 和 /* */ 注释
    """
    if not isinstance(code, str):
        return ""

    # 去掉块注释
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    # 去掉行注释
    code = re.sub(r"//.*?$", " ", code, flags=re.MULTILINE)
    return code


def normalize_whitespace(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\n\s*\n+", "\n", code)
    return code.strip()


def normalize_code_basic(code: str) -> str:
    """
    基础归一化：
    - 去注释
    - 统一空白
    """
    code = remove_comments(code)
    code = normalize_whitespace(code)
    return code


def normalize_code_template(code: str) -> str:
    """
    更强的模板归一化，用于模板去重 / 组划分：
    - 去注释
    - 统一地址
    - 统一字符串常量
    - 统一数字
    - 尽量保留 Solidity 关键字，泛化普通标识符
    """
    code = remove_comments(code)

    # 小写化，减少格式差异
    code = code.lower()

    # 地址常量替换
    code = re.sub(r"0x[a-f0-9]{40}", "ADDR", code)

    # 字符串常量替换
    code = re.sub(r'"([^"\\]|\\.)*"', "STR", code)
    code = re.sub(r"'([^'\\]|\\.)*'", "STR", code)

    # 数字替换
    code = re.sub(r"\b\d+\b", "NUM", code)

    # 常见 Solidity 关键字保留
    solidity_keywords = {
        "pragma", "solidity", "contract", "library", "interface", "function",
        "constructor", "modifier", "event", "mapping", "struct", "enum",
        "address", "uint", "uint256", "uint8", "uint16", "uint32", "uint64",
        "int", "bool", "string", "bytes", "public", "private", "internal",
        "external", "view", "pure", "payable", "returns", "return", "if",
        "else", "for", "while", "do", "break", "continue", "require", "assert",
        "revert", "emit", "new", "memory", "storage", "calldata", "this",
        "msg", "sender", "value", "block", "timestamp", "now", "tx", "origin",
        "transfer", "send", "call", "delegatecall", "selfdestruct", "suicide",
        "owner", "balance", "balances", "push", "length", "true", "false"
    }

    # 把标识符统一成 ID，但保留关键字
    def replace_identifier(match):
        token = match.group(0)
        if token in solidity_keywords:
            return token
        return "ID"

    code = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", replace_identifier, code)

    # 空白统一
    code = normalize_whitespace(code)
    return code


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


# =========================
# 数据处理主逻辑
# =========================
def load_and_standardize(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="gbk")

    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"code", "label", "explain"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要字段: {missing}")

    df = df[["code", "label", "explain"]].copy()
    df["code"] = df["code"].apply(safe_str)
    df["explain"] = df["explain"].apply(safe_str)
    df["label"] = df["label"].astype(int)

    # 去掉空 code
    df = df[df["code"].str.strip() != ""].reset_index(drop=True)
    return df


def build_signatures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["code_basic_norm"] = df["code"].apply(normalize_code_basic)
    df["code_template_norm"] = df["code"].apply(normalize_code_template)

    df["raw_hash"] = df["code"].apply(sha256_text)
    df["basic_hash"] = df["code_basic_norm"].apply(sha256_text)
    df["template_hash"] = df["code_template_norm"].apply(sha256_text)

    return df


def exact_deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    先按 raw_hash 去重，再按 basic_hash 去重
    """
    stats = {}

    n0 = len(df)
    df1 = df.drop_duplicates(subset=["raw_hash"], keep="first").reset_index(drop=True)
    n1 = len(df1)

    df2 = df1.drop_duplicates(subset=["basic_hash"], keep="first").reset_index(drop=True)
    n2 = len(df2)

    stats["before"] = n0
    stats["after_raw_dedup"] = n1
    stats["after_basic_dedup"] = n2
    stats["removed_by_raw_hash"] = n0 - n1
    stats["removed_by_basic_hash"] = n1 - n2

    return df2, stats


def assign_template_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 template_hash 分组，保证同模板尽量不跨集合
    """
    df = df.copy()
    df["group_id"] = df["template_hash"]
    return df


def stratified_group_split(df: pd.DataFrame,
                           train_ratio=0.8,
                           val_ratio=0.1,
                           test_ratio=0.1,
                           seed=42):
    """
    近似的按 group + label 分层划分。
    目标：
    - 同一个 group_id 只进一个集合
    - 尽量保持标签比例
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    rng = random.Random(seed)

    # 每个组统计
    group_info = []
    for gid, sub in df.groupby("group_id"):
        label_counts = sub["label"].value_counts().to_dict()
        group_info.append({
            "group_id": gid,
            "size": len(sub),
            "label_counts": label_counts,
        })

    rng.shuffle(group_info)

    total = len(df)
    target_train = total * train_ratio
    target_val = total * val_ratio
    target_test = total * test_ratio

    train_groups, val_groups, test_groups = set(), set(), set()
    train_size = val_size = test_size = 0

    # 贪心分配：优先按当前集合大小接近期望值
    for g in sorted(group_info, key=lambda x: x["size"], reverse=True):
        size = g["size"]

        deficits = {
            "train": target_train - train_size,
            "val": target_val - val_size,
            "test": target_test - test_size,
        }

        # 优先放到“最缺”的集合
        chosen = max(deficits, key=deficits.get)

        if chosen == "train":
            train_groups.add(g["group_id"])
            train_size += size
        elif chosen == "val":
            val_groups.add(g["group_id"])
            val_size += size
        else:
            test_groups.add(g["group_id"])
            test_size += size

    train_df = df[df["group_id"].isin(train_groups)].copy()
    val_df = df[df["group_id"].isin(val_groups)].copy()
    test_df = df[df["group_id"].isin(test_groups)].copy()

    return train_df, val_df, test_df


def summarize_split(df: pd.DataFrame, name: str) -> dict:
    label_dist = df["label"].value_counts(dropna=False).to_dict()
    return {
        "name": name,
        "size": int(len(df)),
        "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
        "unique_template_groups": int(df["group_id"].nunique()),
    }



def build_paper_alignment_report(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    total = int(len(df))
    label_dist = {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).to_dict().items()}
    expected = {
        "total_contracts": PONZI_E_TOTAL_CONTRACTS,
        "positive_contracts": PONZI_E_POSITIVE_CONTRACTS,
        "negative_contracts": PONZI_E_NEGATIVE_CONTRACTS,
        "positive_label": PONZI_E_POSITIVE_LABEL,
        "split_ratio": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
    }
    observed_pos = int((df["label"].astype(int) == PONZI_E_POSITIVE_LABEL).sum())
    observed_neg = int(total - observed_pos)
    warnings = []
    if total != PONZI_E_TOTAL_CONTRACTS:
        warnings.append(f"expected {PONZI_E_TOTAL_CONTRACTS} contracts, observed {total}")
    if observed_pos != PONZI_E_POSITIVE_CONTRACTS:
        warnings.append(f"expected {PONZI_E_POSITIVE_CONTRACTS} positive Ponzi contracts, observed {observed_pos}")
    if observed_neg != PONZI_E_NEGATIVE_CONTRACTS:
        warnings.append(f"expected {PONZI_E_NEGATIVE_CONTRACTS} negative contracts, observed {observed_neg}")
    return {
        "paper_source_of_truth": "ccs2026b-paper4715.pdf",
        "expected": expected,
        "observed": {
            "total_contracts": total,
            "positive_contracts": observed_pos,
            "negative_contracts": observed_neg,
            "label_distribution": label_dist,
            "split_sizes": {
                "train": int(len(train_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
            },
        },
        "warnings": warnings,
    }

def save_outputs(train_df, val_df, test_df, report: dict, output_dir: str):
    ensure_dir(output_dir)

    # 训练时只保留必要字段
    keep_cols = ["code", "label", "explain"]

    train_df[keep_cols].to_csv(
        os.path.join(output_dir, "train.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    val_df[keep_cols].to_csv(
        os.path.join(output_dir, "val.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    test_df[keep_cols].to_csv(
        os.path.join(output_dir, "test.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 也保存一个完整的带签名和分组信息的数据，方便审计
    full_df = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ], axis=0).reset_index(drop=True)

    full_df.to_csv(
        os.path.join(output_dir, "full_processed_with_groups.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main():
    ensure_dir(OUTPUT_DIR)

    print("[1] 读取原始数据")
    df = load_and_standardize(INPUT_CSV)
    print(f"原始样本数: {len(df)}")

    print("[2] 构建归一化与签名")
    df = build_signatures(df)

    print("[3] 执行严格去重")
    df_dedup, dedup_stats = exact_deduplicate(df)
    print("去重统计:", dedup_stats)

    print("[4] 分配模板组")
    df_grouped = assign_template_groups(df_dedup)

    print("[5] 按模板组划分 train/val/test")
    train_df, val_df, test_df = stratified_group_split(
        df_grouped,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED,
    )

    # 最终列安全检查
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(part) == 0:
            raise ValueError(f"{name} 集为空，请调整划分策略或检查数据")

    report = {
        "input_csv": INPUT_CSV,
        "output_dir": OUTPUT_DIR,
        "random_seed": RANDOM_SEED,
        "split_ratio": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "dedup_stats": dedup_stats,
        "paper_alignment": build_paper_alignment_report(df_dedup, train_df, val_df, test_df),
        "splits": {
            "train": summarize_split(train_df, "train"),
            "val": summarize_split(val_df, "val"),
            "test": summarize_split(test_df, "test"),
        }
    }

    print("[6] 保存结果")
    save_outputs(train_df, val_df, test_df, report, OUTPUT_DIR)

    print("处理完成。输出文件如下：")
    print(os.path.join(OUTPUT_DIR, "train.csv"))
    print(os.path.join(OUTPUT_DIR, "val.csv"))
    print(os.path.join(OUTPUT_DIR, "test.csv"))
    print(os.path.join(OUTPUT_DIR, "full_processed_with_groups.csv"))
    print(os.path.join(OUTPUT_DIR, "report.json"))


if __name__ == "__main__":
    main()
