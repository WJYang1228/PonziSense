import pandas as pd
import torch
from torch.utils.data import Dataset

from data.feature_extractor import convert_code_to_features, build_attention_mask
from graph.statement_graph import build_statement_graph_tensors
from utils.statements import build_statement_labels


class PonziDFGDataset(Dataset):
    def __init__(self, dataframe, tokenizer, cfg):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        code = str(row["code"])
        raw_label = int(row["label"])
        label = 1 if raw_label == self.cfg.POSITIVE_LABEL else 0
        explain = "" if pd.isna(row.get("explain", "")) else str(row.get("explain", ""))

        feature = convert_code_to_features(code, label, explain, self.tokenizer, self.cfg)
        attn_mask = build_attention_mask(feature, self.cfg, self.tokenizer)

        statements, stmt_labels, stmt_blocks = build_statement_labels(code, explain)

        if len(statements) > self.cfg.EXPLAIN_MAX_STATEMENTS:
            statements = statements[: self.cfg.EXPLAIN_MAX_STATEMENTS]
            stmt_labels = stmt_labels[: self.cfg.EXPLAIN_MAX_STATEMENTS]
            stmt_blocks = stmt_blocks[: self.cfg.EXPLAIN_MAX_STATEMENTS]

        graph_adj, graph_mask, _ = build_statement_graph_tensors(code, self.cfg)

        return {
            "input_ids": torch.tensor(feature.input_ids, dtype=torch.long),
            "position_idx": torch.tensor(feature.position_idx, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.bool),
            "label": torch.tensor(feature.label, dtype=torch.long),
            "code": code,
            "explain": explain,
            "input_tokens": feature.input_tokens,
            "statements": statements,
            "statement_labels": torch.tensor(stmt_labels, dtype=torch.float),
            "statement_meta": stmt_blocks,
            "graph_adj": torch.tensor(graph_adj, dtype=torch.float32),
            "graph_mask": torch.tensor(graph_mask, dtype=torch.float32),
        }


def read_one_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="gbk")

    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"code", "label", "explain"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少必要字段: {missing}")

    df = df[["code", "label", "explain"]].copy()
    df = df[df["code"].notna()].reset_index(drop=True)
    return df


def load_datasets(tokenizer, cfg):
    train_df = read_one_csv(cfg.TRAIN_PATH)
    val_df = read_one_csv(cfg.VAL_PATH)
    test_df = read_one_csv(cfg.TEST_PATH)

    train_set = PonziDFGDataset(train_df, tokenizer, cfg)
    val_set = PonziDFGDataset(val_df, tokenizer, cfg)
    test_set = PonziDFGDataset(test_df, tokenizer, cfg)

    return train_set, val_set, test_set