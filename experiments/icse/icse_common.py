from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from configs.config import Config
from data.collate import collate_fn
from data.dataset import PonziDFGDataset, read_one_csv
from models.model import PonziModel
from utils.metrics import cls_metrics

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover - sklearn is already a project dependency.
    average_precision_score = None
    roc_auc_score = None


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def save_json(obj, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_cfg(args=None) -> Config:
    cfg = Config()
    if args is not None:
        if getattr(args, "output_dir", None):
            cfg.OUTPUT_DIR = args.output_dir
        if getattr(args, "test_path", None):
            cfg.TEST_PATH = args.test_path
        if getattr(args, "val_path", None):
            cfg.VAL_PATH = args.val_path
        if getattr(args, "train_path", None):
            cfg.TRAIN_PATH = args.train_path
        if getattr(args, "batch_size", None):
            cfg.EVAL_BATCH_SIZE = int(args.batch_size)
        if getattr(args, "top_k", None):
            cfg.EXPLAIN_EVAL_TOP_K = int(args.top_k)
        if getattr(args, "max_statements", None):
            cfg.EXPLAIN_MAX_STATEMENTS = int(args.max_statements)
    return cfg


def cfg_to_dict(cfg: Config) -> dict:
    out = asdict(cfg) if is_dataclass(cfg) else dict(cfg.__dict__)
    out["CKPT_DIR"] = cfg.CKPT_DIR
    out["LOG_DIR"] = cfg.LOG_DIR
    out["PRED_DIR"] = cfg.PRED_DIR
    out["TABLES_DIR"] = cfg.TABLES_DIR
    return out


def get_device(cfg: Config, device_name: str | None = None) -> torch.device:
    wanted = device_name or cfg.DEVICE
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(wanted)


def load_model_tokenizer(
    cfg: Config,
    device: torch.device,
    checkpoint: str | None = None,
) -> tuple[PonziModel, RobertaTokenizer]:
    tokenizer = RobertaTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
    hf_config = RobertaConfig.from_pretrained(cfg.CONFIG_NAME)
    encoder = RobertaForSequenceClassification.from_pretrained(cfg.MODEL_NAME, config=hf_config)
    model = PonziModel(encoder, hf_config, num_clusters=cfg.NUM_CLUSTERS_K, cfg=cfg).to(device)

    ckpt_path = checkpoint or os.path.join(cfg.CKPT_DIR, "best_model.pt")
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, tokenizer


def make_dataset_from_csv(path: str, tokenizer, cfg: Config) -> PonziDFGDataset:
    return PonziDFGDataset(read_one_csv(path), tokenizer, cfg)


def make_dataset_from_frame(df: pd.DataFrame, tokenizer, cfg: Config) -> PonziDFGDataset:
    tmp = df.copy()
    tmp.columns = [c.strip().lower() for c in tmp.columns]
    return PonziDFGDataset(tmp[["code", "label", "explain"]], tokenizer, cfg)


def make_loader(dataset, cfg: Config, device: torch.device, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.NUM_WORKERS > 0,
    )


def _autocast_cm(cfg: Config, device: torch.device):
    return torch.autocast(device_type="cuda", enabled=cfg.USE_AMP and device.type == "cuda")


@torch.inference_mode()
def forward_batch_ponzi_probs(model, tokenizer, batch, cfg: Config, device: torch.device):
    with _autocast_cm(cfg, device):
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
    return probs[:, 1].detach().float().cpu()


@torch.inference_mode()
def evaluate_classifier(model, tokenizer, loader, cfg: Config, device: torch.device, threshold: float = 0.5) -> dict:
    y_true, y_score = collect_scores(model, tokenizer, loader, cfg, device)
    return classification_from_scores(y_true, y_score, threshold=threshold)


@torch.inference_mode()
def collect_scores(model, tokenizer, loader, cfg: Config, device: torch.device) -> tuple[list[int], list[float]]:
    y_true: list[int] = []
    y_score: list[float] = []
    for batch in loader:
        scores = forward_batch_ponzi_probs(model, tokenizer, batch, cfg, device)
        labels = batch["labels"].detach().cpu().tolist()
        y_true.extend(int(x) for x in labels)
        y_score.extend(float(x) for x in scores.tolist())
    return y_true, y_score


def expected_calibration_error(y_true: list[int], y_score: list[float], bins: int = 10) -> float:
    if not y_true:
        return float("nan")
    ys = np.asarray(y_true, dtype=int)
    ps = np.asarray(y_score, dtype=float)
    pred = (ps >= 0.5).astype(int)
    confs = np.maximum(ps, 1.0 - ps)
    ece = 0.0
    for left in np.linspace(0.0, 1.0, bins, endpoint=False):
        right = min(1.0, left + 1.0 / bins)
        if right >= 1.0:
            mask = (confs >= left) & (confs <= right)
        else:
            mask = (confs >= left) & (confs < right)
        if not np.any(mask):
            continue
        conf = float(np.mean(confs[mask]))
        acc = float(np.mean(ys[mask] == pred[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return ece


def best_f1_threshold(y_true: list[int], y_score: list[float]) -> dict:
    if not y_true:
        return {"threshold": float("nan"), "f1": float("nan")}
    candidates = sorted(set(float(x) for x in y_score))
    if not candidates:
        return {"threshold": float("nan"), "f1": float("nan")}
    best = {"threshold": 0.5, "f1": -1.0}
    for threshold in candidates:
        y_pred = [1 if s >= threshold else 0 for s in y_score]
        metrics = cls_metrics(y_true, y_pred)
        if metrics["f1"] > best["f1"]:
            best = {"threshold": float(threshold), "f1": float(metrics["f1"])}
    return best


def classification_from_scores(
    y_true: list[int],
    y_score: list[float],
    threshold: float = 0.5,
    *,
    ece_bins: int = 10,
) -> dict:
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    metrics = cls_metrics(y_true, y_pred)
    metrics["threshold"] = threshold
    metrics["n"] = len(y_true)
    metrics["positive_n"] = int(sum(y_true))
    metrics["negative_n"] = int(len(y_true) - sum(y_true))
    metrics["mean_ponzi_score"] = float(np.mean(y_score)) if y_score else float("nan")
    metrics["ece"] = expected_calibration_error(y_true, y_score, bins=ece_bins)
    metrics["best_f1_threshold"] = best_f1_threshold(y_true, y_score)
    if average_precision_score is not None and len(set(y_true)) > 1:
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auprc"] = float("nan")
    if roc_auc_score is not None and len(set(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["auroc"] = float("nan")
    return metrics


def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now_seconds(device: torch.device) -> float:
    synchronize_cuda(device)
    return time.perf_counter()


def quantiles(values: Iterable[float]) -> dict:
    xs = np.asarray(list(values), dtype=float)
    if xs.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(xs)),
        "median": float(np.median(xs)),
        "p90": float(np.quantile(xs, 0.90)),
        "p95": float(np.quantile(xs, 0.95)),
    }
