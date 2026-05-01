"""论文 RQ1 表格：Precision, Recall, F1, AUC, AUPRC, MCC（需正类概率用于 AUC/AUPRC）。"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics_from_scores(
    y_true: list | np.ndarray,
    y_score_ponzi: list | np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    y_true: 0/1，1=Ponzi（与训练内部标签一致）。
    y_score_ponzi: 正类（Ponzi）概率。
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score_ponzi, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    out = {
        "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auc"] = float("nan")
        out["auprc"] = float("nan")
    return out
