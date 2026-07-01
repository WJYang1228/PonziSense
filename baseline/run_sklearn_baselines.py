#!/usr/bin/env python3
"""
在 Ponzi-E 风格划分（train/val/test CSV）上训练经典基线，与 RQ1 相同指标输出 JSON。

对应论文命名（见 baseline/README.md）：
  Ridge-NC, SVM-NC, XGBoost-TF-IDF

用法（仓库根目录）::
    python baseline/run_sklearn_baselines.py --split test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC

from baseline.features import make_count_vectorizer, make_tfidf_vectorizer
from configs.config import Config  # noqa: E402
from experiments.common.classification_metrics import binary_metrics_from_scores  # noqa: E402
from utils.io import ensure_dir  # noqa: E402


def _load_split(path: str, positive_label: int) -> tuple[pd.Series, np.ndarray]:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    if "code" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV 需含 code, label: {path}")
    texts = df["code"].astype(str)
    y = (df["label"].astype(int) == int(positive_label)).astype(np.int32).values
    return texts, y


def _ridge_classifier():
    try:
        return RidgeClassifier(class_weight="balanced", random_state=42)
    except TypeError:
        return RidgeClassifier(random_state=42)


def run_ridge_nc(X_train, y_train, X_test, max_features: int) -> np.ndarray:
    vec = make_count_vectorizer(max_features=max_features)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    base = _ridge_classifier()
    cv = min(5, max(2, len(y_train) // 400))
    clf = CalibratedClassifierCV(base, cv=cv, method="sigmoid")
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]


def run_svm_nc(X_train, y_train, X_test, max_features: int) -> np.ndarray:
    vec = make_count_vectorizer(max_features=max_features)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    base = LinearSVC(class_weight="balanced", dual=False, max_iter=8000, random_state=42)
    cv = min(5, max(2, len(y_train) // 400))
    clf = CalibratedClassifierCV(base, cv=cv, method="sigmoid")
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]


def run_xgb_tfidf(X_train, y_train, X_test, max_features: int) -> np.ndarray:
    import xgboost as xgb

    vec = make_tfidf_vectorizer(max_features=max_features)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = (neg / max(1, pos)) if pos else 1.0
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1,
        reg_lambda=1.0,
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=float(spw),
        eval_metric="logloss",
    )
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-features", type=int, default=40000)
    ap.add_argument(
        "--only",
        type=str,
        default="ridge,svm,xgb",
        help="逗号分隔: ridge,svm,xgb",
    )
    args = ap.parse_args()

    cfg = Config()
    pos = cfg.POSITIVE_LABEL
    te_path = cfg.TEST_PATH if args.split == "test" else cfg.VAL_PATH

    X_train, y_train = _load_split(tr_path, pos)
    X_test, y_test = _load_split(te_path, pos)

    want = {x.strip().lower() for x in args.only.split(",") if x.strip()}
    rows_out = []

    if "ridge" in want:
        print("Training Ridge-NC ...", flush=True)
        scores = run_ridge_nc(X_train, y_train, X_test, args.max_features)
        m = binary_metrics_from_scores(y_test, scores, threshold=args.threshold)
        rows_out.append({"paper_name": "Ridge-NC", "impl_id": "ridge_nc_char_ngram", "metrics": m})

    if "svm" in want:
        print("Training SVM-NC ...", flush=True)
        scores = run_svm_nc(X_train, y_train, X_test, args.max_features)
        m = binary_metrics_from_scores(y_test, scores, threshold=args.threshold)
        rows_out.append({"paper_name": "SVM-NC", "impl_id": "svm_nc_char_ngram", "metrics": m})

    if "xgb" in want:
        try:
            print("Training XGBoost-TF-IDF ...", flush=True)
            scores = run_xgb_tfidf(X_train, y_train, X_test, args.max_features)
            m = binary_metrics_from_scores(y_test, scores, threshold=args.threshold)
            rows_out.append(
                {"paper_name": "XGBoost-TF-IDF", "impl_id": "xgboost_tfidf_char_ngram", "metrics": m}
            )
        except ImportError:
            print("Skip XGBoost: pip install xgboost", file=sys.stderr)

    out = {
        "note": "Classic sklearn baselines aligned with paper names; not bit-identical to Zheng et al. features.",
        "split": args.split,
        "threshold": args.threshold,
        "n_train": int(len(y_train)),
        "n_eval": int(len(y_test)),
        "positive_label_raw": pos,
        "max_features": args.max_features,
        "baselines": rows_out,
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))
    out_dir = os.path.join(cfg.OUTPUT_DIR, "logs", "baselines")
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"sklearn_baselines_{args.split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved:", path)


if __name__ == "__main__":
    main()
