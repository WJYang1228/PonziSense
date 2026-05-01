#!/usr/bin/env bash
# 在仓库根目录执行：训练并评估 sklearn/XGBoost 基线，写入 outputs/logs/baselines/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
pip install -q -r baseline/requirements.txt 2>/dev/null || true
python baseline/run_sklearn_baselines.py --split test --threshold 0.5 "$@"
