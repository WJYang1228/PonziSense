"""论文图表输出路径：统一放在 ``outputs/figures/<rq>/`` 与 ``outputs/tables/``。"""
from __future__ import annotations

import os

from configs.config import Config
from utils.io import ensure_dir


def figure_dir(cfg: Config, rq: str) -> str:
    ensure_dir(cfg.OUTPUT_DIR)
    p = os.path.join(cfg.FIGURES_DIR, rq)
    ensure_dir(p)
    return p


def table_dir(cfg: Config) -> str:
    ensure_dir(cfg.OUTPUT_DIR)
    ensure_dir(cfg.TABLES_DIR)
    return cfg.TABLES_DIR
