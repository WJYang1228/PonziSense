"""
将仓库根目录加入 sys.path。

用 ``python experiments/rq1/xxx.py`` 运行时，解释器默认只把脚本所在目录加入 path，
无法解析 ``import experiments``。入口脚本须先调用
``bootstrap_experiments_path(__file__)``，再 ``from experiments.common...``。
"""
from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_experiments_path(entry_file: str) -> Path:
    """
    entry_file 传 ``__file__``。脚本在 ``experiments/<子目录>/xxx.py`` 或
    ``experiments/case_study/xxx.py`` 时，向上两级均为仓库根。
    """
    root = Path(entry_file).resolve().parents[2]
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def repo_root() -> Path:
    """本文件位于 experiments/common/project.py，parents[2] 为仓库根。"""
    return Path(__file__).resolve().parents[2]


def ensure_repo_importable() -> Path:
    root = repo_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root
