"""
将算法仓库根目录加入 sys.path，使 `system` 包可导入根目录下的 `predict`、
`models` 等模块（算法与系统分目录但不拆成两个 Git 子工程时使用）。
"""
from __future__ import annotations

import sys
from pathlib import Path


def algorithm_project_root() -> Path:
    """含 train.py、predict.py、models/ 的仓库根目录。"""
    return Path(__file__).resolve().parent.parent


def ensure_algorithm_on_path() -> Path:
    root = algorithm_project_root()
    r = str(root)
    if r not in sys.path:
        sys.path.insert(0, r)
    return root
