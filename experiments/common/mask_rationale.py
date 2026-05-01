"""从源码中移除指定语句块（用于 FD 近似：f(x) - f(x \\ S_e)）。"""
from __future__ import annotations

from utils.mask_code import remove_statement_blocks_by_id

__all__ = ["remove_statement_blocks_by_id"]
