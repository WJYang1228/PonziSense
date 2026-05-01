"""从源码中移除指定语句块（FD / 扰动式解释训练）。"""
from __future__ import annotations

from utils.statements import split_solidity_statements


def remove_statement_blocks_by_id(code: str, stmt_ids: set[int]) -> str:
    blocks = split_solidity_statements(code)
    kept = [b.text for b in blocks if b.stmt_id not in stmt_ids]
    return "\n".join(kept).strip()
