"""
Build the paper's semantic-preserving contract views for contrastive learning.
"""
from __future__ import annotations

import random
import re

from configs.config import Config


def augment_contract_code(code: str, cfg: Config | None = None) -> str:
    """Backward-compatible single augmented view."""
    return build_semantic_views(code, cfg, num_views=2)[1]


def build_semantic_views(code: str, cfg: Config | None = None, num_views: int | None = None) -> list[str]:
    """
    Return semantic-preserving views x^(1), x^(2), x^(3).

    x^(1) is the original source. Additional views combine benign identifier
    masking, whitespace perturbation, and optional trivial-statement dropout.
    These transformations are intentionally conservative so Solidity remains
    parseable for GraphCodeBERT/DFG extraction.
    """
    cfg = cfg or Config()
    n = int(num_views or getattr(cfg, "SEMANTIC_VIEW_COUNT", 3))
    views = [code]

    if n >= 2:
        v2 = code
        if getattr(cfg, "AUGMENT_IDENTIFIER_MASK", True):
            v2 = _mask_identifiers(v2, p=float(getattr(cfg, "AUGMENT_ID_MASK_PROB", 0.12)))
        views.append(v2 if v2.strip() else code)

    if n >= 3:
        v3 = code
        if getattr(cfg, "AUGMENT_BLANK_LINES", True):
            v3 = _insert_blank_lines(v3, p=float(getattr(cfg, "AUGMENT_BLANK_PROB", 0.35)))
        if getattr(cfg, "AUGMENT_STMT_DROP", False):
            v3 = _maybe_drop_trivial_statements(v3, p=float(getattr(cfg, "AUGMENT_STMT_DROP_PROB", 0.08)))
        views.append(v3 if v3.strip() else code)

    while len(views) < n:
        views.append(code)
    return views[:n]


def _mask_identifiers(code: str, p: float) -> str:
    def repl(m: re.Match) -> str:
        if random.random() > p:
            return m.group(0)
        return f"_{m.group(0)}_"

    return re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", repl, code)


def _insert_blank_lines(code: str, p: float) -> str:
    lines = code.split("\n")
    out = []
    for line in lines:
        out.append(line)
        if random.random() < p and line.strip().endswith(";"):
            out.append("")
    return "\n".join(out)


def _maybe_drop_trivial_statements(code: str, p: float) -> str:
    lines = code.split("\n")
    kept = []
    for line in lines:
        st = line.strip()
        if st in {"", "{", "}"} and random.random() < p:
            continue
        kept.append(line)
    return "\n".join(kept) if kept else code
