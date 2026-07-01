"""
Program semantic graph constructor for PonziSense.

The paper defines a statement-aligned graph G=(V,E) whose edge weight is
w_ij = alpha*w_cfg + beta*w_dfg + gamma*w_prop. This implementation keeps the
source-level statement alignment used by the rationale benchmark while using a
lightweight static analysis that is robust on incomplete Solidity snippets.
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

from configs.config import Config
from utils.statements import is_trivial_statement, split_solidity_statements

_ID_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_\.]*\b")
_ASSIGN_RE = re.compile(r"(?<![=!<>])=(?!=)")

_KEYWORDS = {
    "address", "assert", "bool", "break", "bytes", "calldata", "constant",
    "constructor", "continue", "contract", "do", "else", "emit", "enum",
    "event", "external", "false", "for", "function", "if", "import", "int",
    "internal", "library", "mapping", "memory", "modifier", "new", "payable",
    "pragma", "private", "public", "pure", "require", "return", "returns",
    "revert", "solidity", "storage", "string", "struct", "true", "uint",
    "uint256", "view", "while",
}

_VALUE_FLOW = {
    "transfer", "send", "call", "msg.value", "address(this).balance",
    "balance", "withdraw", "deposit", "payout", "pay", "reward", "bonus",
}

_PONZI_STATE = {
    "investor", "investors", "participant", "participants", "referrer",
    "referrals", "queue", "matrix", "level", "paid", "idx", "index",
    "owner", "fee", "commission", "amount", "deposit", "reward", "payout",
}


def _tokens(text: str) -> set[str]:
    toks = {t for t in _ID_RE.findall(text) if t not in _KEYWORDS}
    lowered = text.lower()
    if "msg.value" in lowered:
        toks.add("msg.value")
    if "address(this).balance" in lowered:
        toks.add("address(this).balance")
    return toks


def _defs_uses(text: str) -> tuple[set[str], set[str]]:
    if _ASSIGN_RE.search(text):
        left, right = _ASSIGN_RE.split(text, maxsplit=1)
        return _tokens(left), _tokens(right)
    return set(), _tokens(text)


def _unit_type(text: str) -> str:
    s = text.strip().lower()
    if s.startswith("if"):
        return "branch"
    if s.startswith("for") or s.startswith("while") or s.startswith("do"):
        return "loop"
    if s.startswith("function"):
        return "function"
    if ".transfer" in s or ".send" in s or ".call" in s:
        return "value_transfer"
    if "return" in s:
        return "return"
    if _ASSIGN_RE.search(s):
        return "assignment"
    return "statement"


def _state_impact(text: str) -> float:
    s = text.lower()
    score = 0.0
    score += sum(1.0 for kw in _VALUE_FLOW if kw in s)
    score += sum(0.5 for kw in _PONZI_STATE if kw in s)
    if "msg.sender" in s:
        score += 0.75
    if "msg.value" in s:
        score += 1.25
    if ".transfer" in s or ".send" in s or ".call" in s:
        score += 1.5
    return score


def _add_edge(adj: np.ndarray, src: int, dst: int, weight: float) -> None:
    if weight <= 0 or src == dst:
        return
    # GraphMPLayer aggregates row dst from column src, so store src->dst as adj[dst, src].
    adj[dst, src] += float(weight)


def build_statement_graph_tensors(
    code: str,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Return padded adjacency, mask, and statement texts.

    Nodes are source-level statements. Edges combine approximate control-flow,
    definition-use/data-flow, and Ponzi-mechanism propagation signals as required
    by the paper's Program Semantic Graph Constructor.
    """
    max_n = int(getattr(cfg, "GRAPH_MAX_STATEMENTS", 32))
    blocks = [b for b in split_solidity_statements(code) if not is_trivial_statement(b.text)]
    blocks = blocks[:max_n]
    stmts = [b.text.strip() for b in blocks]
    n = len(stmts)

    adj = np.zeros((max_n, max_n), dtype=np.float32)
    mask = np.zeros((max_n,), dtype=np.float32)
    if n == 0:
        return adj, mask, []
    mask[:n] = 1.0

    alpha = float(getattr(cfg, "GRAPH_WEIGHT_ALPHA", 1.0))
    beta = float(getattr(cfg, "GRAPH_WEIGHT_BETA", 1.0))
    gamma = float(getattr(cfg, "GRAPH_WEIGHT_GAMMA", 1.0))

    defs_uses = [_defs_uses(s) for s in stmts]
    uses = [du[1] for du in defs_uses]
    defs = [du[0] for du in defs_uses]
    impacts = [_state_impact(s) for s in stmts]
    types = [_unit_type(s) for s in stmts]

    # CFG: sequential execution plus branch/loop approximations for distant control effects.
    for i in range(n - 1):
        _add_edge(adj, i, i + 1, alpha)
        if types[i] == "branch" and i + 2 < n:
            _add_edge(adj, i, i + 2, alpha * 0.65)
        if types[i] == "loop":
            _add_edge(adj, i + 1, i, alpha * 0.75)

    # DFG: definition-use dependencies and shared state references.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            du = defs[i] & uses[j]
            if du:
                denom = max(1, len(defs[i] | uses[j]))
                _add_edge(adj, i, j, beta * (len(du) / denom))
            shared = (uses[i] | defs[i]) & (uses[j] | defs[j]) & _PONZI_STATE
            if shared:
                _add_edge(adj, i, j, beta * 0.25 * len(shared))

    # Propagation: connect mechanism-bearing statements even when the link is not lexical.
    max_impact = max(impacts) if impacts else 0.0
    if max_impact > 0:
        for i in range(n):
            if impacts[i] <= 0:
                continue
            adj[i, i] += gamma * min(1.0, impacts[i] / max_impact)
            for j in range(i + 1, n):
                if impacts[j] <= 0:
                    continue
                distance_decay = 1.0 / (1.0 + abs(i - j))
                strength = gamma * min(1.0, (impacts[i] + impacts[j]) / (2.0 * max_impact))
                _add_edge(adj, i, j, strength * distance_decay)
                _add_edge(adj, j, i, strength * distance_decay * 0.5)

    return adj, mask, stmts
