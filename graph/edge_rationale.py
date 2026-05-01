"""
将语句级重要性分数投影到加权依赖图的边上，得到论文中的边分数 s_ij 的工程近似。

训练仍以语句 BCE / 稀疏项为主；此处用于**推理期**可审计的边列表（不引入额外可训练头）。
s_ij ∝ w_ij · (重要性(i,j) 的平滑组合)，其中重要性来自语句 explainer 的 sigmoid 分数。
"""
from __future__ import annotations

from typing import Any, List

import numpy as np

from utils.statements import StatementBlock


def project_statement_scores_to_edges(
    adj: np.ndarray,
    mask: np.ndarray,
    blocks: List[StatementBlock],
    stmt_scores: List[float],
    *,
    top_k: int = 30,
    eps: float = 1e-8,
) -> List[dict[str, Any]]:
    """
    adj: [N_max, N_max], mask: [N_max] 有效 1/0
    blocks / stmt_scores: 长度 n 语句，与 adj 前 n 行一致
    返回按 s_ij 降序的边列表。
    """
    n = min(len(blocks), len(stmt_scores), int(mask.sum()))
    if n <= 0:
        return []

    adj = np.asarray(adj, dtype=np.float64)[:n, :n]
    w_max = float(adj.max()) + eps

    out: List[dict[str, Any]] = []
    for i in range(n):
        for j in range(n):
            w = float(adj[i, j])
            if w <= 0:
                continue
            si = float(stmt_scores[i])
            sj = float(stmt_scores[j])
            geom = float(np.sqrt(max(si, 0.0) * max(sj, 0.0)))
            s_ij = geom * (w / w_max)
            bi, bj = blocks[i], blocks[j]
            out.append(
                {
                    "i": i,
                    "j": j,
                    "stmt_id_i": bi.stmt_id,
                    "stmt_id_j": bj.stmt_id,
                    "w_ij": w,
                    "s_ij": s_ij,
                    "preview_i": bi.text[:120].replace("\n", " "),
                    "preview_j": bj.text[:120].replace("\n", " "),
                }
            )

    out.sort(key=lambda x: -x["s_ij"])
    return out[:top_k]
