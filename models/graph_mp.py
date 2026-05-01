"""
论文式 (79)-(83)：加权消息传递的一层近似实现（批内 padding）。
h_i^{(l+1)} = σ(W1 h_i^{(l)} + Σ_j w_ji W2 h_j^{(l)})
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def row_normalize_adj(adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """adj [B,N,N], mask [B,N]；仅在有效节点上归一化入边。"""
    # 对每行 i（作为目标），按入边求和归一化
    b, n, _ = adj.shape
    m = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B,N,N]
    a = adj * m
    denom = a.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return a / denom


class GraphMPLayer(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h: [B, N, H], adj: [B, N, N] 非负权重, mask: [B, N] 1/0
        """
        a = row_normalize_adj(adj, mask)
        agg = torch.bmm(a, h)
        out = F.relu(self.drop(self.w1(h) + self.w2(agg)))
        out = out * mask.unsqueeze(-1)
        return out


def masked_mean_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """h [B,N,H], mask [B,N] -> [B,H]；全 padding 样本输出 0。"""
    w = mask.unsqueeze(-1)
    s = (h * w).sum(dim=1)
    d = mask.sum(dim=1, keepdim=True)
    out = s / d.clamp(min=1e-6)
    dead = (d.squeeze(-1) <= 0).unsqueeze(-1)
    return torch.where(dead, torch.zeros_like(out), out)
