"""
论文式 (91)-(111)：L_spar、L_stab（语句级）；L_fid 在 ``utils/losses`` 中与掩码前向组合实现。
"""
from __future__ import annotations

import torch


def sparsity_loss_from_stmt_logits(stmt_logits: torch.Tensor) -> torch.Tensor:
    if stmt_logits.numel() == 0:
        return stmt_logits.new_zeros(())
    return torch.sigmoid(stmt_logits).mean()


def stability_loss_stmt_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> torch.Tensor:
    if logits_a.numel() == 0:
        return logits_a.new_zeros(())
    n = min(logits_a.numel(), logits_b.numel())
    if n == 0:
        return logits_a.new_zeros(())
    pa = torch.sigmoid(logits_a[:n])
    pb = torch.sigmoid(logits_b[:n])
    return (pa - pb).abs().mean()
