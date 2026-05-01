"""
论文 ``3-new_methdology.tex`` 式 (32)-(35)：跨增强视图的 InfoNCE（对称交叉熵）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_symmetric(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    z_a, z_b: [B, H]，同一合约的两路增强表示；正样本为对角线 (i,i)。
    """
    if z_a.size(0) < 2 or temperature <= 0:
        return z_a.new_zeros(())

    z_a = F.normalize(z_a.float(), dim=-1)
    z_b = F.normalize(z_b.float(), dim=-1)
    logits_ab = torch.matmul(z_a, z_b.T) / temperature
    logits_ba = logits_ab.T
    targets = torch.arange(z_a.size(0), device=z_a.device, dtype=torch.long)
    la = F.cross_entropy(logits_ab, targets)
    lb = F.cross_entropy(logits_ba, targets)
    return 0.5 * (la + lb).to(z_a.dtype)
