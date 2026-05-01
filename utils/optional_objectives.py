"""
论文 Sec.3 中的 L_con（监督对比 / InfoNCE 变体）、L_clu（DEC 聚类 KL）。

L_con: 对合约 CLS 嵌入做 **Supervised Contrastive**（同 batch 同类为正、异类为负），
温度 τ 对应 ``Config.TAU_CONTRASTIVE``。需 batch≥2 且至少一个 anchor 有同类正样本。

L_clu: 在 ``models.model.PonziModel.dec_clustering_loss`` 中实现，使用可学习原型与 ``NUM_CLUSTERS_K``。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    contract_cls: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    监督对比损失（Khosla et al., SupContrast 思想）：拉近同类 CLS、推远异类。
    contract_cls: [B, H]，labels: [B] long。

    当 batch 过小或单类 batch 导致无正样本时，返回 0 标量（不参与梯度欺骗）。
    """
    device = contract_cls.device
    dtype = contract_cls.dtype
    b = contract_cls.shape[0]
    if b < 2 or temperature <= 0:
        return torch.zeros((), device=device, dtype=dtype)

    z = F.normalize(contract_cls.float(), dim=1)
    sim = torch.matmul(z, z.T) / temperature
    not_self = ~torch.eye(b, dtype=torch.bool, device=device)
    # log sum_{a != i} exp(sim_ia)，数值稳定
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    sim_s = sim - sim_max
    exp_sim = torch.exp(sim_s) * not_self.float()
    log_denom = torch.log(exp_sim.sum(dim=1).clamp(min=1e-12)) + sim_max.squeeze(1)

    lab = labels.view(-1, 1)
    same = (lab == lab.T) & not_self
    n_pos = same.sum(dim=1)
    pos_sum = (same.float() * sim).sum(dim=1)
    mean_pos = pos_sum / n_pos.clamp(min=1e-12)
    per = -mean_pos + log_denom
    ok = n_pos > 0
    if not ok.any():
        return torch.zeros((), device=device, dtype=dtype)
    return per[ok].mean().to(dtype)
