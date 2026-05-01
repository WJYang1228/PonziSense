"""训练/评估共用的混合精度设置（与 configs.config 对齐）。"""
from __future__ import annotations

import torch


def autocast_from_config(cfg, device: torch.device):
    """USE_AMP + USE_BF16_AMP 决定 autocast；bf16 时不用 GradScaler。"""
    use_amp = getattr(cfg, "USE_AMP", False) and device.type == "cuda"
    if not use_amp:
        return torch.autocast(device_type="cuda", enabled=False)
    use_bf16 = getattr(cfg, "USE_BF16_AMP", False) and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    return torch.autocast(device_type="cuda", enabled=True, dtype=dtype)


def grad_scaler_from_config(cfg, device: torch.device):
    use_amp = getattr(cfg, "USE_AMP", False) and device.type == "cuda"
    if not use_amp:
        return None
    use_bf16 = getattr(cfg, "USE_BF16_AMP", False) and torch.cuda.is_bf16_supported()
    if use_bf16:
        return None
    return torch.amp.GradScaler("cuda", enabled=True)
