"""从源码列表构造与训练一致的批量张量（用于增强视图第二路前向）。"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from configs.config import Config
from data.feature_extractor import build_attention_mask, convert_code_to_features


def batch_tensors_from_codes(
    codes: List[str],
    tokenizer: PreTrainedTokenizerBase,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = []
    position_idx = []
    attn_mask = []
    for code in codes:
        feat = convert_code_to_features(code, 0, "", tokenizer, cfg)
        m = build_attention_mask(feat, cfg, tokenizer)
        input_ids.append(feat.input_ids)
        position_idx.append(feat.position_idx)
        attn_mask.append(np.asarray(m, dtype=np.bool_))
    return (
        torch.from_numpy(np.asarray(input_ids, dtype=np.int64)),
        torch.from_numpy(np.asarray(position_idx, dtype=np.int64)),
        torch.from_numpy(np.stack(attn_mask, axis=0)),
    )
