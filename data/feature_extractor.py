"""
合约 → GraphCodeBERT 输入特征。

论文 ``3-new_methdology`` 中的 CFG 加权依赖图与显式边权 w_ij=α·Ctrl+β·DefUse+γ·StateImpact
在此**未**逐条建模；本模块用 ``extract_dataflow`` 得到的 DFG 与代码 token 拼接，
作为结构感知编码的近似（与 GraphCodeBERT 原始设定一致）。
"""
import torch
import numpy as np
from dataclasses import dataclass

from data.parser_utils import extract_dataflow


@dataclass
class InputFeatures:
    input_tokens: list
    input_ids: list
    position_idx: list
    dfg_to_code: list
    dfg_to_dfg: list
    label: int
    explain: str
    code: str


def convert_code_to_features(code, label, explain, tokenizer, cfg):
    code_tokens, dfg = extract_dataflow(code)

    code_tokens = [
        tokenizer.tokenize("@ " + x)[1:] if idx != 0 else tokenizer.tokenize(x)
        for idx, x in enumerate(code_tokens)
    ]

    ori2cur_pos = {-1: (0, 0)}
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (
            ori2cur_pos[i - 1][1],
            ori2cur_pos[i - 1][1] + len(code_tokens[i])
        )

    flat_code_tokens = [y for x in code_tokens for y in x]

    flat_code_tokens = flat_code_tokens[
        : cfg.CODE_LENGTH + cfg.DATA_FLOW_LENGTH - 3 - min(len(dfg), cfg.DATA_FLOW_LENGTH)
    ][:512 - 3]

    source_tokens = [tokenizer.cls_token] + flat_code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]

    dfg = dfg[: cfg.CODE_LENGTH + cfg.DATA_FLOW_LENGTH - len(source_tokens)]

    source_tokens += [x[0] for x in dfg]
    position_idx += [0 for _ in dfg]
    source_ids += [tokenizer.unk_token_id for _ in dfg]

    padding_length = cfg.CODE_LENGTH + cfg.DATA_FLOW_LENGTH - len(source_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length

    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx

    reindexed_dfg = []
    for idx, x in enumerate(dfg):
        reindexed_dfg.append(
            x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
        )

    dfg_to_dfg = [x[-1] for x in reindexed_dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in reindexed_dfg]
    dfg_to_code = [(x[0] + 1, x[1] + 1) for x in dfg_to_code]

    return InputFeatures(
        input_tokens=source_tokens,
        input_ids=source_ids,
        position_idx=position_idx,
        dfg_to_code=dfg_to_code,
        dfg_to_dfg=dfg_to_dfg,
        label=label,
        explain=explain,
        code=code,
    )


def build_attention_mask(feature, cfg, tokenizer):
    max_len = cfg.CODE_LENGTH + cfg.DATA_FLOW_LENGTH
    attn_mask = np.zeros((max_len, max_len), dtype=bool)

    node_index = sum([i > 1 for i in feature.position_idx])
    max_length = sum([i != 1 for i in feature.position_idx])

    attn_mask[:node_index, :node_index] = True

    for idx, token_id in enumerate(feature.input_ids):
        if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
            attn_mask[idx, :max_length] = True

    for idx, (a, b) in enumerate(feature.dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True

    for idx, nodes in enumerate(feature.dfg_to_dfg):
        for a in nodes:
            if a + node_index < len(feature.position_idx):
                attn_mask[idx + node_index, a + node_index] = True

    return attn_mask