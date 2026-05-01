"""在数据集上批量前向，收集标签与 Ponzi 概率。"""
from __future__ import annotations

import torch
from tqdm import tqdm


@torch.no_grad()
def collect_ponzi_probs_and_labels(model, tokenizer, dataloader, device, cfg):
    model.eval()
    use_amp = cfg.USE_AMP and device.type == "cuda"
    autocast = torch.autocast(device_type="cuda", enabled=use_amp)
    y_true = []
    y_score = []
    for batch in tqdm(dataloader, desc="Forward", leave=False):
        input_ids = batch["input_ids"].to(device)
        position_idx = batch["position_idx"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast:
            probs, _, _ = model.forward_contract(
                input_ids,
                position_idx,
                attn_mask,
                labels=None,
                graph_adj=batch["graph_adj"].to(device),
                graph_mask=batch["graph_mask"].to(device),
                statements=batch["statements"],
                codes=batch["codes"],
                tokenizer=tokenizer,
            )
        ponzi_prob = probs[:, 1].float().cpu().tolist()
        y_score.extend(ponzi_prob)
        y_true.extend(labels.cpu().tolist())
    return y_true, y_score
