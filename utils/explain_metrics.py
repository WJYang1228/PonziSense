"""
语句级可解释性集合指标（与弱监督标注 explain 对齐的 stmt_id 集合）。

- MIoU: 预测重要语句集合与 GT 集合的平均 IoU（|P∩G|/|P∪G|）
- MSP: 平均语句级 Precision（仅在 |P|>0 的样本上平均）
- MSR: 平均语句级 Recall（仅在 |G|>0 的样本上平均）

GT 来自 data.dataset 中 statement_labels（由 explain 文本与语句匹配得到）；
预测集合取模型对语句打分后的 Top-K（与 EXPLAIN_EVAL_TOP_K 一致）。
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from utils.amp_helpers import autocast_from_config
from utils.rationale_extractor import node_perturbation_rationales


def _iou_precision_recall(pred: set, gt: set) -> tuple[float, float, float]:
    """单样本：语句 id 集合上的 IoU / Precision / Recall。"""
    if not pred and not gt:
        return 1.0, 1.0, 1.0
    inter = len(pred & gt)
    union = len(pred | gt)
    iou = inter / union if union else 0.0
    prec = inter / len(pred) if pred else 0.0
    rec = inter / len(gt) if gt else 0.0
    return iou, prec, rec


@torch.no_grad()
def compute_explainability_macro(
    model,
    tokenizer,
    dataloader,
    device,
    cfg,
    desc: str = "Explain metrics",
):
    """
    在整集上计算 MSR / MSP / MIoU 及有效样本数。
    """
    model.eval()
    use_amp = cfg.USE_AMP and device.type == "cuda"
    autocast_cm = (
        autocast_from_config(cfg, device)
        if use_amp
        else torch.autocast(device_type="cuda", enabled=False)
    )

    top_k = max(1, int(getattr(cfg, "EXPLAIN_EVAL_TOP_K", 5)))

    ious_all = []
    precs_pred_nonempty = []
    recs_gt_nonempty = []
    n_samples = 0
    n_gt_positive = 0
    n_pred_positive = 0

    for batch in tqdm(dataloader, desc=desc, leave=False):
        bs = batch["labels"].shape[0]
        for bi in range(bs):
            stmts = batch["statements"][bi]
            labs = batch["statement_labels"][bi]
            metas = batch["statement_meta"][bi]
            if len(stmts) == 0:
                continue

            input_ids = batch["input_ids"][bi : bi + 1].to(device)
            position_idx = batch["position_idx"][bi : bi + 1].to(device)
            attn_mask = batch["attn_mask"][bi : bi + 1].to(device)

            with autocast_cm:
                # 必须与 input_ids 的 batch=1 一致：只传入当前样本的图与语句，否则
                # _encode_statement_nodes_padded 会按整批 statements 建索引而 contract_cls 仅为 1。
                _, _, outputs = model.forward_contract(
                    input_ids,
                    position_idx,
                    attn_mask,
                    labels=None,
                    graph_adj=batch["graph_adj"][bi : bi + 1].to(device),
                    graph_mask=batch["graph_mask"][bi : bi + 1].to(device),
                    statements=[batch["statements"][bi]],
                    codes=[batch["codes"][bi]],
                    tokenizer=tokenizer,
                )
                graph_adj = batch["graph_adj"][bi : bi + 1].to(device)
                graph_mask = batch["graph_mask"][bi : bi + 1].to(device)
                if getattr(cfg, "EXPLAIN_EVAL_USE_PERTURBATION", True):
                    items = node_perturbation_rationales(
                        model,
                        tokenizer,
                        input_ids,
                        position_idx,
                        attn_mask,
                        graph_adj,
                        graph_mask,
                        stmts,
                        metas,
                        batch["codes"][bi],
                        top_k=top_k,
                    )
                    pred_ids = {item["stmt_id"] for item in items}
                else:
                    contract_cls = outputs[:, 0, :]
                    n = len(stmts)
                    cexp = contract_cls.expand(n, -1)
                    _, stmt_probs = model.forward_statements(
                        stmts,
                        tokenizer=tokenizer,
                        device=device,
                        max_len=cfg.EXPLAIN_STMT_MAX_LEN,
                        contract_cls_emb=cexp,
                    )
                    scores = stmt_probs.float().cpu().tolist()
                    k = min(top_k, len(stmts))
                    order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:k]
                    pred_ids = {metas[j].stmt_id for j in order}

            labs_f = labs.float().cpu().tolist()
            gt_ids = {metas[j].stmt_id for j in range(len(metas)) if labs_f[j] >= 0.5}

            iou, prec, rec = _iou_precision_recall(pred_ids, gt_ids)
            ious_all.append(iou)
            n_samples += 1

            if len(gt_ids) > 0:
                n_gt_positive += 1
                recs_gt_nonempty.append(rec)
            if len(pred_ids) > 0:
                n_pred_positive += 1
                precs_pred_nonempty.append(prec)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "MIoU": _mean(ious_all),
        "MSP": _mean(precs_pred_nonempty),
        "MSR": _mean(recs_gt_nonempty),
        "explain_n_samples": n_samples,
        "explain_n_gt_positive": n_gt_positive,
        "explain_n_pred_positive": n_pred_positive,
        "EXPLAIN_EVAL_TOP_K": top_k,
    }
