"""Explanation loss for source-level rationales: weak supervision plus paper-style sparsity/stability."""
import torch
import torch.nn.functional as F

from utils.perturbation_losses import sparsity_loss_from_stmt_logits, stability_loss_stmt_logits


def compute_explainer_loss(
    model,
    tokenizer,
    batch,
    device,
    cfg,
    contract_outputs,
    contract_outputs_b=None,
):
    """
    contract_outputs_b: 增强视图的 encoder 输出；若提供则计算 L_stab。
    """
    cls_emb = contract_outputs[:, 0, :]
    cls_emb_b = contract_outputs_b[:, 0, :] if contract_outputs_b is not None else None

    all_stmt_texts = []
    all_labels = []
    all_contract_cls = []
    all_contract_cls_b = []
    sample_ids = []
    sid = 0

    for i, (stmt_texts, stmt_labels) in enumerate(zip(batch["statements"], batch["statement_labels"])):
        if len(stmt_texts) == 0:
            continue
        stmt_labels = stmt_labels.to(device)
        c = cls_emb[i]
        cb = cls_emb_b[i] if cls_emb_b is not None else None
        for st, sl in zip(stmt_texts, stmt_labels):
            all_stmt_texts.append(st)
            all_labels.append(sl)
            all_contract_cls.append(c)
            if cls_emb_b is not None:
                all_contract_cls_b.append(cb)
            sample_ids.append(sid)
        sid += 1

    if not all_stmt_texts:
        return torch.tensor(0.0, device=device)

    all_labels = torch.stack(all_labels).float()
    all_contract_cls = torch.stack(all_contract_cls, dim=0)
    if cls_emb_b is not None:
        all_contract_cls_b_t = torch.stack(all_contract_cls_b, dim=0)
    else:
        all_contract_cls_b_t = None
    sample_ids = torch.tensor(sample_ids, device=device, dtype=torch.long)
    chunk = max(1, int(getattr(cfg, "EXPLAIN_FORWARD_CHUNK", 128)))
    elem_losses = []
    stmt_logits_parts = []
    stmt_logits_b_parts = []

    cuda_clear = device.type == "cuda" and getattr(cfg, "EXPLAIN_EMPTY_CUDA_CACHE", False)

    for start in range(0, len(all_stmt_texts), chunk):
        end = min(start + chunk, len(all_stmt_texts))
        texts = all_stmt_texts[start:end]
        labels = all_labels[start:end]
        ccls = all_contract_cls[start:end]

        if all_contract_cls_b_t is not None:
            cclsb = all_contract_cls_b_t[start:end]
            texts2 = texts + texts
            ccls2 = torch.cat([ccls, cclsb], dim=0)
            stmt_logits2, _ = model.forward_statements(
                texts2,
                tokenizer=tokenizer,
                device=device,
                max_len=cfg.EXPLAIN_STMT_MAX_LEN,
                contract_cls_emb=ccls2,
            )
            n = len(texts)
            stmt_logits = stmt_logits2[:n]
            stmt_logits_b = stmt_logits2[n:]
            stmt_logits_parts.append(stmt_logits)
            stmt_logits_b_parts.append(stmt_logits_b)
        else:
            stmt_logits, _ = model.forward_statements(
                texts,
                tokenizer=tokenizer,
                device=device,
                max_len=cfg.EXPLAIN_STMT_MAX_LEN,
                contract_cls_emb=ccls,
            )
            stmt_logits_parts.append(stmt_logits)

        le = F.binary_cross_entropy_with_logits(stmt_logits, labels, reduction="none")
        elem_losses.append(le)

        if cuda_clear:
            torch.cuda.empty_cache()

    elem_loss = torch.cat(elem_losses, dim=0)
    stmt_logits_cat = torch.cat(stmt_logits_parts, dim=0)

    unique = torch.unique(sample_ids)
    per_sample = []
    for u in unique:
        mask = sample_ids == u
        per_sample.append(elem_loss[mask].mean())
    bce = torch.stack(per_sample).mean()

    mode = getattr(cfg, "EXPLAIN_LOSS_MODE", "both")
    if mode == "bce":
        return bce

    l_spar = sparsity_loss_from_stmt_logits(stmt_logits_cat)
    l_stab = stmt_logits_cat.new_zeros(())
    if cls_emb_b is not None and stmt_logits_b_parts:
        stmt_b_cat = torch.cat(stmt_logits_b_parts, dim=0)
        l_stab = stability_loss_stmt_logits(stmt_logits_cat, stmt_b_cat)

    ls = float(getattr(cfg, "LAMBDA_SPAR", 1.0))
    lt = float(getattr(cfg, "LAMBDA_STAB", 0.1))
    paper_core = ls * l_spar + lt * l_stab

    if mode == "paper":
        return paper_core

    w_bce = float(getattr(cfg, "EXPLAIN_BCE_IN_PAPER_MODE", 0.5))
    return w_bce * bce + (1.0 - w_bce) * paper_core
