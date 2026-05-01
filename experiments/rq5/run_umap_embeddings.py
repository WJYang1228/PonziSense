#!/usr/bin/env python3
"""
RQ5: 合同级 CLS 表征的 UMAP 二维投影（论文 Figure 对比聚类正则前后；本仓库若仅有一份 checkpoint，则做单模型可视化）。

用法:
  pip install umap-learn matplotlib  # 若未安装
  python experiments/rq5/run_umap_embeddings.py [--split test] [--max-samples 800]

输出: outputs/logs/experiments/rq5_umap_<split>.png 与 .npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.common.project import ensure_repo_importable

ensure_repo_importable()

from experiments.common.load_model import load_trained_ponzimodel

from configs.config import Config  # noqa: E402
from data.collate import collate_fn  # noqa: E402
from data.dataset import load_datasets  # noqa: E402
from utils.io import ensure_dir  # noqa: E402
from experiments.common.output_paths import figure_dir  # noqa: E402


@torch.no_grad()
def extract_cls_matrix(model, tokenizer, dataloader, device, cfg, max_samples: int):
    model.eval()
    use_amp = cfg.USE_AMP and device.type == "cuda"
    autocast = torch.autocast(device_type="cuda", enabled=use_amp)
    embs = []
    labels = []
    n = 0
    for batch in tqdm(dataloader, desc="CLS", leave=False):
        if max_samples and n >= max_samples:
            break
        input_ids = batch["input_ids"].to(device)
        position_idx = batch["position_idx"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        labs = batch["labels"].to(device)
        bsz = input_ids.size(0)
        if max_samples and n + bsz > max_samples:
            take = max_samples - n
            input_ids = input_ids[:take]
            position_idx = position_idx[:take]
            attn_mask = attn_mask[:take]
            labs = labs[:take]
            bsz = take
        with autocast:
            nodes_mask = position_idx.eq(0)
            token_mask = position_idx.ge(2)
            emb_layer = model.encoder.roberta.embeddings.word_embeddings(input_ids)
            nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
            nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask.float(), emb_layer)
            inputs_embeddings = emb_layer * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
            out = model.encoder.roberta(
                inputs_embeds=inputs_embeddings,
                attention_mask=attn_mask,
                position_ids=position_idx,
            )[0]
            cls_vec = out[:, 0, :].float().cpu().numpy()
        embs.append(cls_vec)
        labels.extend(labs.cpu().tolist())
        n += bsz
        if max_samples and n >= max_samples:
            break
    X = np.concatenate(embs, axis=0)
    y = np.array(labels, dtype=int)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val", "train"], default="test")
    ap.add_argument("--max-samples", type=int, default=800)
    args = ap.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap
    except ImportError as e:
        raise SystemExit(
            "请安装: pip install umap-learn matplotlib\n" + str(e)
        ) from e

    cfg = Config()
    model, tokenizer, _, device = load_trained_ponzimodel()
    train_set, val_set, test_set = load_datasets(tokenizer, cfg)
    if args.split == "train":
        ds = train_set
    elif args.split == "val":
        ds = val_set
    else:
        ds = test_set

    loader = DataLoader(
        ds,
        batch_size=min(8, cfg.EVAL_BATCH_SIZE),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    X, y = extract_cls_matrix(model, tokenizer, loader, device, cfg, args.max_samples)
    nn = min(15, max(2, len(X) - 1))
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=nn)
    Z = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for lab, name, c in [(0, "Non-Ponzi", "tab:blue"), (1, "Ponzi", "tab:red")]:
        m = y == lab
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.65, label=name, c=c)
    ax.legend()
    ax.set_title("UMAP of contract CLS embeddings (current checkpoint)")
    out_dir = os.path.join(cfg.OUTPUT_DIR, "logs", "experiments")
    ensure_dir(out_dir)
    png = os.path.join(out_dir, f"rq5_umap_{args.split}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    pdf = os.path.join(figure_dir(cfg, "rq5"), f"umap_{args.split}.pdf")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    npz = os.path.join(out_dir, f"rq5_embeddings_{args.split}.npz")
    np.savez(npz, X=X, y=y, Z=Z)
    print("Saved:", png, pdf, npz)


if __name__ == "__main__":
    main()
