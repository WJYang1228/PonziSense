from __future__ import annotations

import os

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from configs.config import Config
from models.model import PonziModel


def load_trained_ponzimodel(device: torch.device | None = None):
    cfg = Config()
    device = device or torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
    hf_config = RobertaConfig.from_pretrained(cfg.CONFIG_NAME)
    encoder = RobertaForSequenceClassification.from_pretrained(cfg.MODEL_NAME, config=hf_config)
    model = PonziModel(encoder, hf_config, num_clusters=cfg.NUM_CLUSTERS_K, cfg=cfg).to(device)
    ckpt = os.path.join(cfg.CKPT_DIR, "best_model.pt")
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, tokenizer, cfg, device
