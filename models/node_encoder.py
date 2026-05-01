import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class NodeTextEncoder(nn.Module):
    def __init__(self, model_name="microsoft/graphcodebert-base", max_len=128,
                 pooling="cls", freeze=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_len = max_len
        self.pooling = pooling

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def encode_statements(self, list_of_statements, device):
        batch = self.tokenizer(
            list_of_statements,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        ).to(device)

        outputs = self.encoder(**batch)
        hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            emb = hidden[:, 0, :]
        else:
            mask = batch["attention_mask"].unsqueeze(-1)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return emb