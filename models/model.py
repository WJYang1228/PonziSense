"""
PonziSense 与论文 ``3-new_methdology.tex`` 对齐的工程实现。

- 编码器：GraphCodeBERT 风格（代码 + DFG）。
- L_rep = L_cls + lambda1 L_con + lambda2 L_clu; L_con supports the paper three-view InfoNCE objective
  或监督对比（``USE_SUPCON_FALLBACK``）。
- L_clu：DEC 风格 KL；软分配可用欧氏距离（论文式 39-41，``CLUSTER_USE_L2``）或余弦 DEC。
- 图分支：语句加权图 + 一层消息传递（式 79-83）与合约 CLS 融合后再分类。
- 解释头：语句级 MLP（弱监督 BCE）+ 论文式稀疏/稳定项（``EXPLAIN_LOSS_MODE``）。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from models.explain_head import StatementExplainer
from models.graph_mp import GraphMPLayer, masked_mean_pool


class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PonziModel(nn.Module):
    def __init__(self, encoder, config, num_clusters: int = 87, cfg=None):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.num_clusters = num_clusters
        self._cfg = cfg
        self.classifier = RobertaClassificationHead(config)
        self.explainer = StatementExplainer(config.hidden_size, config.hidden_dropout_prob)

        self.cluster_prototypes = nn.Parameter(
            torch.randn(num_clusters, config.hidden_size) * (1.0 / (config.hidden_size ** 0.5))
        )

        self.use_graph_branch = bool(cfg and getattr(cfg, "USE_GRAPH_BRANCH", False))
        if self.use_graph_branch:
            self.graph_mp = GraphMPLayer(config.hidden_size, config.hidden_dropout_prob)
            self.fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)
        else:
            self.graph_mp = None
            self.fusion = None

    def _dec_logits(self, contract_cls: torch.Tensor, temperature: float) -> torch.Tensor:
        """软分配 logits：L2 距离（论文）或余弦点积（旧版）。"""
        use_l2 = self._cfg is None or getattr(self._cfg, "CLUSTER_USE_L2", True)
        if use_l2:
            z = contract_cls.float()
            mu = self.cluster_prototypes.float()
            z2 = (z**2).sum(dim=1, keepdim=True)
            m2 = (mu**2).sum(dim=1)
            zm = z @ mu.T
            d2 = z2 + m2.unsqueeze(0) - 2 * zm
            return -d2 / max(temperature, 1e-6)
        z = F.normalize(contract_cls.float(), dim=-1)
        mu = F.normalize(self.cluster_prototypes.float(), dim=-1)
        return torch.matmul(z, mu.T) / max(temperature, 1e-6)

    def dec_clustering_loss(
        self,
        contract_cls: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        logits = self._dec_logits(contract_cls, temperature)
        q = F.softmax(logits, dim=-1)
        f = q.sum(dim=0, keepdim=True).clamp(min=1e-12)
        num = (q**2) / f
        p = num / num.sum(dim=1, keepdim=True).clamp(min=1e-12)
        loss = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=-1).mean()
        return loss.to(contract_cls.dtype)

    def forward(
        self,
        input_ids,
        position_idx,
        attn_mask,
        labels=None,
        graph_adj=None,
        graph_mask=None,
        statements=None,
        codes=None,
        tokenizer=None,
    ):
        return self.forward_contract(
            input_ids,
            position_idx,
            attn_mask,
            labels=labels,
            graph_adj=graph_adj,
            graph_mask=graph_mask,
            statements=statements,
            codes=codes,
            tokenizer=tokenizer,
        )

    def _roberta_outputs(
        self,
        input_ids: torch.Tensor,
        position_idx: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(input_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask.float(), inputs_embeddings)
        inputs_embeddings = (
            inputs_embeddings * (~nodes_mask)[:, :, None]
            + avg_embeddings * nodes_mask[:, :, None]
        )
        return self.encoder.roberta(
            inputs_embeds=inputs_embeddings,
            attention_mask=attn_mask,
            position_ids=position_idx,
        )[0]

    def _encode_statement_nodes_padded(
        self,
        statements_batch: Sequence[Sequence[str]],
        tokenizer,
        device: torch.device,
        max_len: int,
        contract_cls: torch.Tensor,
        n_pad: int,
    ) -> torch.Tensor:
        """返回 [B, N_pad, H]。按块前向语句 RoBERTa，避免整批语句一次塞满显存。"""
        hidden = contract_cls.size(-1)
        B = contract_cls.size(0)
        out = torch.zeros(B, n_pad, hidden, device=device, dtype=contract_cls.dtype)
        all_texts: List[str] = []
        sample_ids: List[int] = []
        pos_in_row: List[int] = []
        for bi, stmts in enumerate(statements_batch):
            if not stmts:
                continue
            for pos, st in enumerate(stmts[:n_pad]):
                all_texts.append(st)
                sample_ids.append(bi)
                pos_in_row.append(pos)
        if not all_texts:
            return out

        chunk = max(1, int(getattr(self._cfg, "GRAPH_STATEMENT_ENCODE_CHUNK", 16)))
        for start in range(0, len(all_texts), chunk):
            end = min(start + chunk, len(all_texts))
            texts = all_texts[start:end]
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            ro = self.encoder.roberta(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )[0]
            stmt_cls = ro[:, 0, :]
            for j in range(end - start):
                bi = sample_ids[start + j]
                pos = pos_in_row[start + j]
                out[bi, pos] = stmt_cls[j]
        return out

    def forward_contract(
        self,
        input_ids: torch.Tensor,
        position_idx: torch.Tensor,
        attn_mask: torch.Tensor,
        labels=None,
        graph_adj: Optional[torch.Tensor] = None,
        graph_mask: Optional[torch.Tensor] = None,
        statements: Optional[Sequence[Sequence[str]]] = None,
        codes: Optional[Sequence[str]] = None,
        tokenizer=None,
    ):
        device = input_ids.device
        outputs = self._roberta_outputs(input_ids, position_idx, attn_mask)
        contract_cls = outputs[:, 0, :]

        use_graph = (
            self.use_graph_branch
            and self.graph_mp is not None
            and self.fusion is not None
            and statements is not None
        )

        if use_graph:
            if graph_adj is None or graph_mask is None:
                if codes is None:
                    logits = self.classifier(outputs)
                else:
                    from graph.statement_graph import build_statement_graph_tensors

                    adjs = []
                    masks = []
                    for c in codes:
                        a, m, _ = build_statement_graph_tensors(str(c), self._cfg)
                        adjs.append(torch.from_numpy(a).to(device))
                        masks.append(torch.from_numpy(m).to(device))
                    graph_adj = torch.stack(adjs, dim=0)
                    graph_mask = torch.stack(masks, dim=0)
            if graph_adj is not None and graph_mask is not None and tokenizer is not None:
                max_stmt = min(
                    graph_mask.size(1),
                    int(getattr(self._cfg, "GRAPH_MAX_STATEMENTS", 32)),
                )
                stmt_max_len = int(getattr(self._cfg, "EXPLAIN_STMT_MAX_LEN", 96))
                hn = self._encode_statement_nodes_padded(
                    statements,
                    tokenizer,
                    device,
                    stmt_max_len,
                    contract_cls,
                    max_stmt,
                )
                ga = graph_adj[:, :max_stmt, :max_stmt].contiguous()
                gm = graph_mask[:, :max_stmt].contiguous()
                h1 = self.graph_mp(hn, ga, gm)
                g = masked_mean_pool(h1, gm)
                fused = self.fusion(torch.cat([contract_cls, g], dim=-1))
                logits = self.classifier(fused.unsqueeze(1))
            else:
                logits = self.classifier(outputs)
        else:
            logits = self.classifier(outputs)

        probs = F.softmax(logits, dim=-1)

        if labels is not None:
            cls_loss = CrossEntropyLoss()(logits, labels)
            return cls_loss, probs, logits, outputs
        return probs, logits, outputs

    def forward_statements(
        self,
        stmt_texts,
        tokenizer,
        device,
        max_len=96,
        contract_cls_emb=None,
    ):
        if len(stmt_texts) == 0:
            empty = torch.zeros((0,), dtype=torch.float, device=device)
            return empty, empty

        enc = tokenizer(
            stmt_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        outputs = self.encoder.roberta(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )[0]

        stmt_cls = outputs[:, 0, :]
        n = stmt_cls.size(0)
        if contract_cls_emb is None:
            contract_cls_emb = torch.zeros(n, stmt_cls.size(1), device=device, dtype=stmt_cls.dtype)
        elif contract_cls_emb.dim() == 2 and contract_cls_emb.size(0) == 1 and n > 1:
            contract_cls_emb = contract_cls_emb.expand(n, -1)
        elif contract_cls_emb.dim() == 2 and contract_cls_emb.size(0) != n:
            raise ValueError("contract_cls_emb batch dim must match number of statements")

        stmt_logits = self.explainer(stmt_cls, contract_cls_emb)
        stmt_probs = torch.sigmoid(stmt_logits)
        return stmt_logits, stmt_probs
