import torch
import torch.nn as nn


class StatementExplainer(nn.Module):
    """
    语句级重要性分数（论文中为边上 s_ij；此处为可实现的简化，输入为语句 CLS 与合约 CLS 拼接）。
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, stmt_embs: torch.Tensor, contract_cls_emb: torch.Tensor) -> torch.Tensor:
        """
        stmt_embs: [N_stmt, hidden]
        contract_cls_emb: [N_stmt, hidden] (contract [CLS] repeated per statement)
        """
        x = torch.cat([stmt_embs, contract_cls_emb], dim=-1)
        return self.mlp(x).squeeze(-1)
