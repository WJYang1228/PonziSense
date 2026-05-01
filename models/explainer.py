import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeExplainer(nn.Module):
    def __init__(self, node_dim, hidden_dim=128):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_emb, edge_index, edge_weight):
        if edge_index.size(1) == 0:
            return torch.zeros((0,), device=node_emb.device)

        src = node_emb[edge_index[0]]
        dst = node_emb[edge_index[1]]
        ew = edge_weight.unsqueeze(-1)
        feat = torch.cat([src, dst, ew], dim=-1)
        scores = torch.sigmoid(self.edge_mlp(feat)).squeeze(-1)
        return scores

    def sample_masks(self, scores, strategy="mixed", rho=0.5):
        """
        mixed:
          - 一部分 hard drop
          - 一部分 soft decay
        """
        if scores.numel() == 0:
            return scores

        rand = torch.rand_like(scores)
        hard = (rand < scores).float()
        soft = rho + (1 - rho) * scores
        masks = torch.where(rand > 0.5, hard, soft)
        return masks