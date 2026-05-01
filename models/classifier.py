import torch.nn as nn


class PonziClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_dim, 1),
        )

    def forward(self, g_emb):
        return self.mlp(g_emb)