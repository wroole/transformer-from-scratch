import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__()
        self.q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C)
        mask = torch.tril(torch.ones(T, T), device=x.device)
        score = score.masked_fill(mask == 0, float("-inf"))
        percent = torch.softmax(score, dim=-1)
        out = torch.matmul(percent, v)
        return out