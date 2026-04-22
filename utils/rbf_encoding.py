# utils/rbf_encoding.py
import torch
import torch.nn as nn

class RBFEncoding(nn.Module):
    def __init__(self, num_centers=64, cutoff=10.0):
        super().__init__()
        self.num_centers = num_centers
        # 均匀取中心
        centers = torch.linspace(cutoff * 0.01, cutoff * 0.99, num_centers)
        self.register_buffer('centers', centers)       # [K]
        self.register_buffer('width', (centers[1] - centers[0]))  # σ = 相邻中心间距

    def forward(self, distances):
        # distances: [B, L, L]
        d = distances.unsqueeze(-1)  # [B, L, L, 1]
        rbf = torch.exp(-((d - self.centers)**2) / (2 * self.width**2))  # [B, L, L, K]
        rbf = rbf.masked_fill(torch.isinf(d), 0.0)  # 屏蔽无穷距离
        return rbf
