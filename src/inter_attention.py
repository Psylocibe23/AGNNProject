import torch
import torch.nn as nn



class InterAttention(nn.Module):
    """
        Inter‐Attention module for loop‐edge embeddings.
        Given two inputs x and y of shape (B, C, H, W),
        computes an attention matrix e of shape (B, N=H*W, N=H*W)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_w = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, y):
        B, C, H, W = x.shape
        N = H * W
        w = self.conv_w(x).view(B, C, N)  # (B, C, N)
        wt = w.permute(0, 2, 1)
        h = y.view(B, C, N)
        e = torch.bmm(wt, h)  # (B, N, N)

        return e
    