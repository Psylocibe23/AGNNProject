import torch 
import torch.nn as nn



class IntraAttention(nn.Module):
    """
        Intra-Attention module for loop-edge embedding.
        Given an input x of shape (B, C, H, W), computes
        out = alpha * softmax(f(x) @ h(x).T) @ l(x) + x
        where f, h, l are 1x1 convolutions and alpha is a lernable scalar
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_f = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_h = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_l = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # alpha statrs at 0
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        f = self.conv_f(x).view(B, C, -1)  # (B, C, N = H*W) we flatten spatial dimension to pay attention over all positions
        h = self.conv_h(x).view(B, C, -1)  # (B, C, N)
        l = self.conv_l(x).view(B, C, -1)  # (B, C, N)

        # Compute attention scores
        ft = f.permute(0, 2, 1)  # (B, N, C)
        attn = torch.bmm(ft, h)  # f @ h.t  (B, N, N)
        attn = torch.softmax(attn, dim=-1)  

        lt = l.permute(0, 2, 1) # (B, N, C)
        out = torch.bmm(attn, lt)  # (B, N, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)

        return self.alpha * out + x 
    