import torch
import torch.nn as nn
import torch.nn.functional as F



class MessageAggregation(nn.Module):
    """
    Given a list of messages [m1, m2, ..., mK] each of shape (B,C,H,W),
    applies a channel-wise gate g(m_i) and returns the sum:
        M = sum_i g(m_i) * m_i
    """
    def __init__(self, in_channels):
        super().__init__()
        # The gating function G: Conv -> Global Average Pooling -> sigmoid
        self.gate_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gate_bias = nn.Parameter(torch.zeros(in_channels))

    def forward(self, messages):
        agg = 0

        for m in messages:
            x = self.gate_conv(m) + self.gate_bias.view(1, -1, 1, 1)  # (B, C, H, W)
            g = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # (B, C)
            g = torch.sigmoid(g).view(m.size(0), -1, 1, 1)  # (B, C, 1, 1)
            # Channel wise Hadamard product
            gated_message = g * m  # (B, C, H, W)
            agg = agg + gated_message

        return agg
    