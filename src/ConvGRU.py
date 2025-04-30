import torch 
import torch.nn as nn



class ConvGRUCell(nn.Module):
    """
    ConvGRUCell implements a convolutional GRU for spatial feature maps
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        """Gating convolution:
          concatenate input x and previous hidden state h along channels dimension (in_channels * hidden_channels)
          output: 2*hidden_channels for the reset gate r and the update gate z"""
        self.conv_gates = nn.Conv2d(
            in_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

        """Candidate state convolution:
          concatenate input x and gated previous hidden state along channels dim 
          output: hidden_channels for candidate hidden state"""
        self.conv_cand = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

    def forward(self, x, h_prev):
        """
        Forward pass for ConvGRUCell.

        Args:
            x (Tensor):     Input tensor of shape (B, in_channels, H, W).
            h_prev (Tensor):Previous hidden state of shape (B, hidden_channels, H, W).

        Returns:
            h_new (Tensor): Updated hidden state of shape (B, hidden_channels, H, W).
        """
        concat = torch.cat([x, h_prev], dim=1)  # (B, in_channels + hidden_channels, H, W)

        # Compute reset gate r and update gate z
        gates = self.conv_gates(concat)  # (B, 2 * hidden_channels, H, W)
        r, z = torch.chunk(gates, chunks=2, dim=1)  # (B, hidden_channels, H, W)
        r = self.sigmoid(r)
        z = self.sigmoid(z)

        concat_cand = torch.cat([x, r * h_prev], dim=1)  # (B, in_channels + hidden_channels, H, W)
        c = self.tanh(self.conv_cand(concat_cand))  # (B, hidden_channels, H, W)

        # New hidden state
        h_new = (1 - z) * h_prev + z * c  # (B, hidden_channels, H, W)

        return h_new
