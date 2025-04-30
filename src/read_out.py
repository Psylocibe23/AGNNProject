import torch
import torch.nn as nn
import torch.nn.functional as F



class ReadOut(nn.Module):
    """
    ReadOut head: a fully convolutional network that maps node embeddings to segmentation masks

    Architecture:
      1) 3×3 conv → BatchNorm → ReLU
      2) 3×3 conv → BatchNorm → ReLU
      3) 1×1 conv → Sigmoid
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h, v):
        # Concatenate the final node state with the original node state for skip connections
        if v is not None:
            x = torch.cat([h, v], dim=1)  # (B, 2*C, H, W)
        else:
            x = h  # (B, C, H, W)

        x = self.conv1(x)   # (B, in_channels, H, W)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)   # (B, in_channels, H, W)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)  # (B, 1, H, W)
        segmentation_mask = self.sigmoid(x)

        return segmentation_mask
    