import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load full DeepLabV3 model
        dlv3 = deeplabv3_resnet50(pretrained=pretrained, progress=True)
        # Take resnet-50 backbone (conv1, layer1 to layer4)
        self.backbone = dlv3.backbone
        # Pyramid Pooling (ASPP) module
        self.aspp = dlv3.classifier[0]  # Outputs 256 channels
        self.conv = dlv3.classifier[1]  # 3x3 convolution
        self.bn_relu = nn.Sequential(
            dlv3.classifier[2],
            dlv3.classifier[3]
        )  # Batch normalization + Relu

    def forward(self, x):
        """
            given as input a batch of video frames x = (B, C, H, W), returns the nodes embeddings
            each node is a video frame in our AGNN
        """
        features = self.backbone(x)['out']  # (B, 2048, H/8, W/8)
        features = self.aspp(features)  # (B, 256, H/8, W/8)
        features = self.conv(features)  # (B, 256, H/8, W/8)
        features = self.bn_relu(features)  # (B, 256, H/8, W/8)
        return features


if __name__=='__main__':
    # 1) Instantiate the model
    model = deeplabv3_resnet50(pretrained=True)
    
    # 2) Print the entire architecture
    print(model)