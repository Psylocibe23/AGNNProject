import torch
import torch.nn as nn
from backbone import DeepLabV3Backbone
from intra_attention import IntraAttention
from inter_attention import InterAttention
from message_aggregation import MessageAggregation
from ConvGRU import ConvGRUCell
from read_out import ReadOut



class AGNN(nn.Module):
    """
    Attentive Graph Neural Network for Zero-Shot Video Object Segmentation
    Architecture:
    1) DeepLabV3 backbone extracts node embeddings v_i
    2) Intra attention mechanism computes loop-edge embeddings e_{i,i}
    3) Inter attention mechanism compute line-edge embeddings e_{i,j}
    4) Messages m_{j,i} = softmax(e_{i,j}) @ h_j are computed
    5) Gated message aggregation 
    6) ConvGRU cell updates the node hidden state h_{i}
    7) After K iterations, the read-out function produces segmentation mask
    (Loss: Weighted binary cross entropy)
    """
    def __init__(self, hidden_channels=256, num_iterations=3, pretrained_backbone=True):
        super().__init__()
        self.num_iterations = num_iterations

        # 1) DeepLabV3
        self.backbone = DeepLabV3Backbone(pretrained=pretrained_backbone)

        # 2) Intra-attention
        self.intra_att = IntraAttention(in_channels=hidden_channels)

        # 3) Inter-attention
        self.inter_att = InterAttention(in_channels=hidden_channels)

        # 4) Message aggregator
        self.message_agg = MessageAggregation(in_channels=hidden_channels)

        # 5) ConvGRU
        self.conv_gru = ConvGRUCell(in_channels=hidden_channels, hidden_channels=hidden_channels)

        # 6) Read-out head
        self.read_out = ReadOut(in_channels=2*hidden_channels)

    def forward(self, frames):
        """
        Frames: Tensor of shape (B, N, 3, H, W) where N = number of frames (nodes)

        Returns:
          masks: Tensor of shape (B, N, 1, H', W') segmentation masks per frame
        """
        B, N, C, H, W = frames.shape

        # Extract initial node embeddings via backbone
        x = frames.view(B * N, C, H, W)  # (B*N, 3, H, W)
        v = self.backbone(x)  # (B*N, hidden, H',W')
        _, hidden, Hf, Wf = v.shape
        v = v.view(B, N, hidden, Hf, Wf)

        # Initialize node state v = h_0
        h = v.clone()

        # Message passing iterations
        for t in range(self.num_iterations):
            h_new = torch.zeros_like(h)
            for i in range(N):
                h_i = h[:, i]  # slice the i-th node
                # Intra-attention
                loop_msg = self.intra_att(h_i)
                messages = [loop_msg]
                # Inter-attention
                for j in range(N):
                    if i == j:
                        continue
                    h_j = h[:, j]  # (B, hidden, Hf,Wf)
                    e_ij = self.inter_att(h_i, h_j)  # (B, Hf*Wf, Hf*Wf)
                    attn = torch.softmax(e_ij, dim=-1)  # (B, H*W, H*W)
                    hj_flat = h_j.view(B, hidden, -1)
                    m_flat = torch.bmm(attn, hj_flat.permute(0,2,1))  # (B, H*W, hidden)
                    m = m_flat.permute(0,2,1).contiguous().view(B, hidden, Hf, Wf)
                    messages.append(m)

                # Gated message aggregation
                m_i = self.message_agg(messages)  # (B, hidden, Hf,Wf)

                # ConvGRU cell
                h_prev = h_i
                h_i_new = self.conv_gru(m_i, h_prev)  # (B, hidden, Hf,Wf)
                h_new[:, i] = h_i_new

            h = h_new

        # Read-out
        masks = []
        for i in range(N):
            hi = h[:, i]       # (B,hidden,Hf,Wf)
            vi = v[:, i]       # (B,hidden,Hf,Wf)
            mask = self.read_out(hi, vi)  # (B,1,Hf,Wf)
            masks.append(mask)

        masks = torch.stack(masks, dim=1)                # (B,N,1,Hf,Wf)
        return masks
    
if __name__=="__main__":
    model = AGNN(hidden_channels=256, num_iterations=3)
    dummy = torch.randn(2, 5, 3, 473, 473)  # e.g. B=2, N=5 frames of 473×473
    masks = model(dummy)                   # expect (2,5,1,60,60)
    print(masks.shape)
