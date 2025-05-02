import torch
import torch.nn.functional as F



def weighted_bce_loss(preds, targets, eps=1e-6):
    """
    Compute weighted binary cross-entropy loss 
        L = - sum_{x}^{H*W} [(1 - eta) * S_x * log(p_x) + eta * (1 - S_x) * log(1 - p_x)]
        Where:
        eta = (# foreground pixels) / (total # pixels) in each sample
        p_x = preds, predicted probability that pixel x is foreground
        S_x = targets, ground-truth mask

    Returns: Scalar loss averaged over the batch
    """
    # avoid log(0)
    preds = preds.clamp(min=eps, max=1 - eps)

    B, _, H, W = targets.shape
    N = H * W

    fg_ratio = targets.view(B, -1).sum(dim=1) # (B,), is one scalar per image
    eta    = fg_ratio / N 

    # Compute weights for foreground and background
    # Reshape in order to multiply element-wise against the prediciton tensor
    w_forg = (1.0 - eta).view(B, 1, 1, 1)           
    w_back = eta.view(B, 1, 1, 1)   

    loss = - (w_forg * targets * torch.log(preds) + w_back * (1.0 - targets) * torch.log(1.0 - preds))

    # Return the mean loss over all pixels and batch
    return loss.mean()
