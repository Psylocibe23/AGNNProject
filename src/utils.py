import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral


def apply_dense_crf(image, mask_prob, n_iters=10, sxy_gaussian=(3, 3), sxy_bilateral=(80, 80), srgb_bilateral=(13, 13, 13)):
    """
    image: HxWx3 uint8 (original RGB frame)
    mask_prob: 2xHxW float32 array of class probabilities (foreground, background)
    returns: CRF‐refined HxW float32 mask in [0,1]
    """
    H, W = image.shape[:2]
    d = dcrf.DenseCRF2D(W, H, 2)  # 2 classes

    # Unary potentials
    U = unary_from_softmax(mask_prob)  # shape (2, H*W)
    d.setUnaryEnergy(U)

    # Pairwise gaussian potentials (spatial smoothness)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=3)

    # Pairwise bilateral potentials (appearance)
    d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral, rgbim=image, compat=10)

    Q = d.inference(n_iters)
    refined = np.array(Q).reshape((2, H, W))[1]
    return refined
