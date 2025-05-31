import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import xml.etree.ElementTree as ET
import cv2



def bbox_xml_to_mask(xml_path, image_shape):
    """
    Reads a DAVIS/YTO-style annotation XML (axis-aligned bounding box)
    and returns a binary mask (H×W) (1 inside the box, 0 outside)
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bbox = root.find("object").find("bndbox")
    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)
    
    # Clip to valid range
    xmin = max(0, min(xmin, W - 1))
    xmax = max(0, min(xmax, W - 1))
    ymin = max(0, min(ymin, H - 1))
    ymax = max(0, min(ymax, H - 1))
    
    mask[ymin : ymax + 1, xmin : xmax + 1] = 1
    return mask


def apply_dense_crf(image, mask_prob, n_iters=10, sxy_gaussian=(3, 3), sxy_bilateral=(80, 80), srgb_bilateral=(13, 13, 13)):
    """
    image: HxWx3
    mask_prob: 2xHxW float32 array of class probabilities (foreground, background)
    returns: CRF‐refined HxW mask in [0,1]
    """
    H, W = image.shape[:2]
    d = dcrf.DenseCRF2D(W, H, 2) 

    # Unary potentials
    U = unary_from_softmax(mask_prob)  # (2, H*W)
    d.setUnaryEnergy(U)

    # Pairwise gaussian potentials (spatial smoothness)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=3)

    # Pairwise bilateral potentials (appearance)
    d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral, rgbim=image, compat=10)

    # Mean-field inference
    Q = d.inference(n_iters)  # list of two H*W arrays [bg_scores, fg_scores]
    refined = np.array(Q).reshape((2, H, W))[1]
    return refined
