import os
import cv2
import csv
import numpy as np
from typing import List, Dict


#-------------------------------------------------------------------------
# Compute region similarity (intersection-over-union) J
#-------------------------------------------------------------------------

def compute_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Compute the Intersection‐over‐Union between two binary masks.
    """
    gt = gt_mask > 0
    pr = pred_mask > 0
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    return 1.0 if union == 0 else inter / union

def compute_region_similarity(
    gt_root: str,
    pred_root: str,
    video_list: List[str]
) -> Dict[str, float]:
    """
    For each video in video_list, load all .png frames from
    gt_root/<video>/ and pred_root/<video>/, compute per-frame IoU,
    then return the global mean and per-video means.
    """
    all_ious = []
    per_video = {}

    for vid in video_list:
        gt_dir = os.path.join(gt_root, vid)
        pr_dir = os.path.join(pred_root, vid)

        if not os.path.isdir(gt_dir):
            print(f"  Warning: ground-truth folder missing for '{vid}', skipping.")
            continue
        if not os.path.isdir(pr_dir):
            print(f"  Warning: prediction folder missing for '{vid}', skipping.")
            continue

        frames = sorted(f for f in os.listdir(gt_dir) if f.endswith('.png'))
        vid_ious = []

        for fn in frames:
            gt_path = os.path.join(gt_dir, fn)
            pr_path = os.path.join(pr_dir, fn)

            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pr_mask = cv2.imread(pr_path, cv2.IMREAD_GRAYSCALE)

            if gt_mask is None:
                print(f"    Warning: failed to load GT '{gt_path}', skipping frame")
                continue
            if pr_mask is None:
                print(f"    Warning: failed to load PR '{pr_path}', skipping frame")
                continue

            vid_ious.append(compute_iou(gt_mask, pr_mask))
            all_ious.append(vid_ious[-1])

        per_video[vid] = float(np.mean(vid_ious)) if vid_ious else 0.0

    Jmean = float(np.mean(all_ious)) if all_ious else 0.0
    return {'Jmean': Jmean, 'per_video': per_video}

def load_vid_list(gt_root: str, split: str = "val") -> List[str]:
    """
    Read DAVIS/ImageSets/480p/{split}.txt and return one video name per line.
    """
    # gt_root is .../Annotations_unsupervised/480p
    base = gt_root.rsplit("Annotations_unsupervised", 1)[0]
    list_file = os.path.join(base, "ImageSets", f"{split}.txt")
    with open(list_file, 'r') as f:
        return [l.strip() for l in f if l.strip()]
    
#-------------------------------------------------------------------------
# Compute boundary accuracy F
#-------------------------------------------------------------------------

def mask2boundary(mask: np.ndarray) -> np.ndarray:
    """
    Given a binary mask (H×W), return its one‐pixel‐wide boundary.
    """
    # ensure uint8 0/1
    m = (mask > 0).astype(np.uint8)
    kernel = np.ones((3,3), dtype=np.uint8)
    eroded = cv2.erode(m, kernel, iterations=1)
    boundary = m - eroded
    return (boundary > 0)

def compute_boundary_f(gt_mask: np.ndarray,
                       pred_mask: np.ndarray,
                       tol: int = 1) -> float:
    """
    Compute the boundary F‐measure between two binary masks.

    tol: number of pixels to allow as “matching tolerance” when comparing
         ground‐truth vs. predicted contours.
    """
    b_gt = mask2boundary(gt_mask)
    b_pr = mask2boundary(pred_mask)

    # trivial case: no contours in both
    if b_gt.sum() == 0 and b_pr.sum() == 0:
        return 1.0

    # build structuring element for tolerance
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))

    # dilate each contour map
    b_gt_d = cv2.dilate(b_gt.astype(np.uint8), se).astype(bool)
    b_pr_d = cv2.dilate(b_pr.astype(np.uint8), se).astype(bool)

    # precision: fraction of predicted boundary pixels that hit GT
    tp_pr = np.logical_and(b_pr, b_gt_d).sum()
    P = tp_pr / b_pr.sum() if b_pr.sum() > 0 else 0.0

    # recall: fraction of GT boundary pixels hit by prediction
    tp_gt = np.logical_and(b_gt, b_pr_d).sum()
    R = tp_gt / b_gt.sum() if b_gt.sum() > 0 else 0.0

    if (P + R) == 0:
        return 0.0
    return 2 * P * R / (P + R)

def compute_boundary_accuracy(gt_root: str,
                              pred_root: str,
                              video_list: List[str],
                              tol: int = 1
                              ) -> Dict[str, object]:
    """
    Walk each video in video_list under gt_root/<vid> and pred_root/<vid>,
    compute per‐frame F, then return overall and per‐video means.
    """
    all_fs = []
    per_video = {}

    for vid in video_list:
        gt_dir = os.path.join(gt_root, vid)
        pr_dir = os.path.join(pred_root, vid)
        frames = sorted(f for f in os.listdir(gt_dir) if f.endswith('.png'))

        vid_fs = []
        for fn in frames:
            gt_path = os.path.join(gt_dir, fn)
            pr_path = os.path.join(pr_dir, fn)

            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pr_mask = cv2.imread(pr_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None or pr_mask is None:
                # skip missing frames
                continue

            f = compute_boundary_f(gt_mask, pr_mask, tol=tol)
            vid_fs.append(f)
            all_fs.append(f)

        per_video[vid] = float(np.mean(vid_fs)) if vid_fs else 0.0

    Fmean = float(np.mean(all_fs)) if all_fs else 0.0
    return {"Fmean": Fmean, "per_video": per_video}

#-------------------------------------------------------------------------
# Compute time stability T
#-------------------------------------------------------------------------

def compute_temporal_stability_frame(
    gt1: np.ndarray,
    gt2: np.ndarray,
    pr1: np.ndarray,
    pr2: np.ndarray
) -> float:
    """
    Compute one frame‐pair temporal stability T_t = 1 − |pr1⊕pr2| / |gt1∪gt2|.

    Args:
      gt1, gt2: H×W binary GT masks for two consecutive frames
      pr1, pr2: H×W binary predicted masks for those frames

    Returns:
      T_t in [0,1], with 1 perfect stability. If both GTs empty and preds
      identical, returns 1; if union(gt1,gt2)==0 but preds differ, returns 0.
    """
    b1 = gt1 > 0
    b2 = gt2 > 0
    p1 = pr1 > 0
    p2 = pr2 > 0

    # symmetric difference of preds
    diff = np.logical_xor(p1, p2).sum()
    # union of GTs
    union_gt = np.logical_or(b1, b2).sum()

    if union_gt == 0:
        # no object in either GT frame
        return 1.0 if diff == 0 else 0.0

    return 1.0 - (diff / float(union_gt))


def compute_time_stability(
    gt_root: str,
    pred_root: str,
    video_list: List[str]
) -> Dict[str, object]:
    """
    Walks through each video in video_list, loads each pair of consecutive
    frames from gt_root/<video>/*.png and pred_root/<video>/*.png,
    computes T_t for each, then returns:

      {
        'Tmean':    average T over all frame‐pairs in all videos,
        'per_video': { vid: average T for that vid, ... }
      }
    """
    all_T = []
    per_video: Dict[str, float] = {}

    for vid in video_list:
        gt_dir = os.path.join(gt_root, vid)
        pr_dir = os.path.join(pred_root, vid)
        if not os.path.isdir(pr_dir):
            # no predictions for this video
            per_video[vid] = 0.0
            continue

        # sorted list of frames
        frames = sorted(f for f in os.listdir(gt_dir) if f.endswith('.png'))
        if len(frames) < 2:
            per_video[vid] = 1.0
            continue

        T_scores = []
        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i+1]
            gt1 = cv2.imread(os.path.join(gt_dir, f1), cv2.IMREAD_GRAYSCALE)
            gt2 = cv2.imread(os.path.join(gt_dir, f2), cv2.IMREAD_GRAYSCALE)
            pr1 = cv2.imread(os.path.join(pr_dir, f1), cv2.IMREAD_GRAYSCALE)
            pr2 = cv2.imread(os.path.join(pr_dir, f2), cv2.IMREAD_GRAYSCALE)

            # skip if either file failed
            if gt1 is None or gt2 is None or pr1 is None or pr2 is None:
                continue

            t_val = compute_temporal_stability_frame(gt1, gt2, pr1, pr2)
            T_scores.append(t_val)
            all_T.append(t_val)

        per_video[vid] = float(np.mean(T_scores)) if T_scores else 0.0

    Tmean = float(np.mean(all_T)) if all_T else 0.0
    return {'Tmean': Tmean, 'per_video': per_video}



if __name__=="__main__":
    # paths must match your layout:
    GT   = "Datasets/davis/DAVIS/Annotations_unsupervised/480p"
    PR   = "outputs/pred_masks"
    vids = load_vid_list(GT, split="val")     # your existing load_vid_list

    # J metric (region similarity)
    Jres = compute_region_similarity(GT, PR, vids)
    print(f"Global J̄ = {Jres['Jmean']:.3f}")

    # F metric (boundary accuracy) at 2px tolerance
    Fres = compute_boundary_accuracy(GT, PR, vids, tol=2)
    print(f"\nGlobal F̄ = {Fres['Fmean']:.3f}")

    # T metric (time stability)
    Tres = compute_time_stability(GT, PR, vids)
    print(f"\nGlobal T̄ = {Tres['Tmean']:.3f}")

    print("\nPer‐video scores:")
    print(f"{'Video':20s}  J̄     F̄     T̄")
    print("-" * 45)
    for vid in vids:
        j = Jres['per_video'].get(vid, 0.0)
        f = Fres['per_video'].get(vid, 0.0)
        t = Tres['per_video'].get(vid, 0.0)
        print(f"{vid:20s}  {j:5.3f}  {f:5.3f}  {t:5.3f}")

    # save to CSV
    csv_path = "results/davis_metrics.csv"
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # header
        writer.writerow(["video", "J_mean", "F_mean", "T_mean"])
        # one row per video
        for vid in vids:
            j = Jres['per_video'].get(vid, 0.0)
            f = Fres['per_video'].get(vid, 0.0)
            t = Tres['per_video'].get(vid, 0.0)
            writer.writerow([vid, f"{j:.3f}", f"{f:.3f}", f"{t:.3f}"])

    print(f"\nSaved per‐video metrics to {csv_path}")
