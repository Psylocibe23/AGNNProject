import os
import cv2
import csv
import numpy as np
from typing import List, Dict
from utils import bbox_xml_to_mask



#-------------------------------------------------------------------------
# Compute region similarity (intersection-over-union) J
#-------------------------------------------------------------------------

def compute_iou(gt_mask, pred_mask):
    """
    Compute the Intersection‐over‐Union between two binary masks
    """
    gt = gt_mask > 0
    pr = pred_mask > 0
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    return 1.0 if union == 0 else inter / union


def compute_region_similarity(gt_root, pred_root,video_list):
    """
    For each video in video_list, load all .png frames, compute per-frame IoU
    then return the global mean and per-video means
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

def load_vid_list(gt_root, split="val"):
    """
    Read DAVIS/ImageSets/480p/{split}.txt and return one video name per line
    """
    base = gt_root.rsplit("Annotations_unsupervised", 1)[0]
    list_file = os.path.join(base, "ImageSets", f"{split}.txt")
    with open(list_file, 'r') as f:
        return [l.strip() for l in f if l.strip()]
    
#-------------------------------------------------------------------------
# Compute boundary accuracy F
#-------------------------------------------------------------------------

def mask2boundary(mask):
    """
    Given a binary mask (H×W), return its one‐pixel‐wide boundary
    """
    # ensure 0/1
    m = (mask > 0).astype(np.uint8)
    kernel = np.ones((3,3), dtype=np.uint8)
    eroded = cv2.erode(m, kernel, iterations=1)
    boundary = m - eroded
    return (boundary > 0)

def compute_boundary_f(gt_mask: np.ndarray,
                       pred_mask: np.ndarray,
                       tol: int = 1) -> float:
    """
    Compute the boundary F‐measure between two binary masks

    tol: number of pixels to allow as “matching tolerance” when comparing
         ground‐truth vs. predicted contours
    """
    b_gt = mask2boundary(gt_mask)
    b_pr = mask2boundary(pred_mask)

    # trivial case
    if b_gt.sum() == 0 and b_pr.sum() == 0:
        return 1.0

    # build structuring element for tolerance
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))

    # dilate each contour map
    b_gt_d = cv2.dilate(b_gt.astype(np.uint8), se).astype(bool)
    b_pr_d = cv2.dilate(b_pr.astype(np.uint8), se).astype(bool)

    # precision: fraction of predicted boundary pixels that hit ground truth (GT)
    tp_pr = np.logical_and(b_pr, b_gt_d).sum()
    P = tp_pr / b_pr.sum() if b_pr.sum() > 0 else 0.0

    # recall: fraction of GT boundary pixels hit by prediction
    tp_gt = np.logical_and(b_gt, b_pr_d).sum()
    R = tp_gt / b_gt.sum() if b_gt.sum() > 0 else 0.0

    if (P + R) == 0:
        return 0.0
    return 2 * P * R / (P + R)

def compute_boundary_accuracy(gt_root, pred_root, video_list, tol=1):
    """
    compute per‐frame F, then return overall and per‐video means
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

def compute_temporal_stability_frame(gt1, gt2, pr1, pr2):
    """
    Compute one frame‐pair temporal stability 

    gt1, gt2: H×W binary GT masks for two consecutive frames
    pr1, pr2: H×W binary predicted masks for those frames

    Returns: T_t in [0,1], with 1 perfect stability
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


def compute_time_stability(gt_root, pred_root, video_list):
    """
    computes T_t for each video in video_list
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

# ------------------------------------------------------------------------------
# Compute J & F for YOUTUBE‐OBJECTS (YTO)
# ------------------------------------------------------------------------------

def compute_region_similarity_yto(yto_jpeg_dir, yto_xml_dir, pred_root, frame_ids):
    """
    For each frame_id (e.g. "aeroplane_00002166") in frame_ids:
      1) Extract class_name 
      2) Load JPEG
      3) Call bbox_xml_to_mask
      4) Load predicted PNG 
      5) Compute IoU(GT_mask, PR_mask)
    """
    all_ious: List[float] = []
    per_class: Dict[str, List[float]] = {}

    for frame_id in frame_ids:
        # Skip any blank lines or invalid IDs
        if "_" not in frame_id:
            continue

        class_name, _rest = frame_id.split("_", 1)

        # 2) Load JPEG to find (H, W)
        jpeg_path = os.path.join(yto_jpeg_dir, f"{frame_id}.jpg")
        if not os.path.exists(jpeg_path):
            jpeg_path = os.path.join(yto_jpeg_dir, f"{frame_id}.png")
            if not os.path.exists(jpeg_path):
                print(f"  [Warning] JPEG missing: '{frame_id}' → {jpeg_path}. Skipping.")
                continue

        img = cv2.imread(jpeg_path)
        if img is None:
            print(f"  [Warning] Failed to read JPEG: {jpeg_path}. Skipping.")
            continue

        H, W = img.shape[:2]

        # 3) Build GT mask from XML
        xml_path = os.path.join(yto_xml_dir, f"{frame_id}.xml")
        if not os.path.exists(xml_path):
            print(f"  [Warning] XML missing: '{frame_id}' → {xml_path}. Skipping.")
            continue

        # bbox_xml_to_mask(xml_path, W, H), returns binary mask of shape (H, W)
        gt_mask = bbox_xml_to_mask(xml_path, (H, W))
        if gt_mask is None:
            print(f"  [Warning] bbox_xml_to_mask failed: {xml_path}. Skipping.")
            continue

        # 4) Load predicted mask PNG
        pr_path = os.path.join(pred_root, "pred_masks", class_name, f"{frame_id}.png")
        if not os.path.exists(pr_path):
            print(f"  [Warning] PR missing: '{frame_id}' → {pr_path}. Skipping.")
            continue

        pr_mask = cv2.imread(pr_path, cv2.IMREAD_GRAYSCALE)
        if pr_mask is None:
            print(f"  [Warning] Failed to read PR: {pr_path}. Skipping.")
            continue

        # 4a) Resize to (W, H) if needed
        if pr_mask.shape[:2] != (H, W):
            pr_mask = cv2.resize(pr_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # 5) Compute IoU
        iou = compute_iou(gt_mask, pr_mask)
        all_ious.append(iou)
        per_class.setdefault(class_name, []).append(iou)

    # Compute per‐class mean
    per_class_mean: Dict[str, float] = {}
    for cls, ious in per_class.items():
        per_class_mean[cls] = float(np.mean(ious)) if ious else 0.0

    Jmean = float(np.mean(all_ious)) if all_ious else 0.0
    return {"Jmean": Jmean, "per_class": per_class_mean}


def compute_boundary_accuracy_yto(yto_jpeg_dir, yto_xml_dir, pred_root, frame_ids, tol):
    all_fs: List[float] = []
    per_class: Dict[str, List[float]] = {}

    for frame_id in frame_ids:
        if "_" not in frame_id:
            continue

        class_name, _rest = frame_id.split("_", 1)

        jpeg_path = os.path.join(yto_jpeg_dir, f"{frame_id}.jpg")
        if not os.path.exists(jpeg_path):
            jpeg_path = os.path.join(yto_jpeg_dir, f"{frame_id}.png")
            if not os.path.exists(jpeg_path):
                continue

        img = cv2.imread(jpeg_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        xml_path = os.path.join(yto_xml_dir, f"{frame_id}.xml")
        if not os.path.exists(xml_path):
            continue
        gt_mask = bbox_xml_to_mask(xml_path, (H, W))
        if gt_mask is None:
            continue

        pr_path = os.path.join(pred_root, "pred_masks", class_name, f"{frame_id}.png")
        if not os.path.exists(pr_path):
            continue
        pr_mask = cv2.imread(pr_path, cv2.IMREAD_GRAYSCALE)
        if pr_mask is None:
            continue

        if pr_mask.shape[:2] != (H, W):
            pr_mask = cv2.resize(pr_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        fscore = compute_boundary_f(gt_mask, pr_mask, tol=tol)
        all_fs.append(fscore)
        per_class.setdefault(class_name, []).append(fscore)

    per_class_mean: Dict[str, float] = {}
    for cls, fs in per_class.items():
        per_class_mean[cls] = float(np.mean(fs)) if fs else 0.0

    Fmean = float(np.mean(all_fs)) if all_fs else 0.0
    return {"Fmean": Fmean, "per_class": per_class_mean}


if __name__=="__main__":
    GT   = "Datasets/davis/DAVIS/Annotations_unsupervised/480p"
    PR   = "outputs/pred_masks"
    vids = load_vid_list(GT, split="val")

    # J (region similarity)
    Jres = compute_region_similarity(GT, PR, vids)
    print(f"Global J̄ = {Jres['Jmean']:.3f}")

    # F (boundary accuracy) at 2px tolerance
    Fres = compute_boundary_accuracy(GT, PR, vids, tol=2)
    print(f"\nGlobal F̄ = {Fres['Fmean']:.3f}")

    # T (time stability)
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

    print("=== YTO VAL METRICS ===")
    yto_jpeg_dir = "Datasets/youtube-objects/YTOdevkit/YTO/JPEGImages"
    yto_xml_dir  = "Datasets/youtube-objects/YTOdevkit/YTO/Annotations"
    pred_root    = "outputs/yto" 
    val_txt = "Datasets/youtube-objects/YTOdevkit/YTO/ImageSets/val.txt"
    with open(val_txt, "r") as f:
        frame_ids = [line.strip() for line in f if line.strip()]

    # YTO J (IoU)
    YJ = compute_region_similarity_yto(yto_jpeg_dir, yto_xml_dir, pred_root, frame_ids)
    print(f"YTO Global J̄ = {YJ['Jmean']:.3f}")
    for cls in sorted(YJ["per_class"].keys()):
        print(f"  {cls:12s}: J = {YJ['per_class'][cls]:.3f}")

    # YTO F (boundary, tol=2)
    YF = compute_boundary_accuracy_yto(yto_jpeg_dir, yto_xml_dir, pred_root, frame_ids, tol=2)
    print(f"\nYTO Global F̄ = {YF['Fmean']:.3f}")
    for cls in sorted(YF["per_class"].keys()):
        print(f"  {cls:12s}: F = {YF['per_class'][cls]:.3f}")

    # save to CSV
    yto_csv = "results/yto_metrics.csv"
    with open(yto_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["class_name", "J_mean", "F_mean"])
        for cls in sorted(YJ["per_class"].keys()):
            jval = YJ["per_class"].get(cls, 0.0)
            fval = YF["per_class"].get(cls, 0.0)
            writer.writerow([cls, f"{jval:.3f}", f"{fval:.3f}"])
    print(f"\nSaved per‐class YTO metrics to {yto_csv}")