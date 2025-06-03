import os
import yaml
import sys

sys.path.insert(0, os.path.abspath("src"))

from train import train          
from test import test            
from metrics import compute_region_similarity, compute_boundary_accuracy, compute_time_stability, load_vid_list



def main():
    # 1) load config
    cfg_path = os.path.join("configs", "default.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found at '{cfg_path}'")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2) check whether checkpoint exists, if not train the network
    ckpt_rel = cfg["test"]["checkpoint"]  
    ckpt_path = os.path.abspath(ckpt_rel)
    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint '{ckpt_path}' not found! Initializing training...")
        train()

        if not os.path.isfile(ckpt_path):
            raise RuntimeError(f"Training completed, but checkpoint '{ckpt_path}' was still not found.")
    else:
        print(f"Found existing checkpoint '{ckpt_path}', skipping training.")

    # 3) run inference on DAVIS17 and YTO (generate predictions for segmentation masks)
    print("\nRunning inference & mask‐generation…")
    test()

    # 4) Compute DAVIS17 metrics
    print("\nComputing DAVIS-17 metrics…")
    from src.metrics import (
        load_vid_list,
        compute_region_similarity,
        compute_boundary_accuracy,
        compute_time_stability
    )

    gt_root = os.path.join(cfg['davis_root'], "Annotations_unsupervised", "480p")
    pred_root = os.path.join(cfg['test']['out_dir'], "pred_masks")
    vids = load_vid_list(gt_root, split="val")

    # J (region similarity)
    Jres = compute_region_similarity(gt_root, pred_root, vids)
    print(f"Global J̄ = {Jres['Jmean']:.3f}")
    for v, jval in Jres['per_video'].items():
        print(f"{v:20s}: J = {jval:.3f}")

    # F (boundary accuracy) at 2-pixel tolerance
    Fres = compute_boundary_accuracy(gt_root, pred_root, vids, tol=2)
    print(f"\nGlobal F̄ = {Fres['Fmean']:.3f}")
    for v, fval in Fres['per_video'].items():
        print(f"{v:20s}: F = {fval:.3f}")

    # T (time stability)
    Tres = compute_time_stability(gt_root, pred_root, vids)
    print(f"\nGlobal T̄ = {Tres['Tmean']:.3f}")
    for v, tval in Tres['per_video'].items():
        print(f"{v:20s}: T = {tval:.3f}")


if __name__ == "__main__":
    main()
