import os
import argparse
import yaml
import torch
import cv2
import numpy as np
from agnn import AGNN
from dataloader import VideoSegDataset
from utils import apply_dense_crf, bbox_xml_to_mask
import math
import xml.etree.ElementTree as ET
import glob



def load_config(config_path="configs/default.yaml"):
    """
    Load the YAML configuration file
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(cfg, checkpoint_path=None):
    """
    Instantiate AGNN, load weights from checkpoint, and switch to eval mode
    """
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

    model = AGNN(
        hidden_channels=cfg['model']['hidden_channels'],
        num_iterations=cfg['model']['num_iterations']
    ).to(device)

    checkpoint_path = checkpoint_path or cfg['test']['checkpoint']
    print(f"Loading checkpoint from {checkpoint_path}")

    chkpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state' in chkpt:
        state_dict = chkpt['model_state']
    elif 'model_state_dict' in chkpt:
        state_dict = chkpt['model_state_dict']
    else:
        state_dict = chkpt

    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def process_davis_videos(cfg, model, device):
    """
    Inference on DAVIS val split and save masks
    """
    print("\nProcessing DAVIS validation videos …")
    N0 = cfg['num_frames_test']

    ds = VideoSegDataset(
        root_dir = cfg['davis_root'],
        split = 'val',
        num_frames = N0,
        train_mode = False
    )
    frame_H, frame_W = ds.frame_size

    out_dir = cfg['test']['out_dir']
    davis_pred_root = os.path.join(out_dir, 'pred_masks')
    os.makedirs(davis_pred_root, exist_ok=True)

    for img_paths, _ in ds.samples:
        seq_name = os.path.basename(os.path.dirname(img_paths[0]))
        seq_out = os.path.join(davis_pred_root, seq_name)
        os.makedirs(seq_out, exist_ok=True)

        total = len(img_paths)
        T = math.ceil(total / N0)
        if T < 1:
            print(f"Skipping DAVIS sequence '{seq_name}' (only {total} frames).")
            continue

        print(f"Davis '{seq_name}': {T} subsets of {N0} frames each")
        for t in range(T):
            indices = [min(t + k * T, total - 1) for k in range(N0)]
            frames = []
            origs = []

            for idx in indices:
                path = img_paths[idx]
                bgr = cv2.imread(path)
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                origs.append(img)

                img_resized = cv2.resize(img, (frame_W, frame_H), interpolation=cv2.INTER_LINEAR)
                img_tensor = ds.transform(img_resized)
                frames.append(img_tensor)

            input_tensor = torch.stack(frames, dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_tensor)                
                prob = out.cpu().numpy()[0, :, 0, :, :]

            for i, frame_idx in enumerate(indices):
                prob_map = prob[i]
                img_orig = origs[i]
                Ho, Wo = img_orig.shape[:2]
                prob_up = cv2.resize(prob_map, (Wo, Ho), interpolation=cv2.INTER_LINEAR)

                raw_bin = (prob_up > 0.5).astype(np.uint8) * 255
                raw_fname = f"{frame_idx:05d}_raw.png"
                cv2.imwrite(os.path.join(seq_out, raw_fname), raw_bin)

                mask_prob = np.stack([1.0 - prob_up, prob_up], axis=0).astype(np.float32)
                refined = apply_dense_crf(img_orig, mask_prob)

                final_mask = (refined > 0.5).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(seq_out, f"{frame_idx:05d}.png"), final_mask)

            print(f"Saved Davis masks for '{seq_name}' → {seq_out}")
        print("")

    print("DAVIS inferencecompleted")


def process_yto_videos(cfg, model, device):
    print("Processing YouTube-Objects (YTO) validation frames …")

    yto_root = cfg['yto_root']
    out_dir = cfg['test']['out_dir']

    yto_val_txt = os.path.join(yto_root, "ImageSets", "val.txt")
    if not os.path.isfile(yto_val_txt):
        raise FileNotFoundError(f"Could not find YTO val.txt at {yto_val_txt}")
    with open(yto_val_txt, 'r') as f:
        frame_ids = [line.strip() for line in f if line.strip()]

    print(f"Number of YTO frame IDs in 'val.txt': {len(frame_ids)}")

    yto_pred_root = os.path.join(out_dir, "yto", "pred_masks")
    os.makedirs(yto_pred_root, exist_ok=True)

    # AGNN was trained on 473×473
    in_H, in_W = 473, 473

    for frame_id in frame_ids:
        if "_" not in frame_id:
            print(f"Warning Unexpected frame_id format: '{frame_id}'. Skipping.")
            continue
        cls_prefix = frame_id.split("_", 1)[0]

        jpg_path = os.path.join(yto_root, "JPEGImages", frame_id + ".jpg")
        xml_path = os.path.join(yto_root, "Annotations", frame_id + ".xml")
        if not os.path.isfile(jpg_path):
            print(f"Warning Missing JPEG '{jpg_path}'.  Skipping '{frame_id}'.")
            continue

        bgr = cv2.imread(jpg_path)
        if bgr is None:
            print(f"Error Failed to read '{jpg_path}'. Skipping.")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        Ho, Wo = rgb.shape[:2]

        # Resize to AGNN input resolution & convert to tensor
        resized = cv2.resize(rgb, (in_W, in_H), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        # mean/std standardization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        input_batch = img_tensor.unsqueeze(1)  # (1, 1, 3, 473, 473)
        with torch.no_grad():
            out = model(input_batch)       # (1, 1, 1, hf, wf)
            prob_map = out.squeeze(0).squeeze(0).squeeze(0).cpu().numpy()  # (hf, wf)

        prob_up = cv2.resize(prob_map, (Wo, Ho), interpolation=cv2.INTER_LINEAR)

        raw_bin = (prob_up > 0.5).astype(np.uint8) * 255
        cls_out = os.path.join(yto_pred_root, cls_prefix)
        os.makedirs(cls_out, exist_ok=True)
        cv2.imwrite(os.path.join(cls_out, frame_id + "_raw.png"), raw_bin)

        mask_prob  = np.stack([1.0 - prob_up, prob_up], axis=0).astype(np.float32)
        refined = apply_dense_crf(rgb, mask_prob)
        final_mask = (refined > 0.5).astype(np.uint8) * 255

        save_path = os.path.join(cls_out, f"{frame_id}.png")
        cv2.imwrite(save_path, final_mask)
        print(f"Saved YTO mask: {save_path}")

    print("\nYouTube-Objects inference completed\n")


def test():
    cfg = load_config()
    model, device = load_model(cfg)
    process_davis_videos(cfg, model, device)
    process_yto_videos(cfg, model, device)
