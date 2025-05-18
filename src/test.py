import os
import yaml
import torch
import cv2
import numpy as np
from agnn import AGNN
from dataloader import VideoSegDataset
from utils import apply_dense_crf


def load_config(config_path="configs/default.yaml"):
    """
    Load the YAML configuration file.
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


def process_videos(cfg, model, device):
    """
    Perform inference on each DAVIS 'val' video in subsets, apply CRF, and save masks
    """
    N0 = cfg['num_frames_test']

    ds = VideoSegDataset(
        root_dir=cfg['davis_root'],
        split='val',
        num_frames=N0,
        train_mode=False
    )

    frame_H, frame_W = ds.frame_size

    out_dir = cfg['test']['out_dir']
    pred_dir = os.path.join(out_dir, 'pred_masks')
    os.makedirs(pred_dir, exist_ok=True)

    for img_paths, _ in ds.samples:
        seq_name = os.path.basename(os.path.dirname(img_paths[0]))
        seq_out = os.path.join(pred_dir, seq_name)
        os.makedirs(seq_out, exist_ok=True)

        total = len(img_paths)
        T = total // N0
        if T < 1:
            print(f"Skipping sequence '{seq_name}' with only {total} frames.")
            continue
        print(f"Processing sequence '{seq_name}': {T} subsets of {N0} frames")

        for t in range(T):
            indices = [t + k * T for k in range(N0)]

            frames = []
            orig_imgs = []
            for idx in indices:
                path = img_paths[idx]
                # Read BGR image and convert to RGB as a contiguous array
                bgr = cv2.imread(path)
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                orig_imgs.append(img)
                img_resized = cv2.resize(img, (frame_W, frame_H))
                img_tensor = ds.transform(img_resized)
                frames.append(img_tensor)

            input_tensor = torch.stack(frames, dim=0).unsqueeze(0).to(device)  # (1, N0, 3, H, W)
            with torch.no_grad():
                out = model(input_tensor)  # (1, N0, 1, Hf, Wf)
                prob = torch.sigmoid(out).cpu().numpy()[0, :, 0, :, :]  # (N0, Hf, Wf)

            for i, frame_idx in enumerate(indices):
                prob_map = prob[i]
                # Upsample to original image size for CRF
                img_orig = orig_imgs[i]
                H_orig, W_orig = img_orig.shape[:2]
                prob_up = cv2.resize(prob_map, (W_orig, H_orig))

                # Build probability array for CRF [bg_prob, fg_prob]
                mask_prob = np.stack([1 - prob_up, prob_up], axis=0).astype(np.float32)
                # Ensure contiguous memory layout
                mask_prob = np.ascontiguousarray(mask_prob)

                # Apply dense CRF at original resolution
                refined = apply_dense_crf(img_orig, mask_prob)

                # Threshold and save
                bin_mask = (refined > 0.5).astype(np.uint8) * 255
                fname = f"{frame_idx:05d}.png"
                out_path = os.path.join(seq_out, fname)
                cv2.imwrite(out_path, bin_mask)

        print(f"Saved masks for sequence '{seq_name}' to {seq_out}\n")

    print("Inference and CRF processing completed.")


if __name__ == '__main__':
    cfg = load_config()
    model, device = load_model(cfg)
    process_videos(cfg, model, device)
