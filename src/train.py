import os
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from agnn import AGNN
from losses import weighted_bce_loss
from dataloader import get_dataloader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, optimizer, train_loader, writer, epoch, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (frames, masks) in enumerate(train_loader):
        frames = frames.to(device)
        masks  = masks.to(device)

        preds = model(frames)  # (B, N, 1, H, W)
        B, N, C, H, W = preds.shape
        preds_flat = preds.view(B*N, C, H, W)
        masks_flat = masks.view(B*N, C, masks.shape[-2], masks.shape[-1])
        masks_flat = F.interpolate(masks_flat, size=(H,W), mode='nearest')

        loss = weighted_bce_loss(preds_flat, masks_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            writer.add_scalar(
                'train/batch_loss',
                loss.item(),
                epoch * len(train_loader) + batch_idx
            )

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    return avg_loss

@torch.no_grad()
def validate(model, val_loader, writer, epoch, device):
    model.eval()
    val_loss = 0.0

    for batch_idx, (frames, masks) in enumerate(val_loader):
        frames = frames.to(device)
        masks  = masks.to(device)

        preds = model(frames)
        if batch_idx == 0:
            # log the first batch of images & masks
            writer.add_images('val/frames',   frames[:,0], epoch, dataformats='NCHW')
            writer.add_images('val/gt_mask',  masks[:,0],  epoch, dataformats='NCHW')
            writer.add_images('val/pred_mask',preds[:,0],  epoch, dataformats='NCHW')
            writer.flush()

        B, N, C, Hf, Wf = preds.shape
        preds_flat = preds.view(B*N, C, Hf, Wf)
        _, _, _, Hm, Wm = masks.shape
        masks_flat = masks.view(B*N, C, Hm, Wm)
        masks_flat = F.interpolate(masks_flat, size=(Hf,Wf), mode='nearest')

        loss = weighted_bce_loss(preds_flat, masks_flat)
        val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar("val/epoch_loss", avg_val_loss, epoch)
    return avg_val_loss

def train():
    # --- load config ---
    with open("configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- dataloaders ---
    train_loader = get_dataloader(
        root_dir=cfg["davis_root"], split="train",
        batch_size=cfg["train_batch_size"], num_frames=cfg["num_frames_train"],
        shuffle=True, num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"], train_mode=True
    )
    val_loader = get_dataloader(
        root_dir=cfg["davis_root"], split="val",
        batch_size=cfg["val_batch_size"], num_frames=cfg["num_frames_test"],
        shuffle=False, num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"], train_mode=False
    )
    print(f"Train batches: {len(train_loader)}   Val batches: {len(val_loader)}")

    # --- model & optimizer ---
    model = AGNN(
        hidden_channels=cfg["model"]["hidden_channels"],
        num_iterations=cfg["model"]["num_iterations"]
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"])
    )

    # --- scheduler: linear warmup → MultiStep decay ---
    warmup_epochs = int(cfg["scheduler"]["warmup_epochs"])
    milestones = [int(x) for x in cfg["scheduler"]["milestones"]]
    gamma = float(cfg["scheduler"]["gamma"])

    warmup_sched = LinearLR(
        optimizer,
        start_factor=0.1, end_factor=1.0,
        total_iters=warmup_epochs
    )
    decay_sched = MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, decay_sched],
        milestones=[warmup_epochs]
    )

    # --- logging & checkpoints setup ---
    writer = SummaryWriter(log_dir=cfg["log_dir"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    best_val = float("inf")
    patience = cfg["early_stopping"]["patience"]
    no_improve = 0

    # --- main training loop ---
    for epoch in range(1, cfg["max_epochs"] + 1):
        tl = train_one_epoch(model, optimizer, train_loader, writer, epoch, device)
        vl = validate(model, val_loader, writer, epoch, device)

        scheduler.step()

        # early stopping logic
        if vl < best_val:
            best_val = vl
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"→ Early stopping at epoch {epoch} (patience={patience})")
                break

        # periodic checkpointing
        if epoch % cfg["save_every"] == 0:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "val_loss": vl
            }
            fname = f"checkpoint_epoch{epoch}.pth"
            torch.save(ckpt, os.path.join(cfg["save_dir"], fname))
            print(f"→ Saved checkpoint: {fname}")

    writer.close()

if __name__ == "__main__":
    train()
