import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from agnn import AGNN
from losses import weighted_bce_loss
from dataloader import get_dataloader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # Load config file
    with open("configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    # Set random seed
    set_seed(cfg.get('seed', 42))

    # Set device
    device = torch.device(cfg["device"] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build dataloaders
    train_loader = get_dataloader(root_dir=cfg['davis_root'],
                                  split='train',
                                  batch_size=cfg['train_batch_size'],
                                  num_frames=cfg['num_frames_train'],
                                  shuffle=True,
                                  num_workers=cfg['num_workers'],
                                  pin_memory=cfg['pin_memory'],
                                  train_mode=True)
    
    val_loader = get_dataloader(root_dir=cfg['davis_root'],
                              split='val',
                              batch_size=cfg['val_batch_size'],
                              num_frames=cfg['num_frames_val'],
                              shuffle=False,
                              num_workers=cfg['num_workers'],
                              pin_memory=cfg['pin_memory'],
                              train_mode=False)
    
    print(f"Train batches : {len(train_loader)}")
    print(f"Valid batches : {len(val_loader)}")

    # Build model
    model = AGNN(hidden_channels=cfg['model']['hidden_channels'], num_iterations=cfg['model']['num_iterations']).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr = cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['scheduler']['milestones'], gamma=cfg['shceduler']['gamma'])

    # Loggin and checkpoints
    writer = SummaryWriter(log_dir=cfg['log_dir'])
    os.makedirs(cfg['save_dir'], exist_ok=True)

    best_val_loss = float('inf')

    # Training helper
    def train_one_epoch(epoch):
        model.train()
        running_loss = 0.0

        for batch_idx, (frames, masks) in enumerate(train_loader):
            frames = frames.to(device)  # (B, N, 3, H, W)
            masks = masks.to(device)  # (B, N, 1, H, W)
            preds = model(frames)  # (B, N, 1, H, W)
            loss = weighted_bce_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                writer.add_scalar('train/batch_loss', loss.item(), epoch * len(train_loader) + batch_idx)

        avg_loss = running_loss / len(train_loader)
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)

        return avg_loss
    
    # Validaiton helper
    @torch.no_grad()
    def validate(epoch):
        model.eval()
        val_loss = 0.0

        for batch_idx, (frames, masks) in enumerate(val_loader):
            frames = frames.to(device)
            masks  = masks.to(device)
            preds = model(frames)
            loss  = weighted_bce_loss(preds, masks)
            val_loss += loss.item()

        if batch_idx == 0:
            writer.add_images('val/frames', frames[:, 0], epoch, dataformats='NCHW')
            writer.add_images('val/get_mask', masks[:, 0], epoch, dataformats='NCHW')
            writer.add_images('val/pred_mask', preds[:, 0], epoch, dataformats='NCHW')

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("val/epoch_loss", avg_val_loss, epoch)
        return avg_val_loss
    
    # Training loop
    for epoch in range(1, cfg['max_epochs'] + 1):
        train_loss = train_one_epoch(epoch)
        val_loss = validate(epoch)
        scheduler.step()

        if epoch == cfg['max_epochs']:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': optimizer.state_dict(),
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(cfg["save_dir"], f"checkpoint_final.pth"))
            print(f"Saved final checkpoint at epoch {epoch}")

    writer.close()
