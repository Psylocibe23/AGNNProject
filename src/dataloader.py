import os
from glob import glob
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms



class VideoSegDataset(Dataset):
    """
    Video object segmentation dataset loader for AGNN.
    Assumes folder-structure:

    root_dir/
      JPEGImages/480p/{video}/frame_xxxxx.jpg
      Annotations_unsupervised/480p/{video}/frame_xxxxx.png
      ImageSets/{split}.txt 
    """
    def __init__(self, root_dir, split='train', num_frames=3, frame_size=(473, 473), transform=None, train_mode=True):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
        self.train_mode = train_mode

        # read the list of video names for this split
        list_file = os.path.join(self.root_dir, 'ImageSets', f"{split}.txt") 
        if not os.path.isfile(list_file):
            raise FileNotFoundError(f"Could not find split file {list_file}")
        # Read all image–mask pairs
        video_dict = {}
        with open(list_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        # Inspect first line to choose parsing mode
        first_parts = lines[0].split()
        if len(first_parts) == 2:
            # ---- frame‐level DAVIS style ----
            video_dict = {}
            for ln in lines:
                img_rel, msk_rel = ln.split()
                img_rel = img_rel.lstrip('/')
                msk_rel = msk_rel.lstrip('/')
                img_path = os.path.join(root_dir, img_rel)
                msk_path = os.path.join(root_dir, msk_rel)
                vid = img_rel.split('/')[2]         
                video_dict.setdefault(vid, []).append((img_path, msk_path))
            self.samples = [
                ([p for p,_ in lst], [q for _,q in lst])
                for lst in video_dict.values()
            ]

        elif len(first_parts) == 1:
            # DAVIS‐unsupervised style: one sequence name per line, masks as PNGs
            from glob import glob
            self.samples = []
            for vid in lines:
                img_dir = os.path.join(root_dir, 'JPEGImages', '480p', vid)
                msk_dir = os.path.join(root_dir, 'Annotations_unsupervised', '480p', vid)
        
                imgs = sorted(glob(os.path.join(img_dir, '*.jpg')))
                msks = sorted(glob(os.path.join(msk_dir, '*.png')))
                if not imgs:
                    raise RuntimeError(f"No images for video '{vid}' in '{img_dir}'")
                if len(imgs) != len(msks):
                    raise RuntimeError(f"Image/mask count mismatch for '{vid}': {len(imgs)} vs {len(msks)}")
        
                # each entry is a (list_of_image_paths, list_of_mask_paths)
                self.samples.append((imgs, msks))


        else:
            raise RuntimeError("Unrecognized split format")

    def __len__(self):
         return len(self.samples)
    
    def __getitem__(self, idx):
        imgs, masks = self.samples[idx]
        N, total = self.num_frames, len(imgs)

        # For training pick N frames randomly
        if self.train_mode:
            segment_size = total // N
            indices = []
            for k in range(N):
                start = k * segment_size
                end = (k+1) * segment_size if k < (N - 1) else total
                index = np.random.randint(start, end)
                indices.append(index)
        else:
            # pick N frames uniformly
            if total >= N:
                step = total / N
                indices = [int(i*step) for i in range(N)]
            else:
                indices = [i % total for i in range(N)]

        frames, segs = [], []
        for i in indices:
            # Load and process the image
            img = cv2.imread(imgs[i])[:, :, ::-1]
            img = cv2.resize(img, self.frame_size)
            img = self.transform(img)
            frames.append(img)

            # Load and process the mask
            mask_path = masks[i]
            if mask_path.lower().endswith('.xml'):
                # parse the XML bounding box (bbox)
                tree = ET.parse(mask_path)
                root = tree.getroot()
                size = root.find('size')
                orig_w = int(size.find('width').text)
                orig_h = int(size.find('height').text)
                # find object→bbox
                bb = root.find('object').find('bndbox')
                xmin = int(bb.find('xmin').text)
                ymin = int(bb.find('ymin').text)
                xmax = int(bb.find('xmax').text)
                ymax = int(bb.find('ymax').text)
            
                # resize coords from original resolution to frame_size
                new_w, new_h = self.frame_size[::-1]  # (W, H)
                x_scale = new_w / orig_w
                y_scale = new_h / orig_h
                xmin = int(xmin * x_scale)
                xmax = int(xmax * x_scale)
                ymin = int(ymin * y_scale)
                ymax = int(ymax * y_scale)
            
                # build binary mask
                m = np.zeros((new_h, new_w), dtype=np.float32)
                m[ymin:ymax, xmin:xmax] = 1.0
                mask_tensor = torch.from_numpy(m).unsqueeze(0)  # (1, H, W)
            
            else:
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, self.frame_size[::-1],
                               interpolation=cv2.INTER_NEAREST)
                m = (m > 0).astype('float32')
                mask_tensor = torch.from_numpy(m).unsqueeze(0)
            
            segs.append(mask_tensor)

        # stack along the “time/node” dim
        frames = torch.stack(frames, dim=0)
        segs = torch.stack(segs, dim=0)
        
        return frames, segs
    
    
def get_dataloader(root_dir, split, batch_size, num_frames, shuffle, num_workers, pin_memory, train_mode=False):
    ds = VideoSegDataset(root_dir=root_dir, split=split, num_frames=num_frames, train_mode=train_mode)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    