from pathlib import Path
from typing import Optional, Callable, Any

import torch
import numpy as np
from PIL import Image

from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor, Compose, Normalize

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


class SimpleTEM(VisionDataset):
    def __init__(
            self,
            root: str | Path,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            patch_size: Optional[tuple[int, int]] = None,
            num_patches_per_image: int = 1,
    ) -> None:
        
        super().__init__(
            root,
            transform = transform,
            target_transform = target_transform,
        )
        self.train = train                                  # Train / test mode
        self.patch_size = patch_size
        self.num_patches_per_image = max(1, num_patches_per_image)

        # Load all data into memory
        self.data, self.labels = self._load_data()

    def _load_data(self):
        folder = self.raw_folder

        # Find all raw_XX.tiff files
        raw_files = sorted(folder.glob("raw_*.tiff"))

        data_list = []
        label_list = []

        for raw_path in raw_files:
            # --- load raw image ---
            raw_img = Image.open(raw_path).convert("L")     # Grayscale
            data_list.append(raw_img)

            # --- load label image ---
            if self.train:
                # get the number XX from "raw_XX.tiff"
                stem = raw_path.stem                        # Filename without suffix: "raw_01"
                idx = stem.split("_")[1]                    # Split and get "01"
                label_path = folder / f"label_{idx}.tiff"   # Construct label filename

                label_img = Image.open(label_path)
                label_arr = np.array(label_img, dtype=np.uint8)

                # Reformat labels in way PyTorch understands: 0/50/100/150/200 -> 0:4
                # Enables use of CrossEntropyLoss directly
                mapping = {0: 0, 50: 1, 100: 2, 150: 3, 200: 4}
                for k, v in mapping.items():
                    label_arr[label_arr == k] = v

                label_list.append(label_arr)

        if self.train:
            return data_list, label_list                    # List of PIL Images / numpy arrays
        else:
            return data_list, None
    
    def _get_crop_coords(self, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
        if h <= ph or w <= pw:
            # Image smaller than patch: just start at (0,0)
            return 0, 0

        if self.train:
            # Random crop for training
            y = np.random.randint(0, h - ph)
            x = np.random.randint(0, w - pw)
        else:
            # Center crop for validation / test
            y = (h - ph) // 2
            x = (w - pw) // 2

        return y, x
  
    def __getitem__(self, index: int):
        # Map multi-patch index to image index
        if self.train and self.patch_size is not None:
            img_idx = index // self.num_patches_per_image
        else:
            img_idx = index

        # Get the full image and label
        img_pil = self.data[img_idx]
        label_arr = None if self.labels is None else self.labels[img_idx]

        # Convert image to numpy for cropping
        img_arr = np.array(img_pil)  # (H, W)

        # --------- PATCH / CROP LOGIC ----------
        if self.patch_size is not None:
            ph, pw = self.patch_size                        # Patch height, width
            H, W = img_arr.shape[:2]                        # Image height, width

            if self.train:
                # Grab random crop for training
                top = np.random.randint(0, max(H - ph + 1, 1))
                left = np.random.randint(0, max(W - pw + 1, 1))
            else:
                # Center crop for val / test
                top = max((H - ph) // 2, 0)
                left = max((W - pw) // 2, 0)

            bottom = min(top + ph, H)
            right = min(left + pw, W)
            img_arr = img_arr[top:bottom, left:right]

            if label_arr is not None:
                label_arr = label_arr[top:bottom, left:right]

        # Flip and rotate augmentations (training only)
        if self.train:
            if np.random.rand() < 0.5:                      # Horizontal flip
                img_arr = np.fliplr(img_arr)
                if label_arr is not None:
                    label_arr = np.fliplr(label_arr)

            if np.random.rand() < 0.5:                      # Vertical flip
                img_arr = np.flipud(img_arr)
                if label_arr is not None:
                    label_arr = np.flipud(label_arr)

            k = np.random.randint(0, 4)
            if k > 0:                                       # 0/90/180/270 degree rotation
                img_arr = np.rot90(img_arr, k)
                if label_arr is not None:
                    label_arr = np.rot90(label_arr, k)
                    
        img_arr = np.ascontiguousarray(img_arr)
        if label_arr is not None:
            label_arr = np.ascontiguousarray(label_arr)


        # Convert cropped image back to PIL so transforms work
        img_pil = Image.fromarray(img_arr, mode = "L")

        # Apply image transforms (ToTensor + Normalize)
        if self.transform is not None:
            img = self.transform(img_pil)
        else:
            img = torch.from_numpy(img_arr).unsqueeze(0).float() / 255.0

        # Convert label to tensor (Training set)
        if label_arr is not None:
            label = torch.from_numpy(label_arr).long()
            if self.target_transform is not None:
                label = self.target_transform(label)
            return img, label

        # Fpr test set, no labels
        return img

    def __len__(self) -> int:
        n_images = len(self.data)
        #  During training, each image can provide multiple patches
        if self.train and self.patch_size is not None:
            return n_images * self.num_patches_per_image
        else:
            return n_images

    @property
    def raw_folder(self) -> Path:
        base = Path(self.root)
        if self.train:
            return base / "train_data_tiff"
        else:
            return base / "test_data_tiff"

def iou_score(preds: torch.Tensor,
              targets: torch.Tensor,
              num_classes: int,
              eps: float = 1e-6) -> torch.Tensor:
    ious = []

    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (targets == c)

        union = (pred_c | true_c).sum().float()
        if union == 0:
            continue

        intersection = (pred_c & true_c).sum().float()
        iou_c = (intersection + eps) / (union + eps)
        ious.append(iou_c)

    if len(ious) == 0:
        return torch.tensor(1.0, device=preds.device)

    return torch.stack(ious).mean()

def dice_score(preds: torch.Tensor,
               targets: torch.Tensor,
               num_classes: int,
               eps: float = 1e-6) -> torch.Tensor:
    """
    preds:   (B, H, W) integer predictions
    targets: (B, H, W) integer ground truth
    """
    dices = []

    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (targets == c)

        # Skip this class entirely if it does not appear in GT AND prediction
        if true_c.sum() == 0 and pred_c.sum() == 0:
            continue

        intersection = (pred_c & true_c).sum().float()
        denom = pred_c.sum().float() + true_c.sum().float()

        dice_c = (2.0 * intersection + eps) / (denom + eps)
        dices.append(dice_c)

    if len(dices) == 0:
        # No classes present at all
        return torch.tensor(1.0, device=preds.device)

    return torch.stack(dices).mean()

def soft_dice_loss_from_logits(logits: torch.Tensor,
                               targets: torch.Tensor,
                               num_classes: int = 5,
                               eps: float = 1e-6) -> torch.Tensor:
    """
    Differentiable multi-class Dice loss from logits.

    logits:  (B, C, H, W)  raw scores from the network
    targets: (B, H, W)     integer class labels in [0, C-1]
    """
    # probabilities over classes
    probs = torch.softmax(logits, dim=1)          # (B, C, H, W)

    # one-hot encode targets to match probs
    one_hot = F.one_hot(targets, num_classes)     # (B, H, W, C)
    one_hot = one_hot.permute(0, 3, 1, 2).float() # (B, C, H, W)

    dims = (0, 2, 3)  # sum over batch + spatial dims
    intersection = (probs * one_hot).sum(dims)
    cardinality  = probs.sum(dims) + one_hot.sum(dims)

    dice = (2.0 * intersection + eps) / (cardinality + eps)  # per-class Dice
    return 1.0 - dice.mean()  # scalar loss

# ----------------------------- Implementing the UNet From Lec 23 ----------------------------- 
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding = 1)

        # 1×1 conv for channel mismatch
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)
    
class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads = 1):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)                                   # (B, 3C, H, W)
        q, k, v = qkv.reshape(B, 3, C, H * W).unbind(dim = 1)

        # Compute attention
        attn = (q.transpose(-1, -2) @ k) / (C ** 0.5)
        attn = torch.softmax(attn, dim = -1)

        h = (attn @ v.transpose(-1, -2)).transpose(-1, -2)
        h = h.reshape(B, C, H, W)

        return x + self.proj(h)
    
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attn):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch)
        self.res2 = ResBlock(out_ch, out_ch)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride = 2, padding = 1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        skip = x
        x = self.pool(x)
        return x, skip
    
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attn):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch)
        self.res2 = ResBlock(out_ch, out_ch)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, use_skip = True):
        x = F.interpolate(x, scale_factor = 2, mode = "nearest")
        x = torch.cat([x, skip], dim = 1)
        x = self.res1(x)
        x = self.res2(x)
        return self.attn(x)
    
class UNet32(nn.Module):
    def __init__(self, in_ch = 1, out_classes = 5, logit = True):
        super().__init__()
        self.logit = logit

        # Initial conv to go from 1 -> 32 channels
        self.inc = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)

        # Encoder
        self.down1 = DownBlock(32, 32, use_attn = False)
        self.down2 = DownBlock(32, 64, use_attn = False)
        self.down3 = DownBlock(64, 128, use_attn = False)

        # Bottleneck
        self.bot1 = ResBlock(128, 256)
        self.bot2 = ResBlock(256, 256)
        self.bot3 = SelfAttention(256)

        # Decoder
        self.up3 = UpBlock(256 + 128, 128, use_attn = False)
        self.up2 = UpBlock(128 + 64, 64, use_attn = False)
        self.up1 = UpBlock(64 + 32, 32, use_attn = False)

        # Final classifier: 5 classes
        self.out = nn.Conv2d(32, out_classes, 1)

    def forward(self, x):
        # Initial conv
        x = self.inc(x)

        # Encoder
        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        x, s3 = self.down3(x)

        # Bottleneck
        x = self.bot1(x)
        x = self.bot2(x)
        x = self.bot3(x)

        # Decoder
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        # Output Logits
        x = self.out(x)
        return x

# -----------------------------
# Testing the dataset
# -----------------------------
if __name__ == "__main__":
    # -----------------------------
    # Setup
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Image transform
    image_transform = Compose([
        ToTensor(),                                     # (H, W) -> (1, H, W), float in [0,1]
        Normalize(mean = [0.5], std = [0.5])            # Roughly center around 0
    ])

    # -----------------------------
    # Building Dataset
    # -----------------------------
    full_train_ds = SimpleTEM(
        root = "ProjectDatasets",
        train = True,
        transform = image_transform,
        patch_size = (256, 256),
        num_patches_per_image = 80,                     # More patches per base image
    )
    print("Total (patch) samples before slicing:", len(full_train_ds))
    ONE_IMAGE_DEBUG = False                             # If True, only use one base image

    if ONE_IMAGE_DEBUG:
        # Keep only raw_01 / label_01 (first base image)
        full_train_ds.data = full_train_ds.data[:1]
        full_train_ds.labels = full_train_ds.labels[:1]
        print("ONE_IMAGE_DEBUG: using only base image #0")
        print("Total (patch) samples after slicing:", len(full_train_ds))

    num_classes = 5
    batch_size = 8

    # Train/val split over patches of this dataset
    train_size = int(0.8 * len(full_train_ds))
    val_size = len(full_train_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_train_ds, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    print("Train patches:", len(train_ds))
    print("Val patches:", len(val_ds))

    # -----------------------------
    # Model, loss, optimizer
    # -----------------------------
    model = UNet32(in_ch = 1, out_classes = 5, logit = True).to(device)

    ce_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = MultiStepLR(
        optimizer,
        milestones = [20, 40],
        gamma = 0.3
    )

    dice_weight = 0.5
    num_epochs = 100

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(num_epochs):                     # Will run for 65 epochs
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)                      # (B, 1, H, W)
            labels = labels.to(device)                  # (B, H, W)

            optimizer.zero_grad()
            logits = model(imgs)                        # (B, C, H, W)

            ce_loss = ce_criterion(logits, labels)
            dice_loss = soft_dice_loss_from_logits(
                logits, labels, num_classes = num_classes
            )
            loss = ce_loss + dice_weight * dice_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ---- VALIDATION ----
        model.eval()
        val_running_loss = 0.0
        dice_sum = 0.0
        iou_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                logits = model(imgs)
                ce_loss = ce_criterion(logits, labels)
                dice_loss = soft_dice_loss_from_logits(
                    logits, labels, num_classes=num_classes
                )
                loss = ce_loss + dice_weight * dice_loss

                val_running_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(logits, dim=1)

                dice_batch = dice_score(
                    preds, labels, num_classes=num_classes
                ).item()
                iou_batch = iou_score(
                    preds, labels, num_classes=num_classes
                ).item()

                dice_sum += dice_batch
                iou_sum += iou_batch
                n_val_batches += 1

        val_loss = val_running_loss / len(val_loader.dataset)
        mean_dice = dice_sum / max(n_val_batches, 1)
        mean_iou = iou_sum / max(n_val_batches, 1)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- train loss: {train_loss:.4f} "
            f"- val loss: {val_loss:.4f} "
            f"- val Dice: {mean_dice:.3f} "
            f"- val IoU: {mean_iou:.3f}"
        )
        # End of epoch loop
        scheduler.step()

    print("Training complete.")

    # -----------------------------
    # Final visualization on one val patch
    # -----------------------------
    model.eval()
    with torch.no_grad():
        chosen_idx = None
        for i in range(len(val_ds)):
            full_idx = val_ds.indices[i]
            _, lbl = full_train_ds[full_idx]
            if torch.unique(lbl).numel() >= 2:
                chosen_idx = i
                break

        if chosen_idx is None:
            chosen_idx = 0                              # Fallback: just use first index

        full_idx = val_ds.indices[chosen_idx]
        img, label = full_train_ds[full_idx]

        imgs_vis = img.unsqueeze(0).to(device)
        labels_vis = label.unsqueeze(0).to(device)

        logits_vis = model(imgs_vis)
        preds_vis = torch.argmax(logits_vis, dim=1)

    img0 = imgs_vis[0]
    label0 = labels_vis[0]
    pred0 = preds_vis[0]

    img_vis = (img0.cpu() * 0.5 + 0.5).clamp(0, 1).squeeze(0).numpy()
    gt_mask = label0.cpu().numpy()
    pred_mask = pred0.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(img_vis, cmap = "gray")
    axs[0].set_title("Input image")
    axs[0].axis("off")

    axs[1].imshow(gt_mask, cmap = "tab10", vmin = 0, vmax = num_classes-1)
    axs[1].set_title("Ground truth mask")
    axs[1].axis("off")

    axs[2].imshow(pred_mask, cmap = "tab10", vmin = 0, vmax = num_classes-1)
    axs[2].set_title("Predicted mask")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


    # ----------------------------------
    #  Test-time inference on test set
    # ----------------------------------
    RUN_TEST_VIS = True                                 # True to visualize test predictions

    if RUN_TEST_VIS:
        test_ds = SimpleTEM(
            root = "ProjectDatasets",
            train = False,                              # Test set
            transform = image_transform,
            patch_size = (256, 256),
            num_patches_per_image = 1
        )

        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        # Load best model weights
        model.to(device)
        model.eval()

        num_test_examples = 4
        import itertools

        for i, imgs in enumerate(itertools.islice(test_loader, num_test_examples)):
            imgs = imgs.to(device)
            with torch.no_grad():
                logits = model(imgs)
                preds = torch.argmax(logits, dim = 1)

            img_np = (imgs[0].cpu() * 0.5 + 0.5).clamp(0, 1).squeeze(0).numpy()
            pred_np = preds[0].cpu().numpy()

            plt.figure(figsize = (6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(img_np, cmap = "gray")
            plt.title(f"Test image #{i}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(pred_np, cmap = "tab10", vmin = 0, vmax = num_classes - 1)
            plt.title("Predicted mask")
            plt.axis("off")

            plt.tight_layout()
            plt.show()
