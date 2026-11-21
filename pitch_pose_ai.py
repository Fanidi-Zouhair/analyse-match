import os
import glob
import random
import math
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34, ResNet34_Weights

# ==============================================================================
# 1. CONFIGURATION PRO (Dataset3 4.7k -> ~9.4k b H-Flip | ResNet34 @ 320x320)
# ==============================================================================
NUM_KEYPOINTS = 39
IMG_SIZE = 320                  # Heatmap 80x80 (diqqa a3la 3la l-khotout r-raqiqa)
DATASET_DIRS = ["/content/dataset3"]
BATCH_SIZE = 32
MAX_EPOCHS = 100
WARMUP_EPOCHS = 5
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 22
SAVE_PATH = "/content/drive/MyDrive/best_pose_pro_320.pth"

# Auto-Remap exact mn Roboflow dataset3 (33 pts, 1-indexed) -> Unified 39 pts (1-indexed)
MAP_33_TO_39 = {
    # Left side
    1: 17, 2: 16, 3: 31, 4: 35, 5: 11, 6: 12, 7: 37, 8: 36, 9: 15, 10: 10,
    27: 32, 28: 34, 29: 33,
    # Center
    21: 18, 22: 19, 23: 21, 24: 22, 25: 20, 26: 9, 33: 39,
    # Right side
    11: 1, 12: 2, 13: 23, 14: 27, 15: 7, 16: 6, 17: 29, 18: 28, 19: 3, 20: 8,
    30: 24, 31: 26, 32: 25,
}

# Symmetric Left <-> Right Swap table (1-indexed f unified 39 pts) l Horizontal Flip
HFLIP_SWAP_PAIRS_1IDX = [
    (17, 1), (16, 2), (15, 3), (12, 6), (11, 7), (10, 8),
    (36, 28), (37, 29), (31, 23), (32, 24), (34, 26), (35, 27), (33, 25),
    (21, 20),
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 2. BROADCAST-GRADE AUGMENTATION + AUTO 33->39 DATASET
# ==============================================================================
class RoboflowPitchDataset(Dataset):
    def __init__(self, dataset_dirs, split_name="train", img_size=IMG_SIZE, num_keypoints=NUM_KEYPOINTS, is_train=True):
        self.img_size = img_size
        self.num_keypoints = num_keypoints
        self.is_train = is_train
        self.samples = []

        # Precompute 0-indexed flip permutation array
        self.flip_perm = np.arange(num_keypoints, dtype=np.int64)
        for a, b in HFLIP_SWAP_PAIRS_1IDX:
            self.flip_perm[a - 1] = b - 1
            self.flip_perm[b - 1] = a - 1

        exts = ("*.jpg", "*.jpeg", "*.png")
        for d_dir in dataset_dirs:
            img_dir = os.path.join(d_dir, split_name, "images")
            lbl_dir = os.path.join(d_dir, split_name, "labels")
            if not os.path.exists(img_dir):
                print(f"[INFO] Dossier '{img_dir}' ma-kaynch (t-jahalnah).")
                continue

            found = []
            for ext in exts:
                found.extend(glob.glob(os.path.join(img_dir, ext)))
            found.sort()

            for img_path in found:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(lbl_dir, base_name + ".txt")
                if os.path.exists(label_path):
                    self.samples.append((img_path, label_path))

            print(f"[DATA] Charga {len(found):4d} tsawar mn '{img_dir}'")

        if len(self.samples) == 0:
            raise FileNotFoundError(f"Ma-lqitch 7ta tswira f '{split_name}' dakhal {dataset_dirs}!")

    def __len__(self):
        return len(self.samples)

    def _parse_yolo_pose_label(self, label_path):
        """
        Auto-detect:
        - Ila label fih 33 keypoints (Roboflow dataset3 direct) -> kay-remap automatiquement l 39 pts!
        - Ila label deja fih 39 keypoints -> kay-qrah direct!
        """
        kpts = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        vis = np.zeros((self.num_keypoints,), dtype=np.float32)

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if not lines:
            return kpts, vis

        parts = list(map(float, lines[0].split()))
        raw_kpts = parts[5:]

        step = 3 if (len(raw_kpts) % 3 == 0 and len(raw_kpts) >= 60) else 2
        n_pts = len(raw_kpts) // step

        if n_pts == 33:
            # Direct Roboflow dataset3 (33 pts) -> Auto-remap exact l unified 39 pts
            for src_1idx, dst_1idx in MAP_33_TO_39.items():
                i = src_1idx - 1
                x = raw_kpts[i * step]
                y = raw_kpts[i * step + 1]
                v = raw_kpts[i * step + 2] if step == 3 else (1.0 if (x > 0 or y > 0) else 0.0)
                if v > 0 and (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0):
                    kpts[dst_1idx - 1, 0] = x
                    kpts[dst_1idx - 1, 1] = y
                    vis[dst_1idx - 1] = 1.0
        else:
            max_pts = min(self.num_keypoints, n_pts)
            for i in range(max_pts):
                x = raw_kpts[i * step]
                y = raw_kpts[i * step + 1]
                v = raw_kpts[i * step + 2] if step == 3 else (1.0 if (x > 0 or y > 0) else 0.0)
                if v > 0 and (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0):
                    kpts[i, 0] = x
                    kpts[i, 1] = y
                    vis[i] = 1.0

        return kpts, vis

    @staticmethod
    def _apply_motion_blur(img_u8):
        ksize = random.choice([5, 7, 9])
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        angle = random.uniform(-25.0, 25.0)
        kernel[ksize // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((ksize / 2.0 - 0.5, ksize / 2.0 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
        s = kernel.sum()
        if s > 1e-6:
            kernel /= s
        return cv2.filter2D(img_u8, -1, kernel)

    @staticmethod
    def _apply_random_shadow(img_f32):
        h, w = img_f32.shape[:2]
        x1, x2 = random.uniform(0, w), random.uniform(0, w)
        yy, xx = np.mgrid[0:h, 0:w]
        mask = ((xx - x1) * h - (x2 - x1) * yy) > 0
        if random.random() < 0.5:
            mask = ~mask
        factor = random.uniform(0.55, 0.85)
        out = img_f32.copy()
        out[mask] *= factor
        return out

    def _apply_augmentation(self, img_u8, kpts, vis):
        h, w = self.img_size, self.img_size

        # 1. Horizontal Flip (50%) + Exact Left<->Right Keypoint Swap (kay-doubel data!)
        if random.random() < 0.50:
            img_u8 = cv2.flip(img_u8, 1)
            kpts = kpts[self.flip_perm].copy()
            vis = vis[self.flip_perm].copy()
            vis_mask = vis > 0.5
            kpts[vis_mask, 0] = 1.0 - kpts[vis_mask, 0]

        # 2. Geometric: Rotation + Scale + Translation (80%)
        if random.random() < 0.80:
            angle = random.uniform(-8.0, 8.0)
            scale = random.uniform(0.82, 1.18)
            tx = random.uniform(-0.09, 0.09) * w
            ty = random.uniform(-0.09, 0.09) * h

            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty

            img_u8 = cv2.warpAffine(
                img_u8, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            vis_idx = np.where(vis > 0.5)[0]
            if len(vis_idx) > 0:
                pts_px = np.ones((len(vis_idx), 3), dtype=np.float32)
                pts_px[:, 0] = kpts[vis_idx, 0] * w
                pts_px[:, 1] = kpts[vis_idx, 1] * h
                new_pts = pts_px @ M.T
                new_x = new_pts[:, 0] / w
                new_y = new_pts[:, 1] / h
                for k_i, idx_pt in enumerate(vis_idx):
                    if 0.0 <= new_x[k_i] <= 1.0 and 0.0 <= new_y[k_i] <= 1.0:
                        kpts[idx_pt, 0] = new_x[k_i]
                        kpts[idx_pt, 1] = new_y[k_i]
                    else:
                        kpts[idx_pt] = 0.0
                        vis[idx_pt] = 0.0

        # 3. Motion Blur (25%) — fast camera pan
        if random.random() < 0.25:
            img_u8 = self._apply_motion_blur(img_u8)

        # 4. Gaussian Blur / Stream Compression (25%)
        if random.random() < 0.25:
            if random.random() < 0.5:
                k = random.choice([3, 5])
                img_u8 = cv2.GaussianBlur(img_u8, (k, k), 0)
            else:
                down_s = random.uniform(0.45, 0.75)
                small = cv2.resize(img_u8, (max(32, int(w * down_s)), max(32, int(h * down_s))), interpolation=cv2.INTER_LINEAR)
                img_u8 = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        # 5. HSV Grass / White-Balance Jitter (55%)
        if random.random() < 0.55:
            hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10.0, 10.0)) % 180.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.75, 1.30), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.75, 1.25), 0, 255)
            img_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        img = img_u8.astype(np.float32) / 255.0

        # 6. Brightness / Contrast / Gamma (60%)
        if random.random() < 0.60:
            gamma = random.uniform(0.75, 1.35)
            alpha = random.uniform(0.80, 1.25)
            beta = random.uniform(-0.10, 0.10)
            img = np.clip((np.power(np.clip(img, 1e-4, 1.0), gamma)) * alpha + beta, 0.0, 1.0)

        # 7. Stadium Shadow (30%)
        if random.random() < 0.30:
            img = self._apply_random_shadow(img)

        # 8. Sensor Noise (20%)
        if random.random() < 0.20:
            noise = np.random.normal(0.0, random.uniform(0.01, 0.03), img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        # 9. Player / Scoreboard Occlusion Cutout (25%)
        if random.random() < 0.25:
            n_boxes = random.randint(1, 3)
            for _ in range(n_boxes):
                bw = random.randint(12, 34)
                bh = random.randint(18, 48)
                x1 = random.randint(0, w - bw)
                y1 = random.randint(0, h - bh)
                col = np.random.uniform(0.0, 0.9, size=(1, 1, 3)).astype(np.float32)
                img[y1:y1 + bh, x1:x1 + bw, :] = col

        return img, kpts, vis

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Ma-qdartch n-qra t-tswira: {img_path}")
        img_u8 = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        kpts, vis = self._parse_yolo_pose_label(label_path)

        if self.is_train:
            img, kpts, vis = self._apply_augmentation(img_u8, kpts, vis)
        else:
            img = img_u8.astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).contiguous()
        kpts_tensor = torch.from_numpy(kpts)
        vis_tensor = torch.from_numpy(vis)
        return img_tensor, kpts_tensor, vis_tensor


# ==============================================================================
# 3. ARCHITECTURE PRO: RESNET34 U-NET + 80x80 SPATIAL SOFTMAX + HEATMAP-FUSED VIS
# ==============================================================================
class PitchPoseNet(nn.Module):
    def __init__(self, num_keypoints=NUM_KEYPOINTS, pretrained=True, img_size=IMG_SIZE):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.img_size = img_size
        hm_size = img_size // 4  # 80x80 for 320x320, 64x64 for 256x256

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        try:
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = resnet34(weights=weights)
        except Exception:
            backbone = resnet34(weights=None)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.enc1 = backbone.layer1  # 64,  H/4
        self.enc2 = backbone.layer2  # 128, H/8
        self.enc3 = backbone.layer3  # 256, H/16
        self.enc4 = backbone.layer4  # 512, H/32

        self.dec3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.10),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.08),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.heatmap_head = nn.Conv2d(64, num_keypoints, kernel_size=1)

        # Global context + per-keypoint heatmap peak fusion for accurate Visibility
        self.vis_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_keypoints),
        )
        self.vis_hm_scale = nn.Parameter(torch.ones(1, num_keypoints) * 0.5)
        self.vis_hm_bias = nn.Parameter(torch.zeros(1, num_keypoints))

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, hm_size),
            torch.linspace(0.0, 1.0, hm_size),
            indexing="ij",
        )
        self.register_buffer("grid_x", grid_x.reshape(1, 1, -1))
        self.register_buffer("grid_y", grid_y.reshape(1, 1, -1))

    def forward(self, x):
        x = (x - self.img_mean) / self.img_std

        s = self.stem(x)
        e1 = self.enc1(s)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        heatmaps = self.heatmap_head(d1)  # (B, K, H/4, W/4)
        B, K, H, W = heatmaps.shape
        hm_flat = heatmaps.view(B, K, -1)
        probs = F.softmax(hm_flat, dim=-1)

        pred_x = (probs * self.grid_x).sum(dim=-1)
        pred_y = (probs * self.grid_y).sum(dim=-1)
        coords = torch.stack([pred_x, pred_y], dim=-1)

        # Fuse global visibility + heatmap peak contrast (max - mean)
        hm_peak = hm_flat.max(dim=-1).values - hm_flat.mean(dim=-1)
        vis_logits = self.vis_global(e4) + self.vis_hm_scale * hm_peak + self.vis_hm_bias

        return coords, vis_logits


# ==============================================================================
# 4. LOSS & METRICS (Sub-pixel SmoothL1 + Visibility BCE + Light Geometry)
# ==============================================================================
class PitchPoseProLoss(nn.Module):
    def __init__(self, img_size=IMG_SIZE, geom_weight=0.05, device="cpu"):
        super().__init__()
        self.img_size = img_size
        self.geom_weight = geom_weight
        self.off_diag = (1.0 - torch.eye(NUM_KEYPOINTS, device=device)).unsqueeze(0)

    def forward(self, pred, vis_logits, target, vis):
        vis_mask = vis.unsqueeze(-1)
        valid_pts = vis_mask.sum().clamp(min=1.0) * 2.0
        pose_loss = (F.smooth_l1_loss(pred, target, beta=0.02, reduction="none") * vis_mask).sum() / valid_pts

        vis_loss = F.binary_cross_entropy_with_logits(vis_logits, vis)
        vis_acc = ((torch.sigmoid(vis_logits) > 0.5).float() == (vis > 0.5).float()).float().mean() * 100.0

        # Pairwise distance + direction consistency on visible keypoints
        diff_p = pred.unsqueeze(1) - pred.unsqueeze(2)
        diff_t = target.unsqueeze(1) - target.unsqueeze(2)
        pair_vis = (vis.unsqueeze(2) * vis.unsqueeze(1)) * self.off_diag
        valid_pairs = pair_vis.sum().clamp(min=1.0)

        dist_p = torch.sqrt((diff_p ** 2).sum(dim=-1) + 1e-6)
        dist_t = torch.sqrt((diff_t ** 2).sum(dim=-1) + 1e-6)
        dist_loss = (((dist_p - dist_t) ** 2) * pair_vis).sum() / valid_pairs

        dir_p = diff_p / dist_p.unsqueeze(-1).clamp(min=1e-4)
        dir_t = diff_t / dist_t.unsqueeze(-1).clamp(min=1e-4)
        cos_sim = (dir_p * dir_t).sum(dim=-1).clamp(-0.9999, 0.9999)
        dir_loss = ((1.0 - cos_sim) * pair_vis).sum() / valid_pairs

        total_loss = pose_loss + 0.04 * vis_loss + self.geom_weight * (dist_loss + 0.25 * dir_loss)

        # Metrics in pixels (at IMG_SIZE)
        px_diff = (pred - target) * self.img_size
        px_dist = torch.sqrt((px_diff ** 2).sum(dim=-1) + 1e-8)
        vis_sum = vis.sum().clamp(min=1.0)
        mean_px = (px_dist * vis).sum() / vis_sum
        pck_5 = (((px_dist <= 5.0).float()) * vis).sum() / vis_sum * 100.0
        pck_10 = (((px_dist <= 10.0).float()) * vis).sum() / vis_sum * 100.0

        return total_loss, {
            "tot": total_loss.item(),
            "pose": pose_loss.item(),
            "px": mean_px.item(),
            "pck5": pck_5.item(),
            "pck10": pck_10.item(),
            "vis": vis_acc.item(),
        }


# ==============================================================================
# 5. TRAINING LOOP (AMP FP16 + AdamW + Warmup CosineAnnealing + Early Stop)
# ==============================================================================
def train_model(dataset_dirs, save_path, img_size=IMG_SIZE, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[INFO] Device: {device} | AMP FP16: {use_amp} | ResNet34 @ {img_size}x{img_size}")

    train_ds = RoboflowPitchDataset(dataset_dirs, split_name="train", img_size=img_size, is_train=True)
    val_ds = RoboflowPitchDataset(dataset_dirs, split_name="valid", img_size=img_size, is_train=False)
    print(f"[INFO] Majmou3 Data: {len(train_ds)} Train (x2 b H-Flip -> {len(train_ds)*2}) | {len(val_ds)} Valid")

    num_workers = 2 if os.name != "nt" else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_amp, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_amp)

    model = PitchPoseNet(num_keypoints=NUM_KEYPOINTS, pretrained=True, img_size=img_size).to(device)
    criterion = PitchPoseProLoss(img_size=img_size, geom_weight=0.05, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def lr_lambda(ep):
        if ep < WARMUP_EPOCHS:
            return float(ep + 1) / float(max(1, WARMUP_EPOCHS))
        progress = float(ep - WARMUP_EPOCHS) / float(max(1, max_epochs - WARMUP_EPOCHS))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    best_val_px = float("inf")
    no_improve = 0

    print("=" * 115)
    print(f"  BIDAYAT T-TADRIB PRO (ResNet34 @ {img_size}x{img_size} + Broadcast Augs) | Save: {save_path}")
    print("=" * 115)

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_stats = {"tot": 0.0, "px": 0.0, "vis": 0.0}
        for imgs, kpts, vis in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            kpts = kpts.to(device, non_blocking=True)
            vis = vis.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_coords, vis_logits = model(imgs)
                loss, m = criterion(pred_coords, vis_logits, kpts, vis)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            tr_stats["tot"] += m["tot"]
            tr_stats["px"] += m["px"]
            tr_stats["vis"] += m["vis"]

        scheduler.step()
        n_tr = max(1, len(train_loader))

        # Validation
        model.eval()
        va_stats = {"px": 0.0, "pck5": 0.0, "pck10": 0.0, "vis": 0.0}
        with torch.no_grad():
            for imgs, kpts, vis in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                kpts = kpts.to(device, non_blocking=True)
                vis = vis.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_coords, vis_logits = model(imgs)
                    _, m = criterion(pred_coords, vis_logits, kpts, vis)
                va_stats["px"] += m["px"]
                va_stats["pck5"] += m["pck5"]
                va_stats["pck10"] += m["pck10"]
                va_stats["vis"] += m["vis"]

        n_va = max(1, len(val_loader))
        val_px = va_stats["px"] / n_va
        val_pck5 = va_stats["pck5"] / n_va
        val_pck10 = va_stats["pck10"] / n_va
        val_vis = va_stats["vis"] / n_va
        cur_lr = optimizer.param_groups[0]["lr"]

        tag = ""
        if val_px < best_val_px:
            best_val_px = val_px
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            tag = " *BEST*"
        else:
            no_improve += 1

        print(
            f"Ep {epoch:03d}/{max_epochs} | LR:{cur_lr:.1e} | "
            f"Tr[Loss:{tr_stats['tot']/n_tr:.4f} Px:{tr_stats['px']/n_tr:5.2f}] | "
            f"Val[Px:{val_px:5.2f}px PCK5:{val_pck5:5.1f}% PCK10:{val_pck10:5.1f}% Vis:{val_vis:5.1f}%]{tag}"
        )

        if no_improve >= PATIENCE:
            print(f"[EARLY STOP] 7bas f Epoch {epoch} (Ahsan Val PxErr: {best_val_px:.2f} px)")
            break

    print(f"\n[SAFI] Ahsan model t-sauvegarda f: {save_path} (Val PxErr: {best_val_px:.2f} px)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PitchPoseNet Pro Training (ResNet34 @ 320x320)")
    parser.add_argument("--data", nargs="+", default=DATASET_DIRS, help="Dataset directories (default: /content/dataset3)")
    parser.add_argument("--save", type=str, default=SAVE_PATH, help="Output .pth weights path")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="Input image size (default: 320)")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Max epochs (default: 100)")
    args = parser.parse_args()

    train_model(args.data, args.save, img_size=args.img_size, batch_size=args.batch, max_epochs=args.epochs)