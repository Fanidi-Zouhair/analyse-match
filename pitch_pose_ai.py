import os
import glob
import random
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# ==============================================================================
# 1. CONFIGURATION DYAL T-TIRAN (39 Points: dataset + dataset2)
# ==============================================================================
NUM_KEYPOINTS = 39          # 38 noqat l-qdam + noqta 39 (Center Spot f dataset2)
IMG_SIZE = 256
DATASET_DIRS = ["dataset", "dataset2"]  # Kay-jma3hom b-jouj automatiquement!
BATCH_SIZE = 16
MAX_EPOCHS = 120
LEARNING_RATE = 0.001
MASK_PROBABILITY = 0.10
MASK_BOX_HALF = 11          # Moraba3 k7al sghir (22x22 px) bach i-khabi ghir noqta wahda


def set_seed(seed=42):
    """Kat-tabbat l-3achwa2iya bach POSE_BRUT w MATH_PLUS_POSE i-bdaw mn nafs l-Weights!"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 2. PHASE 0: 1-PASS ALL-PAIRS PITCH MEMORY (L-Math Qbal AI Training)
# ==============================================================================
def build_all_pairs_pitch_memory(train_dataset):
    """
    Kat-doz MARRA WAHDA (1 Pass) 3la tsawar dyal Train bo7dha (bla Valid).
    Kat-7sab l-3alaqa bin KOL noqtat irtikaz (i) w KOL noqta okhra (j).
    """
    K = train_dataset.num_keypoints
    sum_ux = np.zeros((K, K), dtype=np.float64)
    sum_uy = np.zeros((K, K), dtype=np.float64)
    sum_dist = np.zeros((K, K), dtype=np.float64)
    counts = np.zeros((K, K), dtype=np.float64)

    for img_path, label_path in train_dataset.samples:
        kpts, vis = train_dataset._parse_yolo_pose_label(label_path)

        vis_idx = np.where(vis > 0.5)[0]
        if len(vis_idx) < 2:
            continue

        pts = kpts[vis_idx]
        diff = pts[None, :, :] - pts[:, None, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1) + 1e-8)

        ux = diff[:, :, 0] / dist
        uy = diff[:, :, 1] / dist

        for a_i, i in enumerate(vis_idx):
            for b_j, j in enumerate(vis_idx):
                if i == j or dist[a_i, b_j] < 1e-4:
                    continue
                sum_ux[i, j] += ux[a_i, b_j]
                sum_uy[i, j] += uy[a_i, b_j]
                sum_dist[i, j] += dist[a_i, b_j]
                counts[i, j] += 1.0

    valid_mask = counts >= 3
    safe_counts = np.where(valid_mask, counts, 1.0)

    mean_ux = np.where(valid_mask, sum_ux / safe_counts, 0.0)
    mean_uy = np.where(valid_mask, sum_uy / safe_counts, 0.0)
    mean_dist = np.where(valid_mask, sum_dist / safe_counts, 0.0)

    consistency = np.sqrt(mean_ux ** 2 + mean_uy ** 2)
    norm_ux = np.where(consistency > 1e-5, mean_ux / np.maximum(consistency, 1e-5), 0.0)
    norm_uy = np.where(consistency > 1e-5, mean_uy / np.maximum(consistency, 1e-5), 0.0)

    mem_conf = np.where(valid_mask & (consistency > 0.70), consistency, 0.0).astype(np.float32)
    mem_dir = np.stack([norm_ux, norm_uy], axis=-1).astype(np.float32)
    mem_dist = mean_dist.astype(np.float32)

    strong_relations = int((mem_conf > 0).sum())
    print(f"[PHASE 0 - MATH MEMORY] 1-Pass salat 3la {len(train_dataset)} tsawar Train!")
    print(f"[PHASE 0 - MATH MEMORY] T3alam {strong_relations} 3alaqa tabta (jiha + zawiya) bin {K} noqta.")
    return mem_dir, mem_dist, mem_conf


# ==============================================================================
# 3. MULTI-DATASET DYAL ROBOFLOW (dataset + dataset2) + 10% RANDOM MASKING
# ==============================================================================
class RoboflowPitchDataset(Dataset):
    def __init__(self, dataset_dirs, split_name="train", img_size=IMG_SIZE, num_keypoints=NUM_KEYPOINTS, is_train=True):
        self.img_size = img_size
        self.num_keypoints = num_keypoints
        self.is_train = is_train
        self.samples = []  # Lista dyal (img_path, label_path) mn ga3 l-datasets

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
                self.samples.append((img_path, label_path))

            print(f"[DATA] Charga {len(found):3d} tsawar mn '{img_dir}'")

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"Ma-lqitch 7ta tswira f '{split_name}' dakhal {dataset_dirs}!"
            )

    def __len__(self):
        return len(self.samples)

    def _parse_yolo_pose_label(self, label_path):
        """
        Kay-qra fichier .txt dyal YOLOv8 Pose:
        Ila kan fichier fih 38 noqta (dataset 1), noqta 39 kat-bqa vis=0 automatiquement!
        """
        kpts = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        vis = np.zeros((self.num_keypoints,), dtype=np.float32)

        if not os.path.exists(label_path):
            return kpts, vis

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            return kpts, vis

        parts = list(map(float, lines[0].split()))
        raw_kpts = parts[5:]

        # Ta7did 3adad l-qiyam l-kol noqta (3 -> x,y,v | 2 -> x,y)
        if len(raw_kpts) % 3 == 0 and len(raw_kpts) >= 60:
            step = 3
        elif len(raw_kpts) % 2 == 0:
            step = 2
        else:
            step = 3

        max_pts = min(self.num_keypoints, len(raw_kpts) // step)
        for i in range(max_pts):
            x = raw_kpts[i * step]
            y = raw_kpts[i * step + 1]
            v = raw_kpts[i * step + 2] if step == 3 else (1.0 if (x > 0 or y > 0) else 0.0)

            if v > 0 and (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0):
                kpts[i, 0] = x
                kpts[i, 1] = y
                vis[i] = 1.0

        return kpts, vis

    def _apply_augmentation(self, img, kpts, vis):
        """
        Data Augmentation f waqt Train:
        1. Random Scale + Shift + Rotation khfifa (kay-badal t-tswira w n-noqat b-jouj)
        2. Color Jitter (taghyir d-daw w l-contrast bhal chams / lil)
        """
        h, w = self.img_size, self.img_size

        # 1. Geometric Augmentation (80% probability)
        if random.random() < 0.80:
            angle = random.uniform(-6.0, 6.0)
            scale = random.uniform(0.85, 1.15)
            tx = random.uniform(-0.08, 0.08) * w
            ty = random.uniform(-0.08, 0.08) * h

            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty

            img = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0.0, 0.0, 0.0)
            )

            # T-ta7wil dyal n-noqat l-baynin b nafs l-Matrice M
            vis_idx = np.where(vis > 0.5)[0]
            if len(vis_idx) > 0:
                pts_px = np.ones((len(vis_idx), 3), dtype=np.float32)
                pts_px[:, 0] = kpts[vis_idx, 0] * w
                pts_px[:, 1] = kpts[vis_idx, 1] * h
                new_pts = pts_px @ M.T  # (N, 2)
                new_x = new_pts[:, 0] / w
                new_y = new_pts[:, 1] / h

                for k_i, idx_pt in enumerate(vis_idx):
                    if 0.0 <= new_x[k_i] <= 1.0 and 0.0 <= new_y[k_i] <= 1.0:
                        kpts[idx_pt, 0] = new_x[k_i]
                        kpts[idx_pt, 1] = new_y[k_i]
                    else:
                        # Ila kharjat n-noqta mn l-cadre b sabab z-zoom/shift -> kat-walli vis=0
                        kpts[idx_pt] = 0.0
                        vis[idx_pt] = 0.0

        # 2. Color Jitter (50% probability)
        if random.random() < 0.50:
            alpha = random.uniform(0.80, 1.20)  # Contrast
            beta = random.uniform(-0.10, 0.10)  # Brightness
            img = np.clip(img * alpha + beta, 0.0, 1.0)

        return img, kpts, vis

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Ma-qdartch n-qra t-tswira: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0

        kpts, vis = self._parse_yolo_pose_label(label_path)

        # Data Augmentation f Train bo7dha
        if self.is_train:
            img, kpts, vis = self._apply_augmentation(img, kpts, vis)

        # 10% MASKING f waqt t-tadrib (moraba3 sghir 22x22 px)
        was_masked = 0
        if self.is_train and random.random() < MASK_PROBABILITY:
            visible_indices = np.where(vis > 0.5)[0]
            if len(visible_indices) >= 4:
                hide_idx = random.choice(visible_indices)
                px = int(kpts[hide_idx, 0] * self.img_size)
                py = int(kpts[hide_idx, 1] * self.img_size)
                y1, y2 = max(0, py - MASK_BOX_HALF), min(self.img_size, py + MASK_BOX_HALF)
                x1, x2 = max(0, px - MASK_BOX_HALF), min(self.img_size, px + MASK_BOX_HALF)
                img[y1:y2, x1:x2, :] = 0.0
                was_masked = 1

        img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        kpts_tensor = torch.from_numpy(kpts)
        vis_tensor = torch.from_numpy(vis)

        return img_tensor, kpts_tensor, vis_tensor, was_masked


# ==============================================================================
# 4. UPGRADE PRO: PRETRAINED RESNET18 U-NET + SPATIAL SOFTMAX + VISIBILITY HEAD
# ==============================================================================
class PitchPoseNet(nn.Module):
    """
    Encoder: Pretrained ResNet18 (kay-fham l-khotout w l-ashkal mn ImageNet)
    Decoder: U-Net Skip Connections -> 39 Heatmaps (64x64) -> Spatial Softmax (x, y)
    Visibility Head: Kay-kharraj confidence (bayna wla la) l-kol noqta mn 39!
    """
    def __init__(self, num_keypoints=NUM_KEYPOINTS, pretrained=True):
        super().__init__()
        self.num_keypoints = num_keypoints

        # ImageNet normalization buffers (dakhal l-model bach t-bqa data [0,1] nqiya)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # --- ENCODER (Pretrained ResNet18) ---
        try:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        except Exception:
            backbone = resnet18(weights=None)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)  # 64x64
        self.enc1 = backbone.layer1  # (B, 64, 64, 64)
        self.enc2 = backbone.layer2  # (B, 128, 32, 32)
        self.enc3 = backbone.layer3  # (B, 256, 16, 16)
        self.enc4 = backbone.layer4  # (B, 512, 8, 8)

        # --- DECODER b Skip Connections ---
        self.dec3 = nn.Sequential(
            nn.Conv2d(512 + 256, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(192 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Head 1: 39 Heatmaps (64x64) -> (x, y)
        self.heatmap_head = nn.Conv2d(64, num_keypoints, kernel_size=1)

        # Head 2: Visibility / Confidence Head -> (B, 39) logits
        self.vis_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_keypoints)
        )

        # Grid dyal i7datiyat (x, y) bin 0 w 1 l Spatial Softmax (64x64)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, 64),
            torch.linspace(0.0, 1.0, 64),
            indexing="ij"
        )
        self.register_buffer("grid_x", grid_x.reshape(1, 1, -1))
        self.register_buffer("grid_y", grid_y.reshape(1, 1, -1))

    def forward(self, x):
        # ImageNet Normalization
        x = (x - self.img_mean) / self.img_std

        # Encoder
        s = self.stem(x)    # (B, 64, 64, 64)
        e1 = self.enc1(s)   # (B, 64, 64, 64)
        e2 = self.enc2(e1)  # (B, 128, 32, 32)
        e3 = self.enc3(e2)  # (B, 256, 16, 16)
        e4 = self.enc4(e3)  # (B, 512, 8, 8)

        # Visibility logits (wach kol noqta bayna f t-tswira?)
        vis_logits = self.vis_head(e4)  # (B, 39)

        # Decoder + Skip Connections
        d3 = F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False)  # 16x16
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)  # 32x32
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)  # 64x64
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Heatmaps -> Spatial Softmax -> (x, y)
        heatmaps = self.heatmap_head(d1)
        B, K, H, W = heatmaps.shape
        probs = F.softmax(heatmaps.view(B, K, -1), dim=-1)

        pred_x = (probs * self.grid_x).sum(dim=-1)
        pred_y = (probs * self.grid_y).sum(dim=-1)
        coords = torch.stack([pred_x, pred_y], dim=-1)  # (B, 39, 2)

        return coords, vis_logits


# ==============================================================================
# 5. LOSS FUNCTION M-SA77A: POSE_BRUT vs MATH+POSE (+ VISIBILITY + PCK@10px)
# ==============================================================================
class PitchBenchmarkLoss(nn.Module):
    def __init__(self, use_math=True, mem_dir=None, mem_conf=None, device="cpu"):
        super().__init__()
        self.use_math = use_math
        self.device = device
        K = NUM_KEYPOINTS
        self.off_diag = (1.0 - torch.eye(K, device=device)).unsqueeze(0)

        if mem_dir is not None and mem_conf is not None:
            self.mem_dir = torch.from_numpy(mem_dir).unsqueeze(0).to(device)
            self.mem_conf = torch.from_numpy(mem_conf).unsqueeze(0).to(device)
        else:
            self.mem_dir = None
            self.mem_conf = None

    def forward(self, pred, vis_logits, target, vis):
        # 1. POSE LOSS (SmoothL1 3la n-noqat li baynin)
        vis_mask = vis.unsqueeze(-1)
        valid_pts = vis_mask.sum().clamp(min=1.0) * 2.0
        point_loss = (F.smooth_l1_loss(pred, target, beta=0.05, reduction="none") * vis_mask).sum() / valid_pts

        # 2. VISIBILITY LOSS (BCEWithLogits) + Accuracy
        vis_loss = F.binary_cross_entropy_with_logits(vis_logits, vis)
        vis_pred_bin = (torch.sigmoid(vis_logits) > 0.5).float()
        vis_acc = (vis_pred_bin == (vis > 0.5).float()).float().mean() * 100.0

        # 3. METRICS B L-PIXELS: Mean PxErr + PCK@10px (% dyal n-noqat s7a7 qal mn 10 px)
        px_diff = (pred - target) * IMG_SIZE
        px_dist = torch.sqrt((px_diff ** 2).sum(dim=-1) + 1e-8)
        vis_sum = vis.sum().clamp(min=1.0)
        mean_px_err = (px_dist * vis).sum() / vis_sum
        pck_10 = (((px_dist <= 10.0).float()) * vis).sum() / vis_sum * 100.0

        # 4. ALL-PAIRS MATH METRICS
        diff_pred = pred.unsqueeze(1) - pred.unsqueeze(2)
        diff_real = target.unsqueeze(1) - target.unsqueeze(2)
        pair_vis = (vis.unsqueeze(2) * vis.unsqueeze(1)) * self.off_diag
        valid_pairs = pair_vis.sum().clamp(min=1.0)

        dist_pred = torch.sqrt((diff_pred ** 2).sum(dim=-1) + 1e-6)
        dist_real = torch.sqrt((diff_real ** 2).sum(dim=-1) + 1e-6)
        dist_loss = (((dist_pred - dist_real) ** 2) * pair_vis).sum() / valid_pairs

        dir_pred = diff_pred / dist_pred.unsqueeze(-1).clamp(min=1e-4)
        dir_real = diff_real / dist_real.unsqueeze(-1).clamp(min=1e-4)
        cos_sim = (dir_pred * dir_real).sum(dim=-1).clamp(-0.9999, 0.9999)
        angle_err_deg = (torch.acos(cos_sim) * (180.0 / math.pi) * pair_vis).sum() / valid_pairs
        dir_loss = ((1.0 - cos_sim) * pair_vis).sum() / valid_pairs

        # BUG FIXED: Phase 0 Memory Loss m-filtriya b `pair_vis` bach ma-t-3aqabch n-noqat li kharij l-cadre!
        if self.mem_dir is not None and self.mem_conf is not None:
            cos_mem = (dir_pred * self.mem_dir).sum(dim=-1).clamp(-0.9999, 0.9999)
            mem_weight = self.mem_conf * pair_vis
            memory_loss = (F.relu(0.80 - cos_mem) * mem_weight).sum() / mem_weight.sum().clamp(min=1.0)
        else:
            memory_loss = torch.tensor(0.0, device=self.device)

        # 5. TOTAL LOSS
        if not self.use_math:
            # NOSKHA 1: POSE BRUT + VISIBILITY (0% Math)
            total_loss = point_loss + 0.15 * vis_loss
        else:
            # NOSKHA 2: MATH + POSE + VISIBILITY (Masafat + Zawaya + Dakirat Phase 0 m-sa77a)
            total_loss = point_loss + 0.15 * vis_loss + 1.0 * dist_loss + 0.5 * dir_loss + 0.15 * memory_loss

        stats = {
            "total": total_loss.item(),
            "pose": point_loss.item(),
            "vis_l": vis_loss.item(),
            "vis_acc": vis_acc.item(),
            "dist": dist_loss.item(),
            "dir": dir_loss.item(),
            "mem": memory_loss.item(),
            "px_err": mean_px_err.item(),
            "pck10": pck_10.item(),
            "ang_deg": angle_err_deg.item(),
        }
        return total_loss, stats


# ==============================================================================
# 6. TADRIB NOSKHA WAHDA B LOG M-FASSAL F KOL EPOCH
# ==============================================================================
def train_experiment(exp_name, use_math, save_path, train_loader, valid_loader, mem_dir, mem_conf, device):
    set_seed(42)  # Nafs l-bdaya l-jouj noskhat bach t-koun moqarana 3adila 100%!

    model = PitchPoseNet(num_keypoints=NUM_KEYPOINTS, pretrained=True).to(device)
    criterion = PitchBenchmarkLoss(
        use_math=use_math,
        mem_dir=mem_dir if use_math else None,
        mem_conf=mem_conf if use_math else None,
        device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    best_val_px = float("inf")
    patience = 20
    patience_counter = 0

    print("\n" + "=" * 125)
    print(f"  BIDAYAT T-TADRIB: [{exp_name}]  |  Save: {save_path}")
    print("=" * 125)

    for epoch in range(1, MAX_EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        tr_tot = tr_pose = tr_dist = tr_dir = tr_mem = tr_px = 0.0
        masked_count = 0

        for imgs, kpts, vis, was_masked in train_loader:
            imgs, kpts, vis = imgs.to(device), kpts.to(device), vis.to(device)
            masked_count += int(was_masked.sum().item())

            optimizer.zero_grad()
            preds, vis_logits = model(imgs)
            loss, st = criterion(preds, vis_logits, kpts, vis)
            loss.backward()
            optimizer.step()

            tr_tot += st["total"]
            tr_pose += st["pose"]
            tr_dist += st["dist"]
            tr_dir += st["dir"]
            tr_mem += st["mem"]
            tr_px += st["px_err"]

        n_tr = len(train_loader)
        tr_tot /= n_tr
        tr_pose /= n_tr
        tr_dist /= n_tr
        tr_dir /= n_tr
        tr_mem /= n_tr
        tr_px /= n_tr

        # --- VALIDATION ---
        model.eval()
        val_pose = val_px = val_pck = val_ang = val_vacc = 0.0
        with torch.no_grad():
            for imgs, kpts, vis, _ in valid_loader:
                imgs, kpts, vis = imgs.to(device), kpts.to(device), vis.to(device)
                preds, vis_logits = model(imgs)
                _, st = criterion(preds, vis_logits, kpts, vis)
                val_pose += st["pose"]
                val_px += st["px_err"]
                val_pck += st["pck10"]
                val_ang += st["ang_deg"]
                val_vacc += st["vis_acc"]

        n_val = len(valid_loader)
        val_pose /= n_val
        val_px /= n_val
        val_pck /= n_val
        val_ang /= n_val
        val_vacc /= n_val

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_px)

        star = ""
        if val_px < best_val_px:
            best_val_px = val_px
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            star = " *BEST*"
        else:
            patience_counter += 1

        # LOG M-FASSAL F KOL EPOCH
        print(
            f"Ep {epoch:03d} | LR:{current_lr:.1e} | "
            f"Tr[Tot:{tr_tot:.4f} Pose:{tr_pose:.4f} Dist:{tr_dist:.4f} Ang:{tr_dir:.4f} Mem:{tr_mem:.4f} Px:{tr_px:5.1f}] | "
            f"Val[Px:{val_px:5.1f}px PCK10:{val_pck:5.1f}% Vis:{val_vacc:5.1f}% Ang:{val_ang:5.2f}°] | Msk:{masked_count:2d}{star}"
        )

        if patience_counter >= patience:
            print(f"[EARLY STOP] [{exp_name}] 7bas f Epoch {epoch} (Ahsan Val PxErr: {best_val_px:.2f} px)")
            break

    return best_val_px


# ==============================================================================
# 7. MOQARANA NIHA2IYA: POSE BRUT vs MATH + POSE (Scoreboard + Tswira)
# ==============================================================================
def evaluate_and_compare(valid_dataset, device):
    model_pose = PitchPoseNet(num_keypoints=NUM_KEYPOINTS, pretrained=False).to(device)
    model_pose.load_state_dict(torch.load("best_pose_brut.pth", map_location=device, weights_only=True))
    model_pose.eval()

    model_math = PitchPoseNet(num_keypoints=NUM_KEYPOINTS, pretrained=False).to(device)
    model_math.load_state_dict(torch.load("best_math_pose.pth", map_location=device, weights_only=True))
    model_math.eval()

    crit_eval = PitchBenchmarkLoss(use_math=False, device=device)

    def eval_model_full(model):
        all_px, all_pck, all_vacc, all_ang, masked_px = [], [], [], [], []
        with torch.no_grad():
            for i in range(len(valid_dataset)):
                img_t, kpts_t, vis_t, _ = valid_dataset[i]
                inp = img_t.unsqueeze(0).to(device)
                kt = kpts_t.unsqueeze(0).to(device)
                vt = vis_t.unsqueeze(0).to(device)
                pred, vis_logits = model(inp)
                _, st = crit_eval(pred, vis_logits, kt, vt)
                all_px.append(st["px_err"])
                all_pck.append(st["pck10"])
                all_vacc.append(st["vis_acc"])
                all_ang.append(st["ang_deg"])

                # Test b noqta m-khabiya
                vis_idx = np.where(vis_t.numpy() > 0.5)[0]
                if len(vis_idx) > 0:
                    hide_idx = int(vis_idx[0])
                    img_m = np.transpose(img_t.numpy(), (1, 2, 0)).copy()
                    hx = int(kpts_t[hide_idx, 0].item() * IMG_SIZE)
                    hy = int(kpts_t[hide_idx, 1].item() * IMG_SIZE)
                    y1, y2 = max(0, hy - MASK_BOX_HALF), min(IMG_SIZE, hy + MASK_BOX_HALF)
                    x1, x2 = max(0, hx - MASK_BOX_HALF), min(IMG_SIZE, hx + MASK_BOX_HALF)
                    img_m[y1:y2, x1:x2, :] = 0.0

                    inp_m = torch.from_numpy(np.transpose(img_m, (2, 0, 1))).unsqueeze(0).to(device)
                    pred_m, _ = model(inp_m)
                    pred_m = pred_m.squeeze(0).cpu().numpy()
                    err = math.hypot(
                        (kpts_t[hide_idx, 0].item() - pred_m[hide_idx, 0]) * IMG_SIZE,
                        (kpts_t[hide_idx, 1].item() - pred_m[hide_idx, 1]) * IMG_SIZE
                    )
                    masked_px.append(err)

        return (
            float(np.mean(all_px)),
            float(np.mean(all_pck)),
            float(np.mean(all_vacc)),
            float(np.mean(all_ang)),
            float(np.mean(masked_px)),
        )

    p_px, p_pck, p_vacc, p_ang, p_msk = eval_model_full(model_pose)
    m_px, m_pck, m_vacc, m_ang, m_msk = eval_model_full(model_math)

    print("\n" + "=" * 82)
    print("  SCOREBOARD NIHA2I: POSE BRUT vs MATH + POSE (3la GA3 tsawar Valid)")
    print("=" * 82)
    print(f"  {'Metric':32s} | {'POSE BRUT':>14s} | {'MATH + POSE':>14s} | {'R-RABEH':>10s}")
    print("-" * 82)
    w_px = "MATH+POSE" if m_px < p_px else "POSE BRUT"
    w_pck = "MATH+POSE" if m_pck > p_pck else "POSE BRUT"
    w_vacc = "MATH+POSE" if m_vacc > p_vacc else "POSE BRUT"
    w_ang = "MATH+POSE" if m_ang < p_ang else "POSE BRUT"
    w_msk = "MATH+POSE" if m_msk < p_msk else "POSE BRUT"
    print(f"  {'Khata2 Motawassit (px)':32s} | {p_px:11.2f} px | {m_px:11.2f} px | {w_px:>10s}")
    print(f"  {'PCK@10px (% S7i7 < 10px)':32s} | {p_pck:13.1f}% | {m_pck:13.1f}% | {w_pck:>10s}")
    print(f"  {'Visibility Accuracy (%)':32s} | {p_vacc:13.1f}% | {m_vacc:13.1f}% | {w_vacc:>10s}")
    print(f"  {'Khata2 Zawiya (deg)':32s} | {p_ang:13.2f}° | {m_ang:13.2f}° | {w_ang:>10s}")
    print(f"  {'Khata2 Noqta M-khabiya (px)':32s} | {p_msk:11.2f} px | {m_msk:11.2f} px | {w_msk:>10s}")
    print("=" * 82)

    # N-rasmo tswira m-qarna: Lisr = POSE BRUT | Limn = MATH + POSE
    img_t, kpts_t, vis_t, _ = valid_dataset[0]
    img_np = np.transpose(img_t.numpy(), (1, 2, 0)).copy()
    vis_idx = np.where(vis_t.numpy() > 0.5)[0]

    def draw_preds(model, title):
        inp = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)
        with torch.no_grad():
            preds, vis_logits = model(inp)
            preds = preds.squeeze(0).cpu().numpy()
            vis_probs = torch.sigmoid(vis_logits).squeeze(0).cpu().numpy()
        canvas = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        canvas = cv2.resize(canvas, (512, 512))
        for idx in vis_idx:
            rx, ry = int(kpts_t[idx, 0].item() * 512), int(kpts_t[idx, 1].item() * 512)
            px, py = int(preds[idx, 0] * 512), int(preds[idx, 1] * 512)
            cv2.circle(canvas, (rx, ry), 5, (0, 255, 0), -1)
            # 7mar ila confidence >= 0.5, Borta9ali ila < 0.5
            color = (0, 0, 255) if vis_probs[idx] >= 0.5 else (0, 165, 255)
            cv2.circle(canvas, (px, py), 4, color, -1)
            cv2.line(canvas, (rx, ry), (px, py), (255, 255, 0), 1)
            cv2.putText(canvas, f"{idx + 1}({int(vis_probs[idx]*100)}%)", (rx + 5, ry - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        return canvas

    left_img = draw_preds(model_pose, "POSE BRUT (ResNet18 + 0% Math)")
    right_img = draw_preds(model_math, "MATH + POSE (ResNet18 + Fixed Math)")
    combined = np.hstack([left_img, right_img])
    cv2.imwrite("comparison_pose_vs_math.jpg", combined)
    print("\n[CHOUF T-TSWIRA] T-sauvegardat f: comparison_pose_vs_math.jpg (Lisr=Pose Brut | Limn=Math+Pose)")


def run_benchmark():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] L-khedma 3la: {device}")

        train_dataset = RoboflowPitchDataset(DATASET_DIRS, split_name="train", is_train=True)
        valid_dataset = RoboflowPitchDataset(DATASET_DIRS, split_name="valid", is_train=False)
        print(f"[INFO] Majmou3 Data: {len(train_dataset)} Train | {len(valid_dataset)} Valid")

        # Phase 0 (kat-khdam ghir marra wahda w kan-sta3mloha f Noskha 2 bo7dha)
        mem_dir, mem_dist, mem_conf = build_all_pairs_pitch_memory(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 1. TADRIB NOSKHA 1: POSE BRUT BO7DO
        train_experiment(
            exp_name="1. POSE BRUT (0% Math)",
            use_math=False,
            save_path="best_pose_brut.pth",
            train_loader=train_loader,
            valid_loader=valid_loader,
            mem_dir=None,
            mem_conf=None,
            device=device
        )

        # 2. TADRIB NOSKHA 2: MATH + POSE F DAQQA
        train_experiment(
            exp_name="2. MATH + POSE (Phase 0 + All-Pairs)",
            use_math=True,
            save_path="best_math_pose.pth",
            train_loader=train_loader,
            valid_loader=valid_loader,
            mem_dir=mem_dir,
            mem_conf=mem_conf,
            device=device
        )

        # 3. MOQARANA NIHA2IYA
        evaluate_and_compare(valid_dataset, device)

    except Exception as e:
        print(f"[ERREUR] {e}")


if __name__ == "__main__":
    run_benchmark()