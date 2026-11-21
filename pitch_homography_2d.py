import cv2
import numpy as np

# ==============================================================================
# FIFA STANDARD PITCH DIMENSIONS (105m x 68m) + 39-KEYPOINT COORDINATES
# ==============================================================================
PITCH_L = 105.0
PITCH_W = 68.0

# I7datiyat FIFA b l-meter (X: 0..105, Y: 0..68) l kol noqta (1-indexed)
PITCH_COORDS_M = {
    # --- L-Jiha l-Lisra ---
    17: (0.0, 0.0),         # Corner foq lisr
    16: (0.0, 13.84),       # 18m foq 3la l-khat
    15: (0.0, 24.84),       # 6m foq 3la l-khat
    12: (0.0, 43.16),       # 6m taht 3la l-khat
    11: (0.0, 54.16),       # 18m taht 3la l-khat
    10: (0.0, 68.0),        # Corner taht lisr
    36: (5.5, 24.84),       # 6m foq dakhal
    37: (5.5, 43.16),       # 6m taht dakhal
    31: (16.5, 13.84),      # 18m foq dakhal
    32: (16.5, 26.69),      # Qaws 18m foq lisr
    34: (16.5, 41.31),      # Qaws 18m taht lisr
    35: (16.5, 54.16),      # 18m taht dakhal
    33: (20.15, 34.0),      # Ras l-qaws lisr

    # --- L-Wast ---
    18: (52.5, 0.0),        # Khat l-wast foq
    19: (52.5, 24.85),      # Dayra foq
    21: (43.35, 34.0),      # Dayra lisr
    39: (52.5, 34.0),       # Markaz d-dayra (Center Spot)
    20: (61.65, 34.0),      # Dayra limn
    22: (52.5, 43.15),      # Dayra taht
    9:  (52.5, 68.0),       # Khat l-wast taht

    # --- L-Jiha l-Limna ---
    1:  (105.0, 0.0),       # Corner foq limn
    2:  (105.0, 13.84),     # 18m foq 3la l-khat
    3:  (105.0, 24.84),     # 6m foq 3la l-khat
    6:  (105.0, 43.16),     # 6m taht 3la l-khat
    7:  (105.0, 54.16),     # 18m taht 3la l-khat
    8:  (105.0, 68.0),      # Corner taht limn
    28: (99.5, 24.84),      # 6m foq dakhal
    29: (99.5, 43.16),      # 6m taht dakhal
    23: (88.5, 13.84),      # 18m foq dakhal
    24: (88.5, 26.69),      # Qaws 18m foq limn
    26: (88.5, 41.31),      # Qaws 18m taht limn
    27: (88.5, 54.16),      # 18m taht dakhal
    25: (84.85, 34.0),      # Ras l-qaws limn
}

# Faradiyat riyadiya (exact full Multi-Hypothesis RANSAC bla taqsir)
REMAP_HYPOTHESES = [
    ("raw", {}),
    ("arc_left_fix", {26: 33, 25: 33}),
    ("center_from_right", {1: 18, 2: 19, 23: 21, 27: 22, 8: 9, 26: 33, 25: 33}),
    ("center_from_left", {17: 18, 16: 19, 31: 20, 35: 22, 10: 9, 34: 25, 33: 25}),
]


class PitchHomography2D:
    """
    Module khass ghir b Homography riyadiya + t-tiran 2D:
    - Full Multi-Hypothesis Projective RANSAC (diqqa kamla 100%)
    - Anti-Swap & Orientation/Aspect Sanity Check
    - Geometric 4-corner EMA smoothing + ROI-only FOV blending
    """

    def __init__(
        self,
        min_conf=0.45,
        fallback_conf=0.35,
        ransac_px_thresh=24.0,
        max_reproj_px=26.0,
        smooth_alpha=0.45,
        max_hold_frames=12,
        scale=5,
        pad=16,
    ):
        self.min_conf = min_conf
        self.fallback_conf = fallback_conf
        self.ransac_px_thresh = ransac_px_thresh
        self.max_reproj_px = max_reproj_px
        self.smooth_alpha = smooth_alpha
        self.max_hold_frames = max_hold_frames
        self.scale = scale
        self.pad = pad

        self.H_p2i = None
        self.H_i2p = None
        self._smooth_ref_img = None
        self._ref_pitch_pts = np.array(
            [[16.5, 13.84], [88.5, 13.84], [88.5, 54.16], [16.5, 54.16]],
            dtype=np.float32,
        )
        self.hold_count = 0

        # Pre-compute homogeneous pitch grid (28x20) + minimap pixels (lossless speedup)
        gx = np.linspace(0.0, PITCH_L, 28)
        gy = np.linspace(0.0, PITCH_W, 20)
        xx, yy = np.meshgrid(gx, gy)
        self._pitch_grid = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        self._pitch_grid_h = np.concatenate(
            [self._pitch_grid, np.ones((len(self._pitch_grid), 1), dtype=np.float32)], axis=1
        )
        self._grid_mm_px = np.stack([
            np.round(self._pitch_grid[:, 0] * self.scale + self.pad),
            np.round(self._pitch_grid[:, 1] * self.scale + self.pad),
        ], axis=-1).astype(np.int32)

        self.template = self._build_pitch_template()

    # ------------------------------------------------------------------
    # 1. FILTRAGE ANTI-SWAP + MULTI-HYPOTHESIS PROJECTIVE SOLVER
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_remap(raw_valid, remap_dict):
        if not remap_dict:
            return list(raw_valid)
        used_ids = set()
        out = []
        for k_id, x, y, conf in raw_valid:
            if k_id not in remap_dict:
                used_ids.add(k_id)
                out.append((k_id, x, y, conf))
        for k_id, x, y, conf in raw_valid:
            if k_id in remap_dict:
                new_id = remap_dict[k_id]
                if new_id not in used_ids:
                    used_ids.add(new_id)
                    out.append((new_id, x, y, conf))
        return out

    @staticmethod
    def _filter_spatial_outliers(valid_list):
        pts = list(valid_list)

        left_pts = [p for p in pts if PITCH_COORDS_M[p[0]][0] < 35.0]
        right_pts = [p for p in pts if PITCH_COORDS_M[p[0]][0] > 70.0]
        if len(left_pts) >= 4 and 0 < len(right_pts) <= 2:
            pts = [p for p in pts if PITCH_COORDS_M[p[0]][0] <= 70.0]
        elif len(right_pts) >= 4 and 0 < len(left_pts) <= 2:
            pts = [p for p in pts if PITCH_COORDS_M[p[0]][0] >= 35.0]

        while len(pts) >= 5:
            n = len(pts)
            violations = np.zeros(n, dtype=np.int32)
            for i in range(n):
                ki, xi, yi, _ = pts[i]
                pxi, pyi = PITCH_COORDS_M[ki]
                for j in range(i + 1, n):
                    kj, xj, yj, _ = pts[j]
                    pxj, pyj = PITCH_COORDS_M[kj]

                    if abs(pyj - pyi) <= 12.0:
                        if (pxj - pxi) > 8.0 and (xj < xi - 15.0):
                            violations[i] += 1
                            violations[j] += 1
                        elif (pxi - pxj) > 8.0 and (xi < xj - 15.0):
                            violations[i] += 1
                            violations[j] += 1

                    if abs(pxj - pxi) <= 12.0:
                        if (pyj - pyi) > 8.0 and (yj < yi - 15.0):
                            violations[i] += 1
                            violations[j] += 1
                        elif (pyi - pyj) > 8.0 and (yi < yj - 15.0):
                            violations[i] += 1
                            violations[j] += 1

            worst_idx = int(np.argmax(violations))
            if violations[worst_idx] == 0:
                break
            pts.pop(worst_idx)

        return pts

    @staticmethod
    def _is_non_collinear(pitch_pts):
        return (
            (float(np.max(pitch_pts[:, 0]) - np.min(pitch_pts[:, 0])) >= 4.0)
            and (float(np.max(pitch_pts[:, 1]) - np.min(pitch_pts[:, 1])) >= 4.0)
        )

    def _is_valid_homography(self, H_p2i, pitch_pts_inliers):
        if H_p2i is None or not np.all(np.isfinite(H_p2i)):
            return False
        if abs(float(np.linalg.det(H_p2i))) < 1e-8:
            return False

        cx = float(np.clip(np.mean(pitch_pts_inliers[:, 0]), 5.0, PITCH_L - 5.0))
        cy = float(np.clip(np.mean(pitch_pts_inliers[:, 1]), 5.0, PITCH_W - 5.0))
        patch = np.array(
            [[[cx - 2.0, cy - 2.0]], [[cx + 2.0, cy - 2.0]],
             [[cx + 2.0, cy + 2.0]], [[cx - 2.0, cy + 2.0]]],
            dtype=np.float32,
        )
        proj = cv2.perspectiveTransform(patch, H_p2i).reshape(-1, 2)
        if not np.all(np.isfinite(proj)):
            return False

        x, y = proj[:, 0], proj[:, 1]
        area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if area <= 10.0:
            return False

        len_x = float(np.linalg.norm(proj[1] - proj[0]))
        len_y = float(np.linalg.norm(proj[3] - proj[0]))
        if len_x < 5.0 or len_y < 2.0 or not (0.10 <= (len_y / len_x) <= 2.20):
            return False

        return (proj[1, 0] - proj[0, 0]) > 0.0 and (proj[3, 1] - proj[0, 1]) > 0.0

    def _solve_single_hypothesis(self, pts_list, ransac_thresh, max_err_thresh, total_raw):
        valid = self._filter_spatial_outliers(pts_list)
        if len(valid) < 4:
            return None

        pitch_pts = np.array([PITCH_COORDS_M[k_id] for (k_id, _, _, _) in valid], dtype=np.float32)
        img_pts = np.array([[x, y] for (_, x, y, _) in valid], dtype=np.float32)

        if not self._is_non_collinear(pitch_pts):
            return None

        # Full exact RANSAC (default maxIters=2000, confidence=0.995) — zero accuracy compromise
        H_cand, mask = cv2.findHomography(
            pitch_pts, img_pts, cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
        )
        if H_cand is None or mask is None:
            return None

        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())

        min_needed = 4 if total_raw <= 5 else max(5, int(np.ceil(len(valid) * 0.55)))
        if inliers < min_needed:
            return None
        if not self._is_valid_homography(H_cand, pitch_pts[inlier_mask]):
            return None

        if inliers > 4:
            H_ref, _ = cv2.findHomography(pitch_pts[inlier_mask], img_pts[inlier_mask], 0)
            if self._is_valid_homography(H_ref, pitch_pts[inlier_mask]):
                H_cand = H_ref

        proj_img = cv2.perspectiveTransform(
            pitch_pts[inlier_mask].reshape(-1, 1, 2), H_cand
        ).reshape(-1, 2)
        reproj_px = float(np.mean(np.linalg.norm(proj_img - img_pts[inlier_mask], axis=1)))
        if reproj_px > max_err_thresh:
            return None

        score = inliers * 100.0 - reproj_px * 1.5
        return {
            "H_p2i": H_cand,
            "inliers": inliers,
            "reproj_px": reproj_px,
            "inlier_pitch": pitch_pts[inlier_mask],
            "score": score,
        }

    def _smooth_homography(self, H_p2i_new, pitch_pts_inliers):
        ref_img_new = cv2.perspectiveTransform(
            self._ref_pitch_pts.reshape(-1, 1, 2), H_p2i_new
        ).reshape(-1, 2)

        if self._smooth_ref_img is None:
            self._smooth_ref_img = ref_img_new
            return H_p2i_new

        jump = float(np.mean(np.linalg.norm(ref_img_new - self._smooth_ref_img, axis=1)))
        if jump > 180.0:
            self._smooth_ref_img = ref_img_new
            return H_p2i_new

        self._smooth_ref_img = (
            self.smooth_alpha * ref_img_new + (1.0 - self.smooth_alpha) * self._smooth_ref_img
        )
        H_smooth = cv2.getPerspectiveTransform(
            self._ref_pitch_pts.astype(np.float32),
            self._smooth_ref_img.astype(np.float32),
        )
        return H_smooth if self._is_valid_homography(H_smooth, pitch_pts_inliers) else H_p2i_new

    def reset(self):
        self.H_p2i = None
        self.H_i2p = None
        self._smooth_ref_img = None
        self.hold_count = 0

    def update(self, kpts_px, frame_w, frame_h):
        px_scale = max(frame_w, frame_h) / 1920.0
        ransac_thresh = self.ransac_px_thresh * px_scale
        max_err_thresh = self.max_reproj_px * px_scale

        raw_valid = [
            (k_id, v[0], v[1], v[2])
            for k_id, v in kpts_px.items()
            if k_id in PITCH_COORDS_M and v[2] >= self.min_conf
        ]
        if len(raw_valid) < 4:
            raw_valid = [
                (k_id, v[0], v[1], v[2])
                for k_id, v in kpts_px.items()
                if k_id in PITCH_COORDS_M and v[2] >= self.fallback_conf
            ]

        best = None
        best_name = "raw"

        if len(raw_valid) >= 4:
            raw_ids = {p[0] for p in raw_valid}
            n_raw = len(raw_valid)

            for hyp_name, remap_dict in REMAP_HYPOTHESES:
                # Skip ONLY if remap_dict touches zero detected keypoints (100% lossless)
                if remap_dict and not (raw_ids & remap_dict.keys()):
                    continue

                cand_pts = self._apply_remap(raw_valid, remap_dict)
                sol = self._solve_single_hypothesis(cand_pts, ransac_thresh, max_err_thresh, total_raw=n_raw)
                if sol is None:
                    continue

                adj_score = sol["score"] + (15.0 if hyp_name == "raw" else 0.0)
                if best is None or adj_score > best["adj_score"]:
                    sol["adj_score"] = adj_score
                    best = sol
                    best_name = hyp_name

        if best is not None:
            H_p2i_smooth = self._smooth_homography(best["H_p2i"], best["inlier_pitch"])
            try:
                H_i2p_smooth = np.linalg.inv(H_p2i_smooth)
                H_i2p_smooth = H_i2p_smooth / H_i2p_smooth[2, 2]
                self.H_p2i = H_p2i_smooth
                self.H_i2p = H_i2p_smooth
                self.hold_count = 0
                return self.H_i2p, self.H_p2i, {
                    "status": "OK",
                    "inliers": best["inliers"],
                    "total": len(raw_valid),
                    "reproj_px": best["reproj_px"],
                    "hypothesis": best_name,
                }
            except np.linalg.LinAlgError:
                pass

        if self.H_i2p is not None and self.hold_count < self.max_hold_frames:
            self.hold_count += 1
            return self.H_i2p, self.H_p2i, {
                "status": "HOLD",
                "inliers": 0,
                "total": len(raw_valid),
                "reproj_px": 0.0,
                "hypothesis": "hold",
            }

        self.reset()
        return None, None, {
            "status": "N/A",
            "inliers": 0,
            "total": len(raw_valid),
            "reproj_px": 0.0,
            "hypothesis": "none",
        }

    # ------------------------------------------------------------------
    # 2. TA7WIL I7DATIYAT (Image <-> Pitch 2D)
    # ------------------------------------------------------------------
    def image_to_pitch(self, pts_xy):
        if self.H_i2p is None or len(pts_xy) == 0:
            return np.empty((0, 2), dtype=np.float32)
        arr = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(arr, self.H_i2p).reshape(-1, 2)

    def pitch_to_image(self, pts_m):
        if self.H_p2i is None or len(pts_m) == 0:
            return np.empty((0, 2), dtype=np.float32)
        arr = np.asarray(pts_m, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(arr, self.H_p2i).reshape(-1, 2)

    # ------------------------------------------------------------------
    # 3. RASM T-TIRAN 2D BO7DO (EA FC / FC27 STYLE)
    # ------------------------------------------------------------------
    def _m2px(self, xm, ym):
        return (self.pad + int(round(xm * self.scale)), self.pad + int(round(ym * self.scale)))

    def _build_pitch_template(self):
        pw = int(round(PITCH_L * self.scale))
        ph = int(round(PITCH_W * self.scale))
        canvas = np.zeros((ph + 2 * self.pad, pw + 2 * self.pad, 3), dtype=np.uint8)
        canvas[:] = (20, 40, 22)

        num_stripes = 12
        stripe_w = pw / num_stripes
        for i in range(num_stripes):
            x1 = self.pad + int(round(i * stripe_w))
            x2 = self.pad + int(round((i + 1) * stripe_w))
            color = (40, 118, 48) if i % 2 == 0 else (33, 102, 41)
            cv2.rectangle(canvas, (x1, self.pad), (x2, self.pad + ph), color, -1)

        line_col = (240, 246, 240)
        t = max(2, self.scale // 4)

        cv2.rectangle(canvas, self._m2px(0, 0), self._m2px(PITCH_L, PITCH_W), line_col, t, cv2.LINE_AA)
        cv2.line(canvas, self._m2px(52.5, 0), self._m2px(52.5, 68.0), line_col, t, cv2.LINE_AA)

        cv2.circle(canvas, self._m2px(52.5, 34.0), int(round(9.15 * self.scale)), line_col, t, cv2.LINE_AA)
        cv2.circle(canvas, self._m2px(52.5, 34.0), max(3, self.scale // 2), line_col, -1, cv2.LINE_AA)

        cv2.rectangle(canvas, self._m2px(0, 13.84), self._m2px(16.5, 54.16), line_col, t, cv2.LINE_AA)
        cv2.rectangle(canvas, self._m2px(0, 24.84), self._m2px(5.5, 43.16), line_col, t, cv2.LINE_AA)
        cv2.circle(canvas, self._m2px(11.0, 34.0), max(3, self.scale // 2), line_col, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, self._m2px(11.0, 34.0), (int(round(9.15 * self.scale)), int(round(9.15 * self.scale))),
                    0, -53, 53, line_col, t, cv2.LINE_AA)

        cv2.rectangle(canvas, self._m2px(88.5, 13.84), self._m2px(105.0, 54.16), line_col, t, cv2.LINE_AA)
        cv2.rectangle(canvas, self._m2px(99.5, 24.84), self._m2px(105.0, 43.16), line_col, t, cv2.LINE_AA)
        cv2.circle(canvas, self._m2px(94.0, 34.0), max(3, self.scale // 2), line_col, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, self._m2px(94.0, 34.0), (int(round(9.15 * self.scale)), int(round(9.15 * self.scale))),
                    0, 127, 233, line_col, t, cv2.LINE_AA)

        r_c = max(4, int(round(1.0 * self.scale)))
        cv2.ellipse(canvas, self._m2px(0, 0), (r_c, r_c), 0, 0, 90, line_col, t, cv2.LINE_AA)
        cv2.ellipse(canvas, self._m2px(0, PITCH_W), (r_c, r_c), 0, 270, 360, line_col, t, cv2.LINE_AA)
        cv2.ellipse(canvas, self._m2px(PITCH_L, 0), (r_c, r_c), 0, 90, 180, line_col, t, cv2.LINE_AA)
        cv2.ellipse(canvas, self._m2px(PITCH_L, PITCH_W), (r_c, r_c), 0, 180, 270, line_col, t, cv2.LINE_AA)

        g_depth = max(6, int(round(2.4 * self.scale)))
        cv2.rectangle(canvas, (self.pad - g_depth, self._m2px(0, 30.34)[1]),
                      (self.pad, self._m2px(0, 37.66)[1]), (220, 220, 220), max(1, t - 1))
        cv2.rectangle(canvas, (self.pad + pw, self._m2px(105, 30.34)[1]),
                      (self.pad + pw + g_depth, self._m2px(105, 37.66)[1]), (220, 220, 220), max(1, t - 1))

        return canvas

    def render_2d_pitch(
        self,
        frame_w=None,
        frame_h=None,
        kpts_px=None,
        players_xy=None,
        ball_xy=None,
        show_fov=True,
        show_ref_ids=False,
    ):
        mm = self.template.copy()

        if show_ref_ids:
            for k_id, (xm, ym) in PITCH_COORDS_M.items():
                px, py = self._m2px(xm, ym)
                cv2.circle(mm, (px, py), 4, (0, 200, 255), -1, cv2.LINE_AA)
                cv2.putText(mm, str(k_id), (px + 5, py - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        if self.H_p2i is None or self.H_i2p is None:
            return mm

        # 1. Camera FOV (pre-computed grid + ROI-only alpha blend)
        if show_fov and frame_w and frame_h:
            proj = self._pitch_grid_h @ self.H_p2i.T
            valid_z = proj[:, 2] > 1e-4
            if np.any(valid_z):
                inv_z = 1.0 / proj[valid_z, 2]
                u = proj[valid_z, 0] * inv_z
                v = proj[valid_z, 1] * inv_z
                in_frame = (u >= 0) & (u <= frame_w) & (v >= 0) & (v <= frame_h)
                pts_mm = self._grid_mm_px[valid_z][in_frame]
                if len(pts_mm) >= 3:
                    hull = cv2.convexHull(pts_mm)
                    x, y, w, h = cv2.boundingRect(hull)
                    roi = mm[y:y + h, x:x + w]
                    fov_roi = roi.copy()
                    cv2.fillPoly(fov_roi, [hull - [x, y]], (255, 225, 60))
                    cv2.addWeighted(fov_roi, 0.20, roi, 0.80, 0, roi)
                    cv2.polylines(mm, [hull], True, (255, 225, 60), 2, cv2.LINE_AA)

        # 2. Projected keypoints
        if kpts_px:
            ids = list(kpts_px.keys())
            img_pts = np.array([[kpts_px[k][0], kpts_px[k][1]] for k in ids], dtype=np.float32)
            pitch_pts = self.image_to_pitch(img_pts)
            for idx, k_id in enumerate(ids):
                xm, ym = float(pitch_pts[idx, 0]), float(pitch_pts[idx, 1])
                if -2.0 <= xm <= PITCH_L + 2.0 and -2.0 <= ym <= PITCH_W + 2.0:
                    px, py = self._m2px(xm, ym)
                    conf = kpts_px[k_id][2]
                    col = (0, 255, 0) if conf >= 0.50 else (0, 215, 255)
                    cv2.circle(mm, (px, py), 5, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(mm, (px, py), 3, col, -1, cv2.LINE_AA)

        # 3. Players
        if players_xy:
            pts = np.array([[p[0], p[1]] for p in players_xy], dtype=np.float32)
            p_m = self.image_to_pitch(pts)
            for idx, (_, _, col) in enumerate(players_xy):
                xm, ym = float(p_m[idx, 0]), float(p_m[idx, 1])
                if -1.0 <= xm <= PITCH_L + 1.0 and -1.0 <= ym <= PITCH_W + 1.0:
                    px, py = self._m2px(xm, ym)
                    cv2.circle(mm, (px, py), 8, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(mm, (px, py), 6, col, -1, cv2.LINE_AA)

        # 4. Ball
        if ball_xy is not None:
            b_m = self.image_to_pitch([ball_xy])
            xm, ym = float(b_m[0, 0]), float(b_m[0, 1])
            if -1.0 <= xm <= PITCH_L + 1.0 and -1.0 <= ym <= PITCH_W + 1.0:
                px, py = self._m2px(xm, ym)
                cv2.circle(mm, (px, py), 6, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(mm, (px, py), 4, (0, 255, 255), -1, cv2.LINE_AA)

        return mm


if __name__ == "__main__":
    pitch = PitchHomography2D(scale=8, pad=30)
    img_2d = pitch.render_2d_pitch(show_fov=False, show_ref_ids=True)
    cv2.imwrite("pitch_2d_preview.jpg", img_2d)
    print(f"[OK] Tswira dyal t-tiran 2D t-sauvegardat f: pitch_2d_preview.jpg ({img_2d.shape[1]}x{img_2d.shape[0]})")