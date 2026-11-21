import os
import time
import argparse
import threading
import queue
import cv2
import numpy as np
import openvino as ov

from pitch_homography_2d import PitchHomography2D

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OV_MODEL_PATH = "ov_model/best_pose_brut_fp16.xml"
OV_DEVICE = "AUTO:GPU,NPU,CPU"   # Baddalha l "GPU" wla "NPU" wla "CPU" ila bghiti device mo7addad
VIDEO_PATH = "Video Project 2.mp4"
OUTPUT_VIDEO_PATH = "Video_Project_2_ov_fast_annotated.mp4"

IMG_SIZE = 256
CONF_THRESHOLD = 0.35       # Keypoints with prob >= 0.35 are considered
HIGH_CONF_THRESHOLD = 0.50  # Keypoints with prob >= 0.50 are highlighted green

# SPEED & STABILITY SETTINGS
AI_EVERY_N_FRAMES = 3       # 1 = AI f kol frame (Async) | 2-3 = AI + Optical Flow (asra3 x2-x3)
FLOW_WIDTH = 640            # Width dyal grayscale l Optical Flow (< 1ms/frame)
FB_MAX_ERR_PX = 2.5         # Forward-Backward Optical Flow error limit (kay-7ayyad n-noqat li harbo)
MIN_TRACKED_PTS = 4         # Ila qall 3adad n-noqat 3la 4 f Flow -> kay-khaddam AI f l-blassa
EMA_ALPHA = 0.70            # Smoothing khfif bach i-7ayyad jitter (1.0 = bla smoothing)

# RADAR 2D (STYLE EA FC / FC27)
RADAR_WIDTH_RATIO = 0.30    # 30% mn 3ard l-video (bhal radar d EA FC f l-wast taht)
RADAR_ALPHA = 0.84          # Chafafiya dyal r-radar foq l-video

# SKELETON CONNECTIONS (1-indexed based on 39 keypoints definition)
SKELETON_EDGES_1IDX = [
    # Sidelines
    (17, 18), (18, 1),
    (10, 9), (9, 8),
    # Halfway line
    (18, 9),
    # Center circle
    (19, 21), (21, 22), (22, 20), (20, 19),
    # Left penalty box
    (16, 31), (31, 35), (35, 11),
    # Left goal box
    (15, 36), (36, 37), (37, 12),
    # Left arc
    (32, 33), (33, 34),
    # Left goal line
    (17, 16), (16, 15), (15, 12), (12, 11), (11, 10),
    # Right penalty box
    (2, 23), (23, 27), (27, 7),
    # Right goal box
    (3, 28), (28, 29), (29, 6),
    # Right arc
    (24, 25), (25, 26),
    # Right goal line
    (1, 2), (2, 3), (3, 6), (6, 7), (7, 8),
]


# ==============================================================================
# THREADED VIDEO READER & WRITER (I/O f threads bo7dhom bach ma y-bloqiwch GPU)
# ==============================================================================
class ThreadedVideoReader:
    def __init__(self, cap, queue_size=6):
        self.cap = cap
        self.q = queue.Queue(maxsize=queue_size)
        self._stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self._stop:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put(None)
                break
            self.q.put(frame)

    def read(self):
        frame = self.q.get()
        if frame is None:
            return False, None
        return True, frame

    def release(self):
        self._stop = True
        self.t.join(timeout=2.0)
        self.cap.release()


class ThreadedVideoWriter:
    def __init__(self, writer, queue_size=24):
        self.writer = writer
        self.q = queue.Queue(maxsize=queue_size)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            frame, sample_path = item
            self.writer.write(frame)
            if sample_path is not None:
                cv2.imwrite(sample_path, frame)

    def write(self, frame, sample_path=None):
        self.q.put((frame, sample_path))

    def release(self):
        self.q.put(None)
        self.t.join()
        self.writer.release()


# Pre-allocated constant overlays for zero-copy ROI blending
_HUD_DARK_CACHE = None
_RADAR_BG_CACHE = None


def preprocess_frame(frame_bgr, img_size=IMG_SIZE):
    """Resize FIRST (1080p -> 256x256), THEN BGR->RGB (32x asra3)."""
    img_resized = cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = (img_rgb.astype(np.float32) * (1.0 / 255.0)).transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(tensor)


def decode_predictions(coords, vis_logits, width, height, prev_kpts=None):
    """Convert raw OpenVINO outputs to {kpt_id: (px, py, conf)} with optional EMA smoothing."""
    vis_probs = 1.0 / (1.0 + np.exp(-vis_logits[0]))  # (39,)
    coords_0 = coords[0]                              # (39, 2)

    kpts_px = {}
    for i in range(coords_0.shape[0]):
        p = float(vis_probs[i])
        if p >= CONF_THRESHOLD:
            px = float(np.clip(coords_0[i, 0] * width, 0, width - 1))
            py = float(np.clip(coords_0[i, 1] * height, 0, height - 1))
            kpt_id = i + 1
            if prev_kpts is not None and kpt_id in prev_kpts and EMA_ALPHA < 1.0:
                old_x, old_y, _ = prev_kpts[kpt_id]
                px = EMA_ALPHA * px + (1.0 - EMA_ALPHA) * old_x
                py = EMA_ALPHA * py + (1.0 - EMA_ALPHA) * old_y
            kpts_px[kpt_id] = (px, py, p)
    return kpts_px


def to_flow_gray(frame_bgr, flow_width=FLOW_WIDTH):
    """Downscale frame to fast grayscale for Lucas-Kanade Optical Flow."""
    h, w = frame_bgr.shape[:2]
    scale = flow_width / float(w)
    new_h = max(1, int(round(h * scale)))
    small = cv2.resize(frame_bgr, (flow_width, new_h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray, scale


def track_keypoints_flow(prev_gray, curr_gray, prev_kpts, scale, width, height):
    """Track visible keypoints with Forward-Backward Lucas-Kanade Optical Flow."""
    if not prev_kpts:
        return {}

    ids = list(prev_kpts.keys())
    pts_prev = np.array(
        [[prev_kpts[k][0] * scale, prev_kpts[k][1] * scale] for k in ids],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    pts_next, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None, **lk_params)
    if pts_next is None or st_fwd is None:
        return {}

    pts_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, pts_next, None, **lk_params)
    if pts_back is None or st_bwd is None:
        return {}

    fb_err = np.linalg.norm(pts_prev.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1)
    inv_scale = 1.0 / scale

    tracked = {}
    for idx, kpt_id in enumerate(ids):
        if st_fwd[idx, 0] == 1 and st_bwd[idx, 0] == 1 and fb_err[idx] <= FB_MAX_ERR_PX:
            nx = float(pts_next[idx, 0, 0] * inv_scale)
            ny = float(pts_next[idx, 0, 1] * inv_scale)
            if 0.0 <= nx < width and 0.0 <= ny < height:
                conf = max(CONF_THRESHOLD, prev_kpts[kpt_id][2] * 0.98)
                tracked[kpt_id] = (nx, ny, conf)

    return tracked


def paste_fc27_radar(canvas, radar_img, width_ratio=RADAR_WIDTH_RATIO, alpha=RADAR_ALPHA):
    """Kat-7at t-tiran 2D f l-wast taht b cadre style EA FC / FC27 (in-place + cached bg)."""
    global _RADAR_BG_CACHE
    fh, fw = canvas.shape[:2]
    rh, rw = radar_img.shape[:2]

    target_w = int(round(fw * width_ratio))
    target_h = int(round(rh * (target_w / float(rw))))
    if target_w == rw and target_h == rh:
        radar_resized = radar_img
    else:
        radar_resized = cv2.resize(radar_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    x1 = (fw - target_w) // 2
    y1 = fh - target_h - 18
    x2 = x1 + target_w
    y2 = y1 + target_h

    if y1 >= 4 and x1 >= 4 and y2 <= fh - 4 and x2 <= fw - 4:
        pad = 4
        bg = canvas[y1 - pad:y2 + pad, x1 - pad:x2 + pad]
        if _RADAR_BG_CACHE is None or _RADAR_BG_CACHE.shape != bg.shape:
            _RADAR_BG_CACHE = np.full(bg.shape, (15, 18, 15), dtype=np.uint8)
        cv2.addWeighted(_RADAR_BG_CACHE, 0.75, bg, 0.25, 0, bg)

        roi = canvas[y1:y2, x1:x2]
        cv2.addWeighted(radar_resized, alpha, roi, 1.0 - alpha, 0, roi)

        cv2.rectangle(canvas, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), (210, 230, 210), 1, cv2.LINE_AA)

    return canvas


def draw_annotations(frame, kpts_px, radar_img, h_info, model_name, device_name, mode_tag, frame_idx, total_frames, fps, live_fps):
    """Draw pitch skeleton, keypoints, FC27 2D radar, and HUD directly in-place (zero full-frame copy)."""
    global _HUD_DARK_CACHE
    canvas = frame

    # Draw pitch skeleton lines
    for (p1_id, p2_id) in SKELETON_EDGES_1IDX:
        if p1_id in kpts_px and p2_id in kpts_px:
            pt1 = (int(round(kpts_px[p1_id][0])), int(round(kpts_px[p1_id][1])))
            pt2 = (int(round(kpts_px[p2_id][0])), int(round(kpts_px[p2_id][1])))
            cv2.line(canvas, pt1, pt2, (255, 200, 0), 2, cv2.LINE_AA)

    # Draw detected keypoints
    for kpt_id, (px_f, py_f, conf) in kpts_px.items():
        px, py = int(round(px_f)), int(round(py_f))
        if conf >= HIGH_CONF_THRESHOLD:
            color_inner = (0, 255, 0)
            color_outer = (0, 180, 0)
        else:
            color_inner = (0, 215, 255)
            color_outer = (0, 140, 255)

        cv2.circle(canvas, (px, py), 7, color_outer, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 4, color_inner, -1, cv2.LINE_AA)

        label = f"P{kpt_id}:{int(conf * 100)}%"
        cv2.putText(canvas, label, (px + 8, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (px + 8, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Paste FC27 2D radar minimap (bottom-center)
    if radar_img is not None:
        canvas = paste_fc27_radar(canvas, radar_img)

    # Status HUD banner (ROI-only blend: 37x asra3 mn full-frame copy!)
    hud_roi = canvas[15:118, 15:560]
    if _HUD_DARK_CACHE is None or _HUD_DARK_CACHE.shape != hud_roi.shape:
        _HUD_DARK_CACHE = np.full(hud_roi.shape, (20, 20, 20), dtype=np.uint8)
    cv2.addWeighted(_HUD_DARK_CACHE, 0.70, hud_roi, 0.30, 0, hud_roi)
    cv2.rectangle(canvas, (15, 15), (560, 118), (255, 200, 0), 1)

    cv2.putText(
        canvas, f"Pitch Pose OpenVINO FP16 ({device_name}) [{mode_tag}]",
        (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas, f"Frame: {frame_idx}/{total_frames} ({frame_idx/fps:.2f}s) | Detected: {len(kpts_px)}/39 | {live_fps:.1f} FPS",
        (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
    )
    h_st = h_info["status"]
    h_txt = f"2D Radar: {h_st} ({h_info['inliers']}/{h_info['total']} | {h_info['reproj_px']:.1f}px)" if h_st == "OK" else f"2D Radar: {h_st}"
    cv2.putText(
        canvas, f"Model: {model_name} | {h_txt}",
        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 120) if h_st == "OK" else (200, 200, 200), 1, cv2.LINE_AA
    )
    return canvas


def process_video(model_path, device, video_path, output_path, ai_every_n):
    print("=" * 75)
    print(f"  OPENVINO FP16 + RADAR 2D (FC27) [THREADED I/O]: {video_path}")
    print("=" * 75)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"OpenVINO model '{model_path}' ma-kaynch! Khaddam 'export_and_benchmark_ov.py' howa l-lawal."
        )
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"L-video '{video_path}' ma-lqaytoush!")

    # 1. Load & Compile OpenVINO FP16 Model
    core = ov.Core()
    print(f"[1/4] Available OpenVINO Devices: {core.available_devices}")
    print(f"[1/4] Compiling '{model_path}' 3la [{device}]...")
    perf_hint = "THROUGHPUT" if ai_every_n == 1 else "LATENCY"
    compiled = core.compile_model(model_path, device, {"PERFORMANCE_HINT": perf_hint})
    out_coords = compiled.output(0)
    out_vis = compiled.output(1)
    model_name = os.path.basename(model_path)
    print(f"[1/4] Model wajed! (Mode: {'Async Full-AI' if ai_every_n == 1 else f'AI kol {ai_every_n} frames + Optical Flow'})")

    # Module dyal t-tiran 2D (mn pitch_homography_2d.py)
    pitch_2d = PitchHomography2D(scale=5, pad=16)

    # 2. Threaded Video Capture & Writer
    raw_cap = cv2.VideoCapture(video_path)
    fps = raw_cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(raw_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[2/4] Video info: {width}x{height} @ {fps:.1f} fps | Majmou3 l-fremat: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap = ThreadedVideoReader(raw_cap, queue_size=6)
    out = ThreadedVideoWriter(raw_out, queue_size=24)

    frame_idx = 0
    ai_frames = 0
    flow_frames = 0
    sample_saved = 0
    detected_counts = []
    prev_kpts = None
    prev_gray = None
    start_time = time.time()

    print("[3/4] Bidayat t-tahlil...")

    if ai_every_n == 1:
        # --- PIPELINED ASYNC MODE (2 ping-pong InferRequests + Threaded I/O) ---
        curr_req = compiled.create_infer_request()
        next_req = compiled.create_infer_request()

        ret, curr_frame = cap.read()
        if ret:
            curr_req.start_async({0: preprocess_frame(curr_frame)})

            while True:
                ret_next, next_frame = cap.read()
                if ret_next:
                    next_req.start_async({0: preprocess_frame(next_frame)})

                curr_req.wait()
                frame_idx += 1
                ai_frames += 1

                coords = curr_req.get_tensor(out_coords).data
                vis_logits = curr_req.get_tensor(out_vis).data
                kpts_px = decode_predictions(coords, vis_logits, width, height, prev_kpts=prev_kpts)
                prev_kpts = kpts_px
                detected_counts.append(len(kpts_px))

                # 7sab Homography + rsam t-tiran 2D (FC27)
                _, _, h_info = pitch_2d.update(kpts_px, width, height)
                radar_img = pitch_2d.render_2d_pitch(frame_w=width, frame_h=height, kpts_px=kpts_px, show_fov=True)

                elapsed = time.time() - start_time
                live_fps = frame_idx / elapsed if elapsed > 0 else 0.0

                canvas = draw_annotations(
                    curr_frame, kpts_px, radar_img, h_info, model_name, device, "AI-ASYNC",
                    frame_idx, total_frames, fps, live_fps
                )

                sample_path = None
                if frame_idx in (15, 75, 150) or (sample_saved < 3 and len(kpts_px) >= 6 and frame_idx % 40 == 0):
                    sample_path = f"sample_ov_frame_{frame_idx}.jpg"
                    sample_saved += 1

                out.write(canvas, sample_path=sample_path)

                if frame_idx % 30 == 0 or frame_idx == total_frames:
                    print(f"  [Progress] Frame {frame_idx:3d}/{total_frames} | Speed: {live_fps:.1f} fps | Detected: {len(kpts_px):2d} pts | 2D: {h_info['status']}")

                if not ret_next:
                    break

                curr_frame = next_frame
                curr_req, next_req = next_req, curr_req
    else:
        # --- HYBRID MODE: OpenVINO Keyframes + Forward-Backward Optical Flow ---
        infer_req = compiled.create_infer_request()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            curr_gray, scale = to_flow_gray(frame, FLOW_WIDTH)
            need_ai = (frame_idx == 1) or ((frame_idx - 1) % ai_every_n == 0) or (prev_kpts is None)

            mode_tag = "AI"
            if not need_ai and prev_gray is not None:
                tracked = track_keypoints_flow(prev_gray, curr_gray, prev_kpts, scale, width, height)
                if len(tracked) >= MIN_TRACKED_PTS:
                    kpts_px = tracked
                    mode_tag = "FLOW"
                    flow_frames += 1
                else:
                    need_ai = True

            if need_ai:
                tensor = preprocess_frame(frame)
                infer_req.infer({0: tensor})
                coords = infer_req.get_tensor(out_coords).data
                vis_logits = infer_req.get_tensor(out_vis).data
                kpts_px = decode_predictions(coords, vis_logits, width, height, prev_kpts=prev_kpts)
                ai_frames += 1

            prev_kpts = kpts_px
            prev_gray = curr_gray
            detected_counts.append(len(kpts_px))

            # 7sab Homography + rsam t-tiran 2D (FC27)
            _, _, h_info = pitch_2d.update(kpts_px, width, height)
            radar_img = pitch_2d.render_2d_pitch(frame_w=width, frame_h=height, kpts_px=kpts_px, show_fov=True)

            elapsed = time.time() - start_time
            live_fps = frame_idx / elapsed if elapsed > 0 else 0.0

            canvas = draw_annotations(
                frame, kpts_px, radar_img, h_info, model_name, device, mode_tag,
                frame_idx, total_frames, fps, live_fps
            )

            sample_path = None
            if frame_idx in (15, 75, 150) or (sample_saved < 3 and len(kpts_px) >= 6 and frame_idx % 40 == 0):
                sample_path = f"sample_ov_frame_{frame_idx}.jpg"
                sample_saved += 1

            out.write(canvas, sample_path=sample_path)

            if frame_idx % 30 == 0 or frame_idx == total_frames:
                print(f"  [Progress] Frame {frame_idx:3d}/{total_frames} | Speed: {live_fps:.1f} fps | Detected: {len(kpts_px):2d} pts ({mode_tag}) | 2D: {h_info['status']}")

    cap.release()
    out.release()

    total_time = time.time() - start_time
    avg_detected = float(np.mean(detected_counts)) if detected_counts else 0.0
    final_fps = frame_idx / total_time if total_time > 0 else 0.0

    print("=" * 75)
    print("[4/4] T-TEST KAMMALL B-NAJA7!")
    print(f"  - Video m-annoté : '{output_path}'")
    print(f"  - Sor3a (End-to-End m3a Video Read/Write): {final_fps:.1f} FPS ({total_time:.2f} s)")
    print(f"  - Frames AI vs Flow: {ai_frames} AI | {flow_frames} Flow")
    print(f"  - Motawassit n-noqat: {avg_detected:.1f} / 39 noqta")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Fast Football Pitch Pose Video Inference (OpenVINO FP16 + FC27 2D Radar)")
    parser.add_argument("--model", type=str, default=OV_MODEL_PATH, help="Path to OpenVINO .xml model")
    parser.add_argument("--device", type=str, default=OV_DEVICE, help="OpenVINO device (AUTO:GPU,NPU,CPU | GPU | NPU | CPU)")
    parser.add_argument("--video", type=str, default=VIDEO_PATH, help="Input video path")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_PATH, help="Output annotated video path")
    parser.add_argument("--ai-every", type=int, default=AI_EVERY_N_FRAMES, help="Run AI every N frames (1=Async Full AI, 2-3=AI+Flow)")
    args = parser.parse_args()

    process_video(args.model, args.device, args.video, args.output, args.ai_every)


if __name__ == "__main__":
    main()