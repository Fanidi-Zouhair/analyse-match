import os
import time
import argparse
import numpy as np
import torch
import openvino as ov

from pitch_pose_ai import PitchPoseNet, NUM_KEYPOINTS, IMG_SIZE


def export_to_onnx(model, onnx_path, img_size=IMG_SIZE):
    """Export PyTorch model to static-shape ONNX (required for Intel NPU)."""
    dummy_input = torch.rand(1, 3, img_size, img_size, dtype=torch.float32)

    print(f"[1/4] Export l ONNX -> '{onnx_path}' (static 1x3x{img_size}x{img_size})...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["coords", "vis_logits"],
    )
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"      [OK] ONNX wajed ({size_mb:.2f} MB)")
    return dummy_input


def convert_to_openvino_fp16(onnx_path, ov_dir, img_size=IMG_SIZE):
    """Convert ONNX model to OpenVINO IR compressed to FP16."""
    os.makedirs(ov_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    xml_path = os.path.join(ov_dir, f"{stem}_fp16.xml")

    print(f"[2/4] Ta7wil l OpenVINO IR FP16 -> '{xml_path}'...")
    ov_model = ov.convert_model(onnx_path, input=[1, 3, img_size, img_size])
    ov.save_model(ov_model, xml_path, compress_to_fp16=True)

    bin_path = xml_path.replace(".xml", ".bin")
    size_mb = os.path.getsize(bin_path) / (1024 * 1024)
    print(f"      [OK] OpenVINO FP16 wajed ({size_mb:.2f} MB)")
    return xml_path


def benchmark_pytorch_cpu(model, dummy_input, warmup=15, runs=100):
    """Measure PyTorch FP32 baseline speed on CPU."""
    with torch.no_grad():
        ref_coords, ref_vis = model(dummy_input)
        for _ in range(warmup):
            _ = model(dummy_input)

        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy_input)
        elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / runs) * 1000.0
    fps = runs / elapsed
    return avg_ms, fps, ref_coords.numpy(), ref_vis.numpy()


def benchmark_openvino_device(core, xml_path, device, dummy_np, ref_coords, ref_vis, img_size=IMG_SIZE, warmup=20, runs=200):
    """Compile and benchmark OpenVINO IR on a specific device (LATENCY mode)."""
    config = {"PERFORMANCE_HINT": "LATENCY"}
    t_comp = time.perf_counter()
    compiled = core.compile_model(xml_path, device, config)
    comp_time = time.perf_counter() - t_comp

    infer_req = compiled.create_infer_request()
    out_coords = compiled.output(0)
    out_vis = compiled.output(1)

    # Warmup + accuracy check
    infer_req.infer({0: dummy_np})
    ov_coords = infer_req.get_tensor(out_coords).data.copy()
    ov_vis = infer_req.get_tensor(out_vis).data.copy()

    # Accuracy diff vs PyTorch FP32
    px_diff = np.sqrt(((ov_coords - ref_coords) * img_size) ** 2).sum(axis=-1)
    max_px_err = float(np.max(px_diff))
    mean_px_err = float(np.mean(px_diff))

    ref_prob = 1.0 / (1.0 + np.exp(-ref_vis))
    ov_prob = 1.0 / (1.0 + np.exp(-ov_vis))
    max_vis_diff = float(np.max(np.abs(ov_prob - ref_prob)))

    for _ in range(warmup):
        infer_req.infer({0: dummy_np})

    t0 = time.perf_counter()
    for _ in range(runs):
        infer_req.infer({0: dummy_np})
    elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / runs) * 1000.0
    fps = runs / elapsed
    return {
        "device": device,
        "compile_s": comp_time,
        "avg_ms": avg_ms,
        "fps": fps,
        "mean_px_diff": mean_px_err,
        "max_px_diff": max_px_err,
        "max_vis_diff": max_vis_diff,
    }


def benchmark_openvino_async(core, xml_path, device, dummy_np, warmup=30, runs=300):
    """Benchmark OpenVINO AsyncInferQueue (THROUGHPUT mode) for video pipelines."""
    compiled = core.compile_model(xml_path, device, {"PERFORMANCE_HINT": "THROUGHPUT"})
    try:
        n_ireq = compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    except Exception:
        n_ireq = 4

    queue = ov.AsyncInferQueue(compiled, n_ireq)

    for _ in range(warmup):
        queue.start_async({0: dummy_np})
    queue.wait_all()

    t0 = time.perf_counter()
    for _ in range(runs):
        queue.start_async({0: dummy_np})
    queue.wait_all()
    elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / runs) * 1000.0
    fps = runs / elapsed
    return avg_ms, fps, n_ireq


def main():
    parser = argparse.ArgumentParser(description="Export PitchPoseNet to ONNX + OpenVINO FP16 and benchmark Ultra 5 135U.")
    parser.add_argument("--weights", type=str, default="best_pose_brut.pth", help="Path to .pth weights")
    parser.add_argument("--onnx", type=str, default="best_pose_brut.onnx", help="Output ONNX path")
    parser.add_argument("--ov-dir", type=str, default="ov_model", help="Output folder for OpenVINO IR")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="Input image size (default: 256)")
    parser.add_argument("--runs", type=int, default=200, help="Benchmark iterations per OpenVINO device")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Malqitch l-weights: '{args.weights}'")

    # Load PyTorch model on CPU
    model = PitchPoseNet(num_keypoints=NUM_KEYPOINTS, pretrained=False)
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 1. Export ONNX
    dummy_torch = export_to_onnx(model, args.onnx, img_size=args.img_size)
    dummy_np = dummy_torch.numpy()

    # 2. Convert to OpenVINO FP16
    xml_path = convert_to_openvino_fp16(args.onnx, args.ov_dir, img_size=args.img_size)

    # 3. Benchmark PyTorch CPU baseline
    print("\n[3/4] Benchmark PyTorch FP32 (CPU Baseline)...")
    pt_ms, pt_fps, ref_coords, ref_vis = benchmark_pytorch_cpu(model, dummy_torch, warmup=10, runs=60)
    print(f"      PyTorch CPU: {pt_ms:.2f} ms/frame | {pt_fps:.1f} FPS")

    # 4. Benchmark OpenVINO devices
    core = ov.Core()
    avail = core.available_devices
    print(f"\n[4/4] OpenVINO Available Devices: {avail}")

    target_devices = [d for d in ["CPU", "GPU", "NPU"] if d in avail]
    if "GPU" in avail or "NPU" in avail:
        target_devices.append("AUTO:GPU,NPU,CPU")

    results = []
    for dev in target_devices:
        print(f"  -> Testing OpenVINO FP16 3la [{dev}] (LATENCY)...")
        try:
            res = benchmark_openvino_device(
                core, xml_path, dev, dummy_np, ref_coords, ref_vis,
                img_size=args.img_size, warmup=20, runs=args.runs
            )
            results.append(res)
        except Exception as e:
            print(f"     [SKIP] [{dev}] fchel: {e}")

    # Print Scoreboard
    print("\n" + "=" * 96)
    print("  BENCHMARK SCOREBOARD (Intel Core Ultra 5 135U — 256x256, 39 Keypoints)")
    print("=" * 96)
    print(f"  {'Backend / Device':22s} | {'Latency':>10s} | {'FPS':>8s} | {'Speedup':>8s} | {'Mean PxDiff':>11s} | {'Max PxDiff':>10s}")
    print("-" * 96)
    print(f"  {'PyTorch FP32 (CPU)':22s} | {pt_ms:7.2f} ms | {pt_fps:8.1f} | {'1.00x':>8s} | {'0.000 px':>11s} | {'0.000 px':>10s}")

    best_single = None
    for r in results:
        speedup = r["fps"] / pt_fps if pt_fps > 0 else 0.0
        print(
            f"  {'OV FP16 (' + r['device'] + ')':22s} | "
            f"{r['avg_ms']:7.2f} ms | {r['fps']:8.1f} | {speedup:7.2f}x | "
            f"{r['mean_px_diff']:8.4f} px | {r['max_px_diff']:7.4f} px"
        )
        if r["device"] in ("CPU", "GPU", "NPU"):
            if best_single is None or r["fps"] > best_single["fps"]:
                best_single = r

    # Test AsyncInferQueue on fastest hardware device
    if best_single is not None:
        dev = best_single["device"]
        print("-" * 96)
        try:
            a_ms, a_fps, n_ireq = benchmark_openvino_async(core, xml_path, dev, dummy_np, warmup=30, runs=300)
            a_speedup = a_fps / pt_fps if pt_fps > 0 else 0.0
            print(
                f"  {'OV Async (' + dev + f' x{n_ireq})':22s} | "
                f"{a_ms:7.2f} ms | {a_fps:8.1f} | {a_speedup:7.2f}x | {'(same IR)':>11s} | {'(same IR)':>10s}"
            )
        except Exception as e:
            print(f"  [INFO] Async test 3la {dev} ma-kammelch: {e}")

    print("=" * 96)
    print(f"\n[SAFI] Files l-wajdin:")
    print(f"  - ONNX        : {os.path.abspath(args.onnx)}")
    print(f"  - OpenVINO IR : {os.path.abspath(xml_path)}")


if __name__ == "__main__":
    main()