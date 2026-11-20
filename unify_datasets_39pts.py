import os
import glob
import argparse

# ==============================================================================
# CONFIGURATION: TAW7ID 32-PTS + 33-PTS -> 39-PTS STANDARD (pitch_pose_ai.py)
# ==============================================================================
TARGET_NUM_KPTS = 39

# 1) Dataset 33 points (image_695948.png / dataset2) -> 39-pt unified IDs (1-indexed)
MAP_33_TO_39 = {
    # Left side
    1: 17,   # Corner top-left
    2: 16,   # 18m top on goal line
    3: 31,   # 18m top inner
    4: 35,   # 18m bottom inner
    5: 11,   # 18m bottom on goal line
    6: 12,   # 6m bottom on goal line
    7: 37,   # 6m bottom inner
    8: 36,   # 6m top inner
    9: 15,   # 6m top on goal line
    10: 10,  # Corner bottom-left
    27: 32,  # Left arc top (on 18m line)
    28: 34,  # Left arc bottom (on 18m line)
    29: 33,  # Left arc apex
    # Middle
    21: 18,  # Halfway line top
    22: 19,  # Center circle top
    23: 21,  # Center circle left
    24: 22,  # Center circle bottom
    25: 20,  # Center circle right
    26: 9,   # Halfway line bottom
    33: 39,  # Center spot
    # Right side
    11: 1,   # Corner top-right
    12: 2,   # 18m top on goal line
    13: 23,  # 18m top inner
    14: 27,  # 18m bottom inner
    15: 7,   # 18m bottom on goal line
    16: 6,   # 6m bottom on goal line
    17: 29,  # 6m bottom inner
    18: 28,  # 6m top inner
    19: 3,   # 6m top on goal line
    20: 8,   # Corner bottom-right
    30: 24,  # Right arc top (on 18m line)
    31: 26,  # Right arc bottom (on 18m line)
    32: 25,  # Right arc apex
}

# 2) Dataset 32 points (image_69590c.png / Roboflow 32-pt) -> 39-pt unified IDs (1-indexed)
# Note: points 9, 14, 19, 24 are not shown on the skeleton diagram -> safely left unmapped (vis=0)
MAP_32_TO_39 = {
    # Left side
    1: 17,   # Corner top-left
    2: 16,   # 18m top on goal line
    3: 15,   # 6m top on goal line
    4: 12,   # 6m bottom on goal line
    5: 11,   # 18m bottom on goal line
    6: 10,   # Corner bottom-left
    7: 36,   # 6m top inner
    8: 37,   # 6m bottom inner
    10: 31,  # 18m top inner
    11: 32,  # Left arc top (on 18m line)
    12: 34,  # Left arc bottom (on 18m line)
    13: 35,  # 18m bottom inner
    # Middle
    15: 18,  # Halfway line top
    16: 19,  # Center circle top
    17: 22,  # Center circle bottom
    18: 9,   # Halfway line bottom
    # Right side
    20: 23,  # 18m top inner
    21: 24,  # Right arc top (on 18m line)
    22: 26,  # Right arc bottom (on 18m line)
    23: 27,  # 18m bottom inner
    25: 28,  # 6m top inner
    26: 29,  # 6m bottom inner
    27: 1,   # Corner top-right
    28: 2,   # 18m top on goal line
    29: 3,   # 6m top on goal line
    30: 6,   # 6m bottom on goal line
    31: 7,   # 18m bottom on goal line
    32: 8,   # Corner bottom-right
}


def detect_schema(num_raw_vals):
    """Auto-detect (source_kpts, step) from YOLO-Pose keypoint value count."""
    if num_raw_vals in (TARGET_NUM_KPTS * 3, TARGET_NUM_KPTS * 2):
        return TARGET_NUM_KPTS, (3 if num_raw_vals == TARGET_NUM_KPTS * 3 else 2)
    if num_raw_vals == 33 * 3:
        return 33, 3
    if num_raw_vals == 33 * 2:
        return 33, 2
    if num_raw_vals == 32 * 3:
        return 32, 3
    if num_raw_vals == 32 * 2:
        return 32, 2
    return None, None


def remap_label_file(txt_path):
    """Convert a 32-pt or 33-pt YOLO-Pose .txt file in-place to unified 39-pt format."""
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        return "empty"

    new_lines = []
    detected_kind = None

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        header = parts[:5]  # class_id, cx, cy, bw, bh
        raw_kpts = parts[5:]

        src_kpts, step = detect_schema(len(raw_kpts))
        if src_kpts == TARGET_NUM_KPTS:
            return "already_39"
        if src_kpts is None:
            print(f"[SKIP] Format ghrib f '{txt_path}' ({len(raw_kpts)} qiyam)")
            return "skipped"

        mapping = MAP_33_TO_39 if src_kpts == 33 else MAP_32_TO_39
        detected_kind = f"converted_{src_kpts}"

        new_kpts = [("0", "0", "0") for _ in range(TARGET_NUM_KPTS)]

        for old_id, new_id in mapping.items():
            old_idx = old_id - 1
            new_idx = new_id - 1

            x_str = raw_kpts[old_idx * step]
            y_str = raw_kpts[old_idx * step + 1]
            if step == 3:
                v_str = raw_kpts[old_idx * step + 2]
            else:
                v_str = "2" if (float(x_str) > 0.0 or float(y_str) > 0.0) else "0"

            new_kpts[new_idx] = (x_str, y_str, v_str)

        flat_kpts = [val for triplet in new_kpts for val in triplet]
        new_lines.append(" ".join(header + flat_kpts))

    if not new_lines:
        return "empty"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    return detected_kind


def process_dataset(dataset_dir):
    print(f"\n--- Processing: '{dataset_dir}' ---")
    stats = {"converted_33": 0, "converted_32": 0, "already_39": 0, "empty": 0, "skipped": 0}

    for split in ("train", "valid", "test"):
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.exists(lbl_dir):
            print(f"  [INFO] '{lbl_dir}' ma-kaynch (skipped).")
            continue

        txt_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
        split_conv = 0
        for txt_path in txt_files:
            status = remap_label_file(txt_path)
            stats[status] = stats.get(status, 0) + 1
            if status.startswith("converted_"):
                split_conv += 1

        print(f"  [OK] {split:5s}: t-7awlo {split_conv}/{len(txt_files)} fichiers")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Unify 32-pt and 33-pt Roboflow YOLO-Pose datasets into 39-pt format.")
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset folders to remap (e.g., dataset2 dataset3)",
    )
    args = parser.parse_args()

    totals = {"converted_33": 0, "converted_32": 0, "already_39": 0, "empty": 0, "skipped": 0}
    for d_dir in args.datasets:
        if not os.path.exists(d_dir):
            print(f"[ERREUR] Dossier '{d_dir}' ma-kaynch!")
            continue
        st = process_dataset(d_dir)
        for k, v in st.items():
            totals[k] = totals.get(k, 0) + v

    print("\n" + "=" * 60)
    print("[SAFI] T-taw7id l 39 points sala b-naja7!")
    print(f"  - T-7awlo (33 -> 39 pts) : {totals['converted_33']}")
    print(f"  - T-7awlo (32 -> 39 pts) : {totals['converted_32']}")
    print(f"  - Deja 39 pts            : {totals['already_39']}")
    print(f"  - Khawyin                : {totals['empty']}")
    print(f"  - T-jahlo                : {totals['skipped']}")
    print("=" * 60)


if __name__ == "__main__":
    main()