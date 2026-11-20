import os
import glob

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASET2_DIR = "dataset2"   # Dossier dyal dataset2
TARGET_NUM_KPTS = 39        # 3adad n-noqat l-mowa7ad (bhal dataset1 + point 39)

# JADWAL T-TA7WIL (1-indexed: Raqm f dataset2 -> Raqm s-s7i7 f dataset1 / 39pts)
OLD_TO_NEW_MAP = {
    # --- L-Jiha l-Lisra (Orange) ---
    1: 17,   # Corner foq lisr
    2: 16,   # 18m foq 3la l-khat
    3: 31,   # 18m foq dakhal
    4: 35,   # 18m taht dakhal
    5: 11,   # 18m taht 3la l-khat
    6: 12,   # 6m taht 3la l-khat
    7: 37,   # 6m taht dakhal
    8: 36,   # 6m foq dakhal
    9: 15,   # 6m foq 3la l-khat
    10: 10,  # Corner taht lisr
    27: 32,  # Qaws 18m foq lisr
    28: 34,  # Qaws 18m taht lisr
    29: 33,  # Ras l-qaws lisr

    # --- L-Wast (Pink / Red) ---
    21: 18,  # Khat l-wast foq
    22: 19,  # Dayra foq
    23: 21,  # Dayra lisr
    24: 22,  # Dayra taht
    25: 20,  # Dayra limn
    26: 9,   # Khat l-wast taht
    33: 39,  # Markaz d-dayra (Center Spot)

    # --- L-Jiha l-Limna (Blue) ---
    11: 1,   # Corner foq limn
    12: 2,   # 18m foq 3la l-khat
    13: 23,  # 18m foq dakhal
    14: 27,  # 18m taht dakhal
    15: 7,   # 18m taht 3la l-khat
    16: 6,   # 6m taht 3la l-khat
    17: 29,  # 6m taht dakhal
    18: 28,  # 6m foq dakhal
    19: 3,   # 6m foq 3la l-khat
    20: 8,   # Corner taht limn
    30: 24,  # Qaws 18m foq limn
    31: 26,  # Qaws 18m taht limn
    32: 25,  # Ras l-qaws limn
}


def remap_yolo_pose_file(txt_path):
    """
    Kay-qra fichier .txt dyal dataset2 (33 points) w kay-3awad i-katbo b 39 points.
    Hada amin: ila lqa fichier deja m-7awal (39 points), kay-tjahlo bach ma-i-khsarch!
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        return "empty"

    new_lines = []
    already_converted = False

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        header = parts[:5]  # class_id, cx, cy, bw, bh
        raw_kpts = parts[5:]

        # Ta7did wach kol noqta fiha 3 arkam (x, y, v) wla 2 (x, y)
        if len(raw_kpts) == TARGET_NUM_KPTS * 3 or len(raw_kpts) == TARGET_NUM_KPTS * 2:
            already_converted = True
            break

        if len(raw_kpts) == 33 * 3:
            step = 3
        elif len(raw_kpts) == 33 * 2:
            step = 2
        else:
            print(f"[ATTENTION] Format ghrib f '{txt_path}' ({len(raw_kpts)} qiyam) -> t-jahalnah.")
            return "skipped"

        # N-wajdo 39 noqta khawyin: (0, 0, 0)
        new_kpts = [("0", "0", "0") for _ in range(TARGET_NUM_KPTS)]

        # N-naqlo kol noqta mn blasstha l-qdima l blasstha j-jdida
        for old_id, new_id in OLD_TO_NEW_MAP.items():
            old_idx = old_id - 1
            new_idx = new_id - 1

            x_str = raw_kpts[old_idx * step]
            y_str = raw_kpts[old_idx * step + 1]
            if step == 3:
                v_str = raw_kpts[old_idx * step + 2]
            else:
                # Ila kanou ghir (x, y), n-diru v=2 ila x wla y > 0
                v_str = "2" if (float(x_str) > 0.0 or float(y_str) > 0.0) else "0"

            new_kpts[new_idx] = (x_str, y_str, v_str)

        # N-jam3o s-satr j-jdid (class + bbox + 39 * 3 = 122 arkam)
        flat_kpts = [val for triplet in new_kpts for val in triplet]
        new_lines.append(" ".join(header + flat_kpts))

    if already_converted:
        return "already_done"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    return "converted"


def main():
    try:
        total_converted = 0
        total_already = 0
        total_empty = 0
        total_skipped = 0

        for split in ["train", "valid", "test"]:
            lbl_dir = os.path.join(DATASET2_DIR, split, "labels")
            if not os.path.exists(lbl_dir):
                print(f"[INFO] Dossier '{lbl_dir}' ma-kaynch (t-jahalnah).")
                continue

            txt_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
            c_split = 0
            for txt_path in txt_files:
                status = remap_yolo_pose_file(txt_path)
                if status == "converted":
                    total_converted += 1
                    c_split += 1
                elif status == "already_done":
                    total_already += 1
                elif status == "empty":
                    total_empty += 1
                else:
                    total_skipped += 1

            print(f"[OK] {split:5s}: t-badlo {c_split}/{len(txt_files)} fichiers .txt")

        print("\n" + "=" * 55)
        print(f"[SAFI] T-ta7wil sala b-naja7!")
        print(f"  - T-badlo (33 -> 39 pts) : {total_converted}")
        print(f"  - Deja m-7awlin          : {total_already}")
        print(f"  - Khawyin                : {total_empty}")
        print(f"  - T-jahlo (format ghrib) : {total_skipped}")
        print("=" * 55)

    except Exception as e:
        print(f"[ERREUR] {e}")


if __name__ == "__main__":
    main()