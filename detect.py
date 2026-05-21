import sys

import cv2
import numpy as np

# ─────────────────────────────────────────────
#  PCB Fault Detector
#  Detects BREAKS and SHORTS in copper traces
#  by comparing a camera feed against an ideal
#  ground truth layout.
# ─────────────────────────────────────────────


def preprocess_camera(img):
    """
    Approach: divide the image into zones based on brightness,
    apply different thresholds per zone, then combine.

    The core issue is severe lighting gradient — center is ~3x
    brighter than corners. A single threshold (even Otsu) fails
    across such a range. We split into bright/dark zones and
    threshold each independently, then merge.
    """
    b_ch, g_ch, r_ch = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 1: Build copper signal (R - G) ─────────────────────────
    # Copper: high R, low G → large positive difference
    # Glare:  high R, high G → difference near zero
    # Background: low R, low G → near zero
    r_f = r_ch.astype(np.float32)
    g_f = g_ch.astype(np.float32)
    copper_signal = np.clip(r_f - g_f, 0, 255).astype(np.uint8)
    copper_signal = cv2.GaussianBlur(copper_signal, (5, 5), 0)

    # ── Step 2: Zone-based thresholding ─────────────────────────────
    # Blur gray heavily to get a smooth brightness map (the "zone" map)
    brightness_map = cv2.GaussianBlur(gray, (101, 101), 0).astype(np.float32)
    brightness_map = np.clip(brightness_map, 1, 255)  # avoid div by zero

    # Normalize copper signal by local brightness
    # This compensates for the lighting gradient across the board
    copper_norm = copper_signal.astype(np.float32) / brightness_map * 128
    copper_norm = np.clip(copper_norm, 0, 255).astype(np.uint8)

    # Now threshold the normalized signal — Otsu works well on uniform signal
    _, binary = cv2.threshold(copper_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Step 3: Mask out hard glare hotspots ────────────────────────
    # Any pixel where ALL channels are very high is pure glare, not copper
    glare_mask = (
        (r_ch.astype(np.int32) + g_ch.astype(np.int32) + b_ch.astype(np.int32)) > 600
    ).astype(np.uint8) * 255
    glare_kernel = np.ones((13, 13), np.uint8)
    glare_mask = cv2.dilate(glare_mask, glare_kernel)
    binary[glare_mask == 255] = 0

    # ── Step 4: Morphological cleanup ───────────────────────────────
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


def preprocess_groundtruth(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if np.sum(binary == 255) / binary.size > 0.6:
        binary = cv2.bitwise_not(binary)
    return binary


def detect_faults(camera_path, ground_truth_path, output_path="output.png"):
    cam_img = cv2.imread(camera_path)
    gt_img = cv2.imread(ground_truth_path)

    if cam_img is None:
        print(f"[ERROR] Could not load camera image: {camera_path}")
        sys.exit(1)
    if gt_img is None:
        print(f"[ERROR] Could not load ground truth: {ground_truth_path}")
        sys.exit(1)

    print("[1/5] Preprocessing camera image...")
    cam_binary = preprocess_camera(cam_img)

    print("[2/5] Preprocessing ground truth...")
    gt_binary = preprocess_groundtruth(gt_img)

    if cam_binary.shape != gt_binary.shape:
        gt_binary = cv2.resize(
            gt_binary,
            (cam_binary.shape[1], cam_binary.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    print("[3/5] Aligning via phase correlation...")
    shift, _ = cv2.phaseCorrelate(np.float32(cam_binary), np.float32(gt_binary))
    dx, dy = int(round(shift[0])), int(round(shift[1]))
    if abs(dx) <= 20 and abs(dy) <= 20:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        h, w = cam_binary.shape
        cam_binary = cv2.warpAffine(cam_binary, M, (w, h))
        print(f"    Shift corrected: dx={dx}, dy={dy}")
    else:
        print(f"    Shift too large ({dx},{dy}), skipping.")

    print("[4/5] Computing difference map...")
    diff = cv2.bitwise_xor(cam_binary, gt_binary)
    kernel = np.ones((5, 5), np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

    print("[5/5] Finding and classifying defects...")
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        diff, connectivity=8
    )

    result_img = cv2.resize(cam_img.copy(), (cam_binary.shape[1], cam_binary.shape[0]))
    defect_count = {"BREAK": 0, "SHORT": 0}
    MIN_AREA = 300

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        region_gt = gt_binary[y : y + h, x : x + w]
        white_ratio = np.sum(region_gt == 255) / region_gt.size

        if white_ratio > 0.3:
            label = "BREAK"
            color = (0, 0, 255)
        else:
            label = "SHORT"
            color = (0, 165, 255)

        defect_count[label] += 1
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            result_img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
        )

    summary = f"BREAKS: {defect_count['BREAK']}  SHORTS: {defect_count['SHORT']}"
    cv2.rectangle(result_img, (5, 5), (360, 35), (0, 0, 0), -1)
    cv2.putText(
        result_img, summary, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    cv2.imwrite(output_path, result_img)
    print(
        f"\nDone. Detected: {defect_count['BREAK']} break(s), {defect_count['SHORT']} short(s)"
    )

    cv2.imwrite(output_path.replace(".png", "_diffmap.png"), diff)
    cv2.imwrite(output_path.replace(".png", "_cam_binary.png"), cam_binary)
    cv2.imwrite(output_path.replace(".png", "_gt_binary.png"), gt_binary)
    print("Debug images saved.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python detect.py <camera_image> <ground_truth_image> [output_image]"
        )
        sys.exit(1)

    detect_faults(
        sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "output.png"
    )
