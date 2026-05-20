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
    Clean up the raw camera image:
    1. Grayscale
    2. Gaussian blur  → kills sensor noise
    3. CLAHE          → evens out glare / uneven lighting
    4. Adaptive threshold → clean binary image
    5. Morphological open/close → remove leftover speckle
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)
    binary = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def preprocess_groundtruth(img):
    """
    Ground truth is already clean black & white —
    just convert to grayscale and threshold cleanly.
    No need for CLAHE or heavy noise removal.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def detect_faults(camera_path, ground_truth_path, output_path="output.png"):
    cam_img = cv2.imread(camera_path)
    gt_img = cv2.imread(ground_truth_path)

    if cam_img is None:
        print(f"[ERROR] Could not load camera image: {camera_path}")
        sys.exit(1)
    if gt_img is None:
        print(f"[ERROR] Could not load ground truth image: {ground_truth_path}")
        sys.exit(1)

    print("[1/5] Preprocessing camera image...")
    cam_binary = preprocess_camera(cam_img)

    print("[2/5] Preprocessing ground truth...")
    gt_binary = preprocess_groundtruth(gt_img)

    # Resize to match if needed
    if cam_binary.shape != gt_binary.shape:
        gt_binary = cv2.resize(
            gt_binary,
            (cam_binary.shape[1], cam_binary.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    print("[3/5] Aligning via translation correction...")
    # Use phase correlation to find any minor shift between images
    shift, _ = cv2.phaseCorrelate(np.float32(cam_binary), np.float32(gt_binary))
    dx, dy = int(round(shift[0])), int(round(shift[1]))
    # Only apply if shift is small (within 20px) — avoids bad corrections
    if abs(dx) <= 20 and abs(dy) <= 20:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        h, w = cam_binary.shape
        cam_binary = cv2.warpAffine(cam_binary, M, (w, h))
        print(f"    Shift corrected: dx={dx}, dy={dy}")
    else:
        print(f"    Shift too large ({dx},{dy}), skipping correction.")

    print("[4/5] Computing difference map...")
    diff = cv2.bitwise_xor(cam_binary, gt_binary)

    # Clean up tiny noise regions in the diff
    kernel = np.ones((5, 5), np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

    print("[5/5] Finding and classifying defects...")
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        diff, connectivity=8
    )

    result_img = cam_img.copy()
    # Resize result to match cam if needed
    if result_img.shape[:2] != (cam_binary.shape[0], cam_binary.shape[1]):
        result_img = cv2.resize(result_img, (cam_binary.shape[1], cam_binary.shape[0]))

    defect_count = {"BREAK": 0, "SHORT": 0}
    MIN_AREA = 300  # ignore regions smaller than this (px)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # Check what ground truth says about this region
        region_gt = gt_binary[y : y + h, x : x + w]
        white_ratio = np.sum(region_gt == 255) / region_gt.size

        if white_ratio > 0.3:
            label = "BREAK"
            color = (0, 0, 255)  # Red
        else:
            label = "SHORT"
            color = (0, 165, 255)  # Orange

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
    print(f"Output saved to: {output_path}")

    diff_path = output_path.replace(".png", "_diffmap.png")
    cv2.imwrite(diff_path, diff)
    print(f"Diff map saved to: {diff_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python detect.py <camera_image> <ground_truth_image> [output_image]"
        )
        print("Example: python detect.py camera.png ground_truth.png result.png")
        sys.exit(1)

    camera_path = sys.argv[1]
    ground_truth_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output.png"

    detect_faults(camera_path, ground_truth_path, output_path)
