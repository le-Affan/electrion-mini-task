import os
import sys

import cv2
import numpy as np


def preprocess_camera(img):
    # ── Step 1: Compute R - G difference ────────────────────────────
    # Copper has higher Red than Green. Background is green substrate.
    b_ch, g_ch, r_ch = cv2.split(img)
    diff_rg = r_ch.astype(np.int16) - g_ch.astype(np.int16)
    
    # Threshold at 4 to isolate copper from green substrate
    binary = (diff_rg > 4).astype(np.uint8) * 255

    # ── Step 2: Clean up ─────────────────────────────────────────────
    # Use morphological closing and opening to remove sensor noise and fill tiny gaps
    k3 = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=1)

    # Clear 10-pixel border to eliminate any potential border artifacts
    margin = 10
    binary[0:margin, :] = 0
    binary[-margin:, :] = 0
    binary[:, 0:margin] = 0
    binary[:, -margin:] = 0

    return binary


def preprocess_groundtruth(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if np.mean(binary == 255) > 0.50:
        binary = cv2.bitwise_not(binary)
        
    # Clear 10-pixel border to eliminate light gray border artifacts
    margin = 10
    binary[0:margin, :] = 0
    binary[-margin:, :] = 0
    binary[:, 0:margin] = 0
    binary[:, -margin:] = 0
    
    print(f"    GT white%={np.mean(binary == 255):.1%}")
    return binary


def align_images(cam_bin, gt_bin):
    h_cam, w_cam = cam_bin.shape
    h_gt, w_gt = gt_bin.shape
    max_h = max(h_cam, h_gt)
    max_w = max(w_cam, w_gt)

    cam_padded = np.zeros((max_h, max_w), dtype=np.uint8)
    cam_padded[0:h_cam, 0:w_cam] = cam_bin
    dist_cam = cv2.distanceTransform(cam_padded, cv2.DIST_L2, 5)

    best_response = -1.0
    best_angle = 0.0
    best_shift = (0.0, 0.0)

    # 1. Coarse search: -2.0 to 2.0 degrees in steps of 0.5
    for angle in np.arange(-2.0, 2.1, 0.5):
        center = (w_gt / 2.0, h_gt / 2.0)
        R = cv2.getRotationMatrix2D(center, angle, 1.0)
        gt_rot = cv2.warpAffine(gt_bin, R, (w_gt, h_gt), flags=cv2.INTER_NEAREST, borderValue=0)
        
        gt_padded = np.zeros((max_h, max_w), dtype=np.uint8)
        gt_padded[0:h_gt, 0:w_gt] = gt_rot
        dist_rot_gt = cv2.distanceTransform(gt_padded, cv2.DIST_L2, 5)
        
        shift, response = cv2.phaseCorrelate(
            dist_cam.astype(np.float32), dist_rot_gt.astype(np.float32)
        )
        if response > best_response:
            best_response = response
            best_angle = angle
            best_shift = shift

    # 2. Fine search: refine around best coarse angle in steps of 0.1
    coarse_best = best_angle
    for angle in np.arange(coarse_best - 0.4, coarse_best + 0.41, 0.1):
        if abs(angle - coarse_best) < 1e-5:
            continue
        center = (w_gt / 2.0, h_gt / 2.0)
        R = cv2.getRotationMatrix2D(center, angle, 1.0)
        gt_rot = cv2.warpAffine(gt_bin, R, (w_gt, h_gt), flags=cv2.INTER_NEAREST, borderValue=0)
        
        gt_padded = np.zeros((max_h, max_w), dtype=np.uint8)
        gt_padded[0:h_gt, 0:w_gt] = gt_rot
        dist_rot_gt = cv2.distanceTransform(gt_padded, cv2.DIST_L2, 5)
        
        shift, response = cv2.phaseCorrelate(
            dist_cam.astype(np.float32), dist_rot_gt.astype(np.float32)
        )
        if response > best_response:
            best_response = response
            best_angle = angle
            best_shift = shift

    dx, dy = int(round(best_shift[0])), int(round(best_shift[1]))
    print(f"    Alignment found: angle={best_angle:.2f} deg, dx={dx}, dy={dy} (response={best_response:.3f})")

    # Warp using the optimal rotation + translation parameters
    center = (w_gt / 2.0, h_gt / 2.0)
    R = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    gt_rot = cv2.warpAffine(gt_bin, R, (w_gt, h_gt), flags=cv2.INTER_NEAREST, borderValue=0)
    
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    gt_aligned = cv2.warpAffine(
        gt_rot, M, (w_cam, h_cam), flags=cv2.INTER_NEAREST, borderValue=0
    )
    
    return cam_bin, gt_aligned, dx, dy


def classify_defect(diff_roi, cam_roi, gt_roi):
    diff_size = np.sum(diff_roi > 0)
    if diff_size == 0:
        return None
    overlap_gt = np.sum((diff_roi > 0) & (gt_roi == 255))
    overlap_cam = np.sum((diff_roi > 0) & (cam_roi == 255))
    return "BREAK" if (overlap_gt / diff_size) >= (overlap_cam / diff_size) else "SHORT"


def detect_faults(camera_path, ground_truth_path, output_path="output.png"):
    cam_img = cv2.imread(camera_path)
    gt_img = cv2.imread(ground_truth_path)
    if cam_img is None:
        print(f"[ERROR] {camera_path}")
        sys.exit(1)
    if gt_img is None:
        print(f"[ERROR] {ground_truth_path}")
        sys.exit(1)

    print("[1/5] Preprocessing camera image...")
    cam_binary = preprocess_camera(cam_img)
    print(f"    cam_binary white%={np.mean(cam_binary == 255):.1%}")

    print("[2/5] Preprocessing ground truth...")
    gt_binary = preprocess_groundtruth(gt_img)

    print("[3/5] Aligning via phase correlation...")
    cam_binary, gt_binary, dx, dy = align_images(cam_binary, gt_binary)

    print("[4/5] Computing difference map...")
    diff = cv2.bitwise_xor(cam_binary, gt_binary)
    diff = cv2.morphologyEx(
        diff, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
    )

    print("[5/5] Finding and classifying defects...")
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        diff, connectivity=8
    )
    result_img = cam_img.copy()  # Draw result on camera image directly
    defect_count = {"BREAK": 0, "SHORT": 0}
    detected_defects = []
    MIN_AREA = 50

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        diff_roi = (labels[y : y + h, x : x + w] == i).astype(np.uint8) * 255
        label = classify_defect(
            diff_roi, cam_binary[y : y + h, x : x + w], gt_binary[y : y + h, x : x + w]
        )
        if label is None:
            continue
        color = (0, 0, 255) if label == "BREAK" else (0, 165, 255)
        defect_count[label] += 1
        detected_defects.append({"type": label, "bbox": [int(x), int(y), int(w), int(h)]})
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

    # Save debug images in a separate debug/ directory
    debug_dir = "debug"
    os.makedirs(debug_dir, exist_ok=True)
    out_name = os.path.basename(output_path).replace(".png", "")
    cv2.imwrite(os.path.join(debug_dir, f"{out_name}_diffmap.png"), diff)
    cv2.imwrite(os.path.join(debug_dir, f"{out_name}_cam_binary.png"), cam_binary)
    cv2.imwrite(os.path.join(debug_dir, f"{out_name}_gt_binary.png"), gt_binary)
    print("Debug images saved.")
    
    return detected_defects


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python detect.py <camera_image> <ground_truth_image> [output_image]"
        )
        sys.exit(1)
    detect_faults(
        sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "output.png"
    )
