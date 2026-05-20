import os
import sys

import cv2
import numpy as np

# ─────────────────────────────────────────────
#  PCB Fault Detector
#  Detects BREAKS and SHORTS in copper traces
#  by comparing a camera feed against an ideal
#  ground truth layout.
# ─────────────────────────────────────────────


def preprocess(img):
    """
    Clean up a raw camera image:
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


def align(camera_binary, ground_truth_binary):
    """
    Align the camera image to the ground truth using
    ORB feature matching + homography (RANSAC).
    Falls back to returning the original if alignment fails.
    """
    orb = cv2.ORB_create(nfeatures=1000)

    kp1, des1 = orb.detectAndCompute(camera_binary, None)
    kp2, des2 = orb.detectAndCompute(ground_truth_binary, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("[WARN] Not enough features for alignment, skipping warp.")
        return camera_binary

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        print("[WARN] Not enough matches for homography, skipping warp.")
        return camera_binary

    # Use top matches
    good = matches[:50]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("[WARN] Homography failed, skipping warp.")
        return camera_binary

    h, w = ground_truth_binary.shape
    aligned = cv2.warpPerspective(camera_binary, H, (w, h))
    return aligned


def detect_faults(camera_path, ground_truth_path, output_path="output.png"):
    """
    Main detection function.
    Loads both images, preprocesses, aligns, diffs, and
    draws labelled bounding boxes for each defect found.
    """
    cam_img = cv2.imread(camera_path)
    gt_img = cv2.imread(ground_truth_path)

    if cam_img is None:
        print(f"[ERROR] Could not load camera image: {camera_path}")
        sys.exit(1)
    if gt_img is None:
        print(f"[ERROR] Could not load ground truth image: {ground_truth_path}")
        sys.exit(1)

    # Resize ground truth to match camera image size if needed
    if cam_img.shape != gt_img.shape:
        gt_img = cv2.resize(gt_img, (cam_img.shape[1], cam_img.shape[0]))

    print("[1/5] Preprocessing camera image...")
    cam_binary = preprocess(cam_img)

    print("[2/5] Preprocessing ground truth...")
    gt_binary = preprocess(gt_img)

    print("[3/5] Aligning images...")
    aligned_cam = align(cam_binary, gt_binary)

    print("[4/5] Computing difference map...")
    diff = cv2.bitwise_xor(aligned_cam, gt_binary)

    # Remove tiny noise regions (anything smaller than 100px area)
    kernel = np.ones((5, 5), np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

    print("[5/5] Finding and classifying defects...")
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        diff, connectivity=8
    )

    # Draw results on a copy of the camera image
    result_img = cam_img.copy()

    defect_count = {"BREAK": 0, "SHORT": 0}

    for i in range(1, num_labels):  # skip label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 100:  # ignore tiny artifacts
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # Sample the ground truth in this region to classify
        region_gt = gt_binary[y : y + h, x : x + w]
        white_ratio = np.sum(region_gt == 255) / region_gt.size

        if white_ratio > 0.3:
            # Ground truth says copper should be here but camera doesn't have it
            label = "BREAK"
            color = (0, 0, 255)  # Red
        else:
            # Ground truth says background but camera shows copper
            label = "SHORT"
            color = (0, 165, 255)  # Orange

        defect_count[label] += 1
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            result_img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    # Summary box in top-left
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

    # Also save the diff map for inspection
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
