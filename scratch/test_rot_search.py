import cv2
import numpy as np
import sys
import os

sys.path.append("/home/affan/Projects/Electrion")
from detect import preprocess_camera, preprocess_groundtruth

# Load test images
for filename in ["cam_001.png", "cam_008.png"]:
    cam_img = cv2.imread(f"/home/affan/Projects/Electrion/synthetic_images/{filename}")
    info_path = "/home/affan/Projects/Electrion/synthetic_images/labels.json"
    import json
    with open(info_path, "r") as f:
        labels = json.load(f)
    info = labels[filename]
    gt_img = cv2.imread(f"/home/affan/Projects/Electrion/synthetic_images/{info['gt_filename']}")

    cam_bin = preprocess_camera(cam_img)
    gt_bin = preprocess_groundtruth(gt_img)

    h_cam, w_cam = cam_bin.shape
    h_gt, w_gt = gt_bin.shape
    max_h = max(h_cam, h_gt)
    max_w = max(w_cam, w_gt)

    cam_padded = np.zeros((max_h, max_w), dtype=np.uint8)
    cam_padded[0:h_cam, 0:w_cam] = cam_bin
    dist_cam = cv2.distanceTransform(cam_padded, cv2.DIST_L2, 5)

    # Search rotation angles
    best_response = -1.0
    best_angle = 0.0
    best_shift = (0.0, 0.0)

    # Try angles from -2.0 to 2.0 with step 0.2
    for angle in np.arange(-2.0, 2.1, 0.2):
        # Rotate gt_bin
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

    print(f"\nImage {filename}:")
    print(f"  Best Angle: {best_angle:.2f} degrees (response={best_response:.4f}, shift={best_shift})")

    # Apply the best rotation and translation
    center = (w_gt / 2.0, h_gt / 2.0)
    R = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    # Apply rotation to GT
    gt_rot = cv2.warpAffine(gt_bin, R, (w_gt, h_gt), flags=cv2.INTER_NEAREST, borderValue=0)
    # Apply translation from phase correlation
    dx, dy = int(round(best_shift[0])), int(round(best_shift[1]))
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    gt_aligned = cv2.warpAffine(
        gt_rot, M, (w_cam, h_cam), flags=cv2.INTER_NEAREST, borderValue=0
    )

    diff = cv2.bitwise_xor(cam_bin, gt_aligned)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)
    print(f"  Defect regions detected: {num_labels - 1}")
    for i in range(1, num_labels):
        print(f"    Region {i}: area={stats[i, cv2.CC_STAT_AREA]}, left={stats[i, cv2.CC_STAT_LEFT]}, top={stats[i, cv2.CC_STAT_TOP]}")
