import cv2
import numpy as np
import sys
import os

# Load images
cam_img = cv2.imread("/home/affan/Projects/Electrion/synthetic_images/cam_001.png")
gt_img = cv2.imread("/home/affan/Projects/Electrion/synthetic_images/gt_001.png")

from detect import preprocess_camera, preprocess_groundtruth, align_images

cam_bin = preprocess_camera(cam_img)
gt_bin = preprocess_groundtruth(gt_img)

# Initial phase correlation alignment
h_cam, w_cam = cam_bin.shape
h_gt, w_gt = gt_bin.shape
max_h = max(h_cam, h_gt)
max_w = max(w_cam, w_gt)

cam_padded = np.zeros((max_h, max_w), dtype=np.uint8)
cam_padded[0:h_cam, 0:w_cam] = cam_bin

gt_padded = np.zeros((max_h, max_w), dtype=np.uint8)
gt_padded[0:h_gt, 0:w_gt] = gt_bin

dist_cam = cv2.distanceTransform(cam_padded, cv2.DIST_L2, 5)
dist_gt = cv2.distanceTransform(gt_padded, cv2.DIST_L2, 5)

shift, response = cv2.phaseCorrelate(
    dist_cam.astype(np.float32), dist_gt.astype(np.float32)
)
dx, dy = int(round(shift[0])), int(round(shift[1]))
print(f"Phase correlation shift: dx={dx}, dy={dy}")

# Initialize Euclidean warp matrix
warp_matrix = np.eye(2, 3, dtype=np.float32)
warp_matrix[0, 2] = -dx
warp_matrix[1, 2] = -dy

# Run ECC
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
try:
    # Use distance transforms for ECC since they have smooth gradients
    cc, warp_matrix = cv2.findTransformECC(
        dist_cam.astype(np.float32),
        dist_gt.astype(np.float32),
        warp_matrix,
        cv2.MOTION_EUCLIDEAN,
        criteria
    )
    print("ECC Converged, cc =", cc)
    print("Warp Matrix:\n", warp_matrix)
    
    # Warp gt_bin with the refined warp matrix
    gt_aligned = cv2.warpAffine(
        gt_bin, warp_matrix, (w_cam, h_cam), flags=cv2.INTER_NEAREST, borderValue=0
    )
    diff = cv2.bitwise_xor(cam_bin, gt_aligned)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)
    print(f"Number of detected defect regions after ECC: {num_labels - 1}")
    for i in range(1, num_labels):
        print(f"  Region {i}: area = {stats[i, cv2.CC_STAT_AREA]}, left={stats[i, cv2.CC_STAT_LEFT]}, top={stats[i, cv2.CC_STAT_TOP]}")
except Exception as e:
    print("ECC failed:", e)
