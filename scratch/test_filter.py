import cv2
import numpy as np
import sys
import os

sys.path.append("/home/affan/Projects/Electrion")
from detect import preprocess_camera, preprocess_groundtruth, align_images

# Load cam_008 (which has 0 real defects but had 9 false positives)
cam_img = cv2.imread("/home/affan/Projects/Electrion/synthetic_images/cam_008.png")
gt_img = cv2.imread("/home/affan/Projects/Electrion/synthetic_images/gt_008.png")

cam_bin = preprocess_camera(cam_img)
gt_bin = preprocess_groundtruth(gt_img)

cam_bin, gt_aligned, dx, dy = align_images(cam_bin, gt_bin)

# Compute difference map
diff = cv2.bitwise_xor(cam_bin, gt_aligned)
diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

# Create boundary band
# Dilate and erode to get the boundary region
k_size = 9 # +/- 4 pixels boundary band
gt_dilated = cv2.dilate(gt_aligned, np.ones((k_size, k_size), np.uint8))
gt_eroded = cv2.erode(gt_aligned, np.ones((k_size, k_size), np.uint8))
boundary_band = cv2.bitwise_xor(gt_dilated, gt_eroded)

# Connected components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)
MIN_AREA = 200

print(f"Total candidate regions before filtering: {num_labels - 1}")

kept_regions = 0
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
        continue
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    
    diff_roi = (labels[y : y + h, x : x + w] == i).astype(np.uint8) * 255
    band_roi = boundary_band[y : y + h, x : x + w]
    
    overlap = np.sum((diff_roi > 0) & (band_roi > 0))
    total = np.sum(diff_roi > 0)
    overlap_ratio = overlap / total
    
    print(f"Candidate {i}: area={total}, overlap_ratio={overlap_ratio:.3f}")
    if overlap_ratio <= 0.85:
        kept_regions += 1
        print("  -> KEPT")
    else:
        print("  -> FILTERED OUT")

print(f"Total kept regions: {kept_regions}")
