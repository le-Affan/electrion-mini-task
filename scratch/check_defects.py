import json
import cv2
import numpy as np
import sys
sys.path.append("/home/affan/Projects/Electrion")
from detect import detect_faults, align_images

with open("synthetic_images/labels.json", "r") as f:
    labels = json.load(f)

# Let's check the first 5 images
for i in range(10):
    filename = f"cam_{i:03d}.png"
    info = labels[filename]
    cam_path = f"synthetic_images/{filename}"
    gt_path = f"synthetic_images/{info['gt_filename']}"
    
    preds = detect_faults(cam_path, gt_path, "synthetic_images/results/test_out.png")
    print(f"Image {filename}:")
    print(f"  GT Defects: {info['defects']}")
    print(f"  Pred Defects: {preds}")
    print("-" * 40)
