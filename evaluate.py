import os
import sys
import json
import cv2
import numpy as np

# Ensure project root is in the path
sys.path.append("/home/affan/Projects/Electrion")
from detect import detect_faults

def compute_iou(box1, box2):
    # box format: [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=8):
    x1, y1 = pt1
    x2, y2 = pt2
    # Draw horizontal dashed lines
    for x in range(x1, x2, dash_len * 2):
        cv2.line(img, (x, y1), (min(x + dash_len, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash_len, x2), y2), color, thickness)
    # Draw vertical dashed lines
    for y in range(y1, y2, dash_len * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_len, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash_len, y2)), color, thickness)

def main():
    dataset_dir = "/home/affan/Projects/Electrion/synthetic_images"
    labels_path = os.path.join(dataset_dir, "labels.json")
    results_dir = os.path.join(dataset_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(labels_path):
        print(f"[ERROR] Labels file not found at {labels_path}")
        sys.exit(1)
        
    with open(labels_path, "r") as f:
        labels = json.load(f)
        
    print(f"Loaded {len(labels)} dataset labels.")
    
    # Metrics accumulator
    defect_tp = 0
    defect_fp = 0
    defect_fn = 0
    
    image_tp = 0  # faulty correctly classified
    image_tn = 0  # clean correctly classified
    image_fp = 0  # clean classified as faulty
    image_fn = 0  # faulty classified as clean
    
    for filename, info in labels.items():
        cam_path = os.path.join(dataset_dir, filename)
        gt_path = os.path.join(dataset_dir, info["gt_filename"])
        
        # Temp output path for debug files created by detect.py
        temp_out = os.path.join(results_dir, f"temp_{filename}")
        
        # Run detection
        predictions = detect_faults(cam_path, gt_path, temp_out)
        
        # Clean up the temporary debug outputs from detect.py to avoid cluttering results
        # keep the main output or recreate it beautifully
        if os.path.exists(temp_out):
            os.remove(temp_out)
        out_name = os.path.basename(temp_out).replace(".png", "")
        for suffix in ["_diffmap.png", "_cam_binary.png", "_gt_binary.png"]:
            p = os.path.join("debug", f"{out_name}{suffix}")
            if os.path.exists(p):
                os.remove(p)
                
        # Ground truth defects
        gt_defects = info["defects"]
        
        # Defect-level matching
        # Sort possible matching pairs by IoU
        pairs = []
        for g_idx, g_def in enumerate(gt_defects):
            for p_idx, p_def in enumerate(predictions):
                if g_def["type"] == p_def["type"]:
                    iou = compute_iou(g_def["bbox"], p_def["bbox"])
                    if iou >= 0.3:
                        pairs.append((iou, g_idx, p_idx))
                        
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        matched_gt = set()
        matched_pred = set()
        
        image_defect_tp = 0
        for iou, g_idx, p_idx in pairs:
            if g_idx not in matched_gt and p_idx not in matched_pred:
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                image_defect_tp += 1
                
        image_defect_fp = len(predictions) - len(matched_pred)
        image_defect_fn = len(gt_defects) - len(matched_gt)
        
        defect_tp += image_defect_tp
        defect_fp += image_defect_fp
        defect_fn += image_defect_fn
        
        # Image-level logic
        gt_has_defect = len(gt_defects) > 0
        pred_has_defect = len(predictions) > 0
        
        if gt_has_defect and pred_has_defect:
            image_tp += 1
        elif not gt_has_defect and not pred_has_defect:
            image_tn += 1
        elif not gt_has_defect and pred_has_defect:
            image_fp += 1
        elif gt_has_defect and not pred_has_defect:
            image_fn += 1
            
        # Draw custom visualization for this image
        cam_visual = cv2.imread(cam_path)
        
        # Draw GT bboxes (Blue dashed lines)
        for g_def in gt_defects:
            x, y, w, h = g_def["bbox"]
            draw_dashed_rect(cam_visual, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
            cv2.putText(cam_visual, f"GT {g_def['type']}", (x, y + h + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
        # Draw Pred bboxes (Red/Orange solid lines)
        for p_def in predictions:
            x, y, w, h = p_def["bbox"]
            color = (0, 0, 255) if p_def["type"] == "BREAK" else (0, 165, 255)
            cv2.rectangle(cam_visual, (x, y), (x + w, y + h), color, 2)
            cv2.putText(cam_visual, f"PRED {p_def['type']}", (x, y - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
        # Draw summary overlay on the image
        summary_text = f"GT: {len(gt_defects)} | PRED: {len(predictions)} (TP:{image_defect_tp} FP:{image_defect_fp} FN:{image_defect_fn})"
        cv2.rectangle(cam_visual, (5, 5), (320, 30), (0, 0, 0), -1)
        cv2.putText(cam_visual, summary_text, (10, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Save to results directory
        vis_path = os.path.join(results_dir, f"eval_{filename}")
        cv2.imwrite(vis_path, cam_visual)
        
    # Print overall statistics
    total_images = len(labels)
    defect_precision = defect_tp / (defect_tp + defect_fp) if (defect_tp + defect_fp) > 0 else 0.0
    defect_recall = defect_tp / (defect_tp + defect_fn) if (defect_tp + defect_fn) > 0 else 0.0
    defect_f1 = 2 * defect_precision * defect_recall / (defect_precision + defect_recall) if (defect_precision + defect_recall) > 0 else 0.0
    
    img_precision = image_tp / (image_tp + image_fp) if (image_tp + image_fp) > 0 else 0.0
    img_recall = image_tp / (image_tp + image_fn) if (image_tp + image_fn) > 0 else 0.0
    img_accuracy = (image_tp + image_tn) / total_images
    img_f1 = 2 * img_precision * img_recall / (img_precision + img_recall) if (img_precision + img_recall) > 0 else 0.0
    
    print("\n" + "="*50)
    print("             EVALUATION REPORT")
    print("="*50)
    print(f"Total Images Evaluated: {total_images}")
    print("-"*50)
    print("DEFECT-LEVEL METRICS:")
    print(f"  True Positives (TP):  {defect_tp}")
    print(f"  False Positives (FP): {defect_fp}")
    print(f"  False Negatives (FN): {defect_fn}")
    print(f"  Precision:            {defect_precision:.4f} ({defect_precision:.1%})")
    print(f"  Recall:               {defect_recall:.4f} ({defect_recall:.1%})")
    print(f"  F1 Score:             {defect_f1:.4f}")
    print("-"*50)
    print("IMAGE-LEVEL METRICS:")
    print(f"  True Positives (TP):  {image_tp}  (faulty correctly flagged)")
    print(f"  True Negatives (TN):  {image_tn}  (clean correctly flagged)")
    print(f"  False Positives (FP): {image_fp}  (clean flagged as faulty)")
    print(f"  False Negatives (FN): {image_fn}  (faulty flagged as clean)")
    print(f"  Accuracy:             {img_accuracy:.4f} ({img_accuracy:.1%})")
    print(f"  Precision:            {img_precision:.4f} ({img_precision:.1%})")
    print(f"  Recall:               {img_recall:.4f} ({img_recall:.1%})")
    print(f"  F1 Score:             {img_f1:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
