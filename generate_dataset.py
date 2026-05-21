import os
import sys
import json
import random
import cv2
import numpy as np

def generate_pcb(img_id, output_dir):
    # Set seed for reproducibility of this specific image
    np.random.seed(img_id)
    random.seed(img_id)
    
    # 1. Initialize images
    W, H = 425, 425
    gt_img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Background green substrate with some texture/noise
    bg_b = np.random.randint(15, 30)
    bg_g = np.random.randint(70, 90)
    bg_r = np.random.randint(10, 20)
    substrate_color = (bg_b, bg_g, bg_r)
    
    cam_img = np.zeros((H, W, 3), dtype=np.uint8)
    cam_img[:] = substrate_color
    
    # Add subtle background texture
    noise_bg = np.random.normal(0, 1.5, cam_img.shape).astype(np.float32)
    cam_img = np.clip(cam_img.astype(np.float32) + noise_bg, 0, 255).astype(np.uint8)
    
    # 2. Generate traces
    num_h = np.random.randint(2, 5)
    num_v = np.random.randint(2, 5)
    
    traces = []
    # Horizontal traces
    for _ in range(num_h):
        y = np.random.randint(40, 385)
        x1 = np.random.randint(30, 100)
        x2 = np.random.randint(325, 395)
        w = np.random.randint(8, 15)
        traces.append({'type': 'H', 'y': y, 'x1': x1, 'x2': x2, 'w': w})
        
    # Vertical traces
    for _ in range(num_v):
        x = np.random.randint(40, 385)
        y1 = np.random.randint(30, 100)
        y2 = np.random.randint(325, 395)
        w = np.random.randint(8, 15)
        traces.append({'type': 'V', 'x': x, 'y1': y1, 'y2': y2, 'w': w})
        
    # Draw traces on gt_img and cam_img
    copper_b = np.random.randint(15, 30)
    copper_g = np.random.randint(110, 125)
    copper_r = np.random.randint(220, 245)
    copper_color = (copper_b, copper_g, copper_r)
    
    for t in traces:
        if t['type'] == 'H':
            p1, p2 = (t['x1'], t['y']), (t['x2'], t['y'])
        else:
            p1, p2 = (t['x'], t['y1']), (t['x'], t['y2'])
            
        cv2.line(gt_img, p1, p2, (255, 255, 255), t['w'])
        cv2.line(cam_img, p1, p2, copper_color, t['w'])
        
    # 3. Determine defect type
    defect_modes = ["CLEAN", "BREAK", "SHORT", "COMBINED"]
    defect_type = defect_modes[img_id % 4]
    
    break_mask = np.zeros((H, W), dtype=np.uint8)
    short_mask = np.zeros((H, W), dtype=np.uint8)
    
    # 4. Inject defects
    # BREAK
    if defect_type in ["BREAK", "COMBINED"]:
        num_breaks = 1 if defect_type == "COMBINED" else np.random.randint(1, 3)
        # Select traces to break
        chosen_traces = random.sample(traces, min(num_breaks, len(traces)))
        for t in chosen_traces:
            r_break = np.random.randint(8, 14)
            if t['type'] == 'H':
                xb = np.random.randint(t['x1'] + 30, t['x2'] - 30)
                break_center = (xb, t['y'])
                temp_trace = np.zeros((H, W), dtype=np.uint8)
                cv2.line(temp_trace, (t['x1'], t['y']), (t['x2'], t['y']), 255, t['w'])
            else:
                yb = np.random.randint(t['y1'] + 30, t['y2'] - 30)
                break_center = (t['x'], yb)
                temp_trace = np.zeros((H, W), dtype=np.uint8)
                cv2.line(temp_trace, (t['x'], t['y1']), (t['x'], t['y2']), 255, t['w'])
                
            temp_eraser = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(temp_eraser, break_center, r_break, 255, -1)
            tb_mask = cv2.bitwise_and(temp_trace, temp_eraser)
            break_mask = cv2.bitwise_or(break_mask, tb_mask)
            
            # Erase copper on cam_img (fill with substrate color)
            cv2.circle(cam_img, break_center, r_break, substrate_color, -1)
            
    # SHORT
    if defect_type in ["SHORT", "COMBINED"]:
        num_shorts = 1 if defect_type == "COMBINED" else np.random.randint(1, 3)
        for _ in range(num_shorts):
            # Select random trace to branch off
            t = random.choice(traces)
            w_short = np.random.randint(6, 10)
            len_short = np.random.randint(15, 30)
            direction = random.choice([-1, 1])
            
            if t['type'] == 'H':
                xs = np.random.randint(t['x1'] + 20, t['x2'] - 20)
                ys = t['y']
                xe = xs
                ye = ys + direction * len_short
            else:
                xs = t['x']
                ys = np.random.randint(t['y1'] + 20, t['y2'] - 20)
                xe = xs + direction * len_short
                ye = ys
                
            cv2.line(cam_img, (xs, ys), (xe, ye), copper_color, w_short)
            cv2.line(short_mask, (xs, ys), (xe, ye), 255, w_short)
            
        # Clean up short mask to only keep added copper that is NOT part of the original GT
        gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        _, gt_bin = cv2.threshold(gt_gray, 127, 255, cv2.THRESH_BINARY)
        short_mask = cv2.bitwise_and(short_mask, cv2.bitwise_not(gt_bin))
        
    # 5. Apply camera effects (glare, brightness, blur, noise)
    # Glare spots
    num_glares = np.random.randint(0, 3)
    for _ in range(num_glares):
        cx = np.random.randint(30, 395)
        cy = np.random.randint(30, 395)
        radius = np.random.randint(50, 160)
        intensity = np.random.uniform(0.2, 0.6)
        
        glare_mask = np.zeros((H, W), dtype=np.float32)
        cv2.circle(glare_mask, (cx, cy), radius, 1.0, -1)
        # Apply massive blur to make it smooth
        glare_mask = cv2.GaussianBlur(glare_mask, (0, 0), sigmaX=radius/1.8)
        glare_3ch = np.stack([glare_mask] * 3, axis=-1)
        
        cam_float = cam_img.astype(np.float32)
        cam_float = cam_float * (1.0 - intensity * glare_3ch) + 255.0 * intensity * glare_3ch
        cam_img = np.clip(cam_float, 0, 255).astype(np.uint8)
        
    # Brightness scale
    brightness = np.random.uniform(0.85, 1.15)
    cam_img = np.clip(cam_img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    
    # Blur
    blur_k = np.random.choice([3, 5])
    cam_img = cv2.GaussianBlur(cam_img, (blur_k, blur_k), 0)
    
    # Add Gaussian Noise
    noise_val = np.random.uniform(1.0, 4.0)
    noise = np.random.normal(0, noise_val, cam_img.shape).astype(np.float32)
    cam_img = np.clip(cam_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 6. Apply translation and rotation warping
    dx = np.random.uniform(-15.0, 15.0)
    dy = np.random.uniform(-15.0, 15.0)
    angle = np.random.uniform(-1.5, 1.5)
    
    center = (W / 2.0, H / 2.0)
    R = cv2.getRotationMatrix2D(center, angle, 1.0)
    R[0, 2] += dx
    R[1, 2] += dy
    
    cam_warped = cv2.warpAffine(cam_img, R, (W, H), borderValue=substrate_color)
    break_warped = cv2.warpAffine(break_mask, R, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    short_warped = cv2.warpAffine(short_mask, R, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    
    # Extract defect bounding boxes
    defects = []
    
    # Get BREAK bboxes
    contours_brk, _ = cv2.findContours(break_warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_brk:
        if cv2.contourArea(c) >= 12:
            x, y, w, h = cv2.boundingRect(c)
            # Make sure it's not a border artifact (away from margins)
            if x >= 5 and y >= 5 and (x + w) <= W - 5 and (y + h) <= H - 5:
                defects.append({"type": "BREAK", "bbox": [int(x), int(y), int(w), int(h)]})
                
    # Get SHORT bboxes
    contours_srt, _ = cv2.findContours(short_warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_srt:
        if cv2.contourArea(c) >= 12:
            x, y, w, h = cv2.boundingRect(c)
            if x >= 5 and y >= 5 and (x + w) <= W - 5 and (y + h) <= H - 5:
                defects.append({"type": "SHORT", "bbox": [int(x), int(y), int(w), int(h)]})
                
    # Save the files
    cam_filename = f"cam_{img_id:03d}.png"
    gt_filename = f"gt_{img_id:03d}.png"
    
    cv2.imwrite(os.path.join(output_dir, cam_filename), cam_warped)
    cv2.imwrite(os.path.join(output_dir, gt_filename), gt_img)
    
    is_clean = len(defects) == 0
    
    return cam_filename, gt_filename, {
        "filename": cam_filename,
        "gt_filename": gt_filename,
        "defect_type": defect_type,
        "defect_count": len(defects),
        "defects": defects,
        "is_clean": is_clean,
        "is_faulty": not is_clean
    }

def main():
    output_dir = "/home/affan/Projects/Electrion/synthetic_images"
    os.makedirs(output_dir, exist_ok=True)
    
    labels = {}
    print("Generating 100 synthetic PCB image pairs...")
    for i in range(100):
        cam_file, gt_file, info = generate_pcb(i, output_dir)
        labels[cam_file] = info
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/100 samples.")
            
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=4)
        
    print(f"Dataset generated. Labels saved to {labels_path}")

if __name__ == "__main__":
    main()
