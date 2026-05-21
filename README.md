# PCB Fault Detection System

An automated visual inspection and defect detection pipeline for Printed Circuit Boards (PCBs). The system aligns camera-captured board images to an ideal reference layout using a rotation-robust coarse-to-fine distance transform search, then detects and classifies trace defects (breaks and shorts) with high precision.

---

## 📂 Repository Directory Layout

The repository is organized cleanly, keeping all input, output, and temporary debug files separated from the core source code. Below is the explanation of each folder and file in the root directory:

```text
.
├── debug/                      # Intermediate pipeline images (for troubleshooting)
│   ├── result_cam_binary.png   # Preprocessed camera binary trace mask
│   ├── result_diffmap.png      # Morphological XOR difference map (mismatches)
│   └── result_gt_binary.png    # Preprocessed ground truth binary trace mask
│
├── inputs/                     # Reference input images for testing
│   ├── camera.png              # Test camera-captured image (with glare, noise, rotation)
│   └── ground_truth.png        # Ideal reference PCB Gerber layout
│
├── outputs/                    # Final annotated output images
│   └── result.png              # Final output image with detected defects highlighted
│
├── synthetic_images/           # Procedurally generated evaluation dataset
│   ├── cam_000.png...cam_099.png # 100 simulated camera images
│   ├── gt_000.png...gt_099.png  # 100 corresponding clean Gerber layout masks
│   ├── labels.json             # Ground truth label metadata (classes and bounding boxes)
│   └── results/                # Visual evaluation report overlays
│
├── scratch/                    # Temporary working directory for scratch files/experimental code
│
├── detect.py                   # Main detection pipeline script
├── evaluate.py                 # Batch dataset evaluation script
├── generate_dataset.py         # Procedurally generated synthetic PCB image pair creator
├── pipeline_documentation.md   # Technical document explaining the mathematical pipeline
├── requirements.txt            # Python library dependencies
└── README.md                   # Repository documentation and run instructions (this file)
```

---

## 🔍 File and Folder Details

### Directories

* **[`inputs/`](file:///home/affan/Projects/Electrion/inputs)**: Houses the default input test images. `camera.png` simulates a real camera acquisition with green substrate, copper traces, glare, lens blur, perspective rotation/shift, and faults. `ground_truth.png` is the vector-like ideal layout.
* **[`outputs/`](file:///home/affan/Projects/Electrion/outputs)**: Holds the final output visualizations. `result.png` displays bounding boxes around detected defect areas, color-coded by class (Red for `BREAK`, Orange for `SHORT`), along with a summary banner.
* **[`debug/`](file:///home/affan/Projects/Electrion/debug)**: Stores the intermediate binary stages of the pipeline to keep the root workspace clean. It includes camera and ground truth binarizations and the difference map used for contour selection.
* **[`synthetic_images/`](file:///home/affan/Projects/Electrion/synthetic_images)**: Holds the procedurally generated 100-pair PCB dataset. The evaluation output overlays showing ground-truth-to-prediction bounding box overlaps are saved under `synthetic_images/results/`.
* **[`scratch/`](file:///home/affan/Projects/Electrion/scratch)**: Reserved directory for debugging scripts, scratch code, and experimental algorithms.

### Source Files

* **[`detect.py`](file:///home/affan/Projects/Electrion/detect.py)**: The main pipeline execution script. It loads the source images, performs glare-robust subtraction binarization, aligns the boards via coarse-to-fine rotation search, computes the difference map, runs connected-component filtering, labels defects, and saves the annotated images.
* **[`generate_dataset.py`](file:///home/affan/Projects/Electrion/generate_dataset.py)**: Generates 100 pairs of PCB captures. It uses randomized parameters for green substrate, trace routing, translation ($\pm 15$ px), rotation ($\pm 1.5^\circ$), lens blur, sensor noise, and glare spots. It outputs labels including class types and bounding boxes to `labels.json`.
* **[`evaluate.py`](file:///home/affan/Projects/Electrion/evaluate.py)**: Loads the synthetic dataset and runs the batch evaluation. It computes bounding box matching using Intersection over Union (IoU $\ge 0.3$) and prints defect-level and image-level Precision, Recall, Accuracy, and F1 scores. It deletes temporary run-time debug logs to maintain cleanliness.
* **[`pipeline_documentation.md`](file:///home/affan/Projects/Electrion/pipeline_documentation.md)**: A detailed technical walkthrough of the underlying vision algorithms (glare subtraction, Euclidean distance transform, phase correlation, and morphological classification).
* **[`requirements.txt`](file:///home/affan/Projects/Electrion/requirements.txt)**: Lists the external Python libraries required to run this project (`numpy`, `opencv-python`).

---

## 🚀 How to Run the Pipeline

### 1. Install Dependencies
Make sure you have Python 3 and the necessary libraries installed:
```bash
pip install -r requirements.txt
```

### 2. Run Fault Detection on Baseline Images
To run the fault detection pipeline on the provided baseline camera and ground truth images:
```bash
python3 detect.py inputs/camera.png inputs/ground_truth.png outputs/result.png
```
This will:
* Save the annotated results image in `outputs/result.png`.
* Save intermediate binary masks and diffmaps in `debug/`.

### 3. Generate the Procedural Synthetic Dataset
To regenerate the 100 evaluation samples and label file:
```bash
python3 generate_dataset.py
```
This generates the images directly under `synthetic_images/`.

### 4. Run the Evaluation Script
To run the evaluation script against the generated dataset:
```bash
python3 evaluate.py
```
This will output the classification metrics report in the console and save the comparison visualizations in `synthetic_images/results/`.
