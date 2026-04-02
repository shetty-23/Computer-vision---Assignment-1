# Motion-Degraded GoPro Imagery: Restoration and Object Detection Pipeline

**Author:** Shashanka Shetty | **Course:** COMP 6001 (Assignment 1)

## 📝 Overview
This repository provides a reproducible pipeline for enhancing object detection in high-motion environments. Using the **GoPro Dataset**, we address the **Domain Shift** caused by motion blur[cite: 4, 25]. The project implements a multi-stage approach: quantifying blur severity using **Laplacian Variance** for objective dataset stratification; then applying **Richardson-Lucy deconvolution** with symmetric edge-padding to restore gradients while minimizing ringing artifacts[cite: 39, 42].

To bridge the gap between blurred inputs and sharp ground truths, we leverage a **YOLOv8s** architecture fine-tuned specifically on the restoration-artifact domain[cite: 50, 55]. By training the model to "see through" mathematical noise, we achieved a final **mAP50 of 0.455**, statistically matching the **0.464** performance ceiling of uncorrupted footage[cite: 57, 58].

## 📂 Repository Structure
```text
├── data/
│   ├── raw/                # Original GoPro Blur/Sharp pairs [cite: 21]
│   └── processed/          # Stratified (Mild/Med/Severe) splits 
├── models/
│   └── weights/            # Pre-trained and fine-tuned .pt checkpoints 
├── scripts/
│   ├── task2_deblur.py     # R-L restoration with edge-padding logic 
│   ├── task3_baseline.py   # Baseline inference and Domain Shift analysis [cite: 45]
│   ├── task4_pipeline.py   # Dataset prep, stratification, and training [cite: 51, 55]
│   └── task5_evaluation.py # Benchmarking and metric generation 
├── outputs/
│   ├── charts/             # Master Dashboards & PR curves [cite: 58]
│   └── inference/          # Side-by-side failure case visualizations [cite: 59]
├── requirements.txt        # Modular dependency list
├── ai_log.md               # Mandatory AI prompt & ethical documentation 
└── README.md               # Setup and architecture documentation [cite: 36]
```

## ⚙️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/shetty-23/Computer-vision---Assignment-1.git
   cd Computer-vision---Assignment-1
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Core dependencies: `ultralytics`, `opencv-python`, `scikit-image`, `pandas`, `matplotlib`)*

## 🚀 Usage Guide
### 1. Data Engineering & Stratification (Task 4)
Categorize the raw dataset into blur intensity levels (70/15/15 split):
```bash
python scripts/task4_pipeline.py --mode stratify --input ./data/raw
```
### 2. Image Restoration (Task 2)
Run batch deconvolution with a 15x15 linear Point Spread Function (PSF):
```bash
python scripts/task2_deblur.py --input ./data/processed/test --iterations 15
```
### 3. Baseline Inference (Task 3)
Analyze the "Domain Shift" impact on off-the-shelf pre-trained models:
```bash
python scripts/task3_baseline.py --input ./outputs/restored --model yolov8n.pt
```
### 4. Model Training (Task 4)
Fine-tune YOLOv8 on the artifact-heavy dataset for 5 epochs:
```bash
python scripts/task4_pipeline.py --mode train --epochs 5 --imgsz 320
```
### 5. Performance Evaluation (Task 5)
Generate tri-domain benchmarking reports (Blurred vs. Deblurred vs. Sharp):
```bash
python scripts/task5_evaluation.py --weights models/weights/best.pt
```

## 📊 Key Metrics (Severe Blur Results)
| Model Domain           | Blur Category | mAP50  | Avg. Confidence |
|------------------------|---------------|--------|-----------------|
| **Sharp Ground Truth** | N/A           | 0.464  | 0.72            |
| **Baseline YOLOv8n**   | Severe        | 0.251  | 0.31            |
| **Retrained YOLOv8s**  | Severe        | 0.455  | 0.64            |

## 🤖 AI Attribution & Ethics
This project utilized Large Language Models (LLMs) as a pair-programmer for boilerplate generation and documentation formatting. Per Task 1, all core restoration logic and performance interpretations were verified by the author. No data was fabricated; all metrics derived from YOLOv8 logs. Detailed prompts and bias mitigation strategies are documented in `ai_log.md`.

## 📜 License
This project is developed for academic assessment purposes (COMP 6001).
