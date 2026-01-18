# DragGAN-CV25Fall: Enhanced Interactive Point-based Image Editing

This repository contains the source code for the "Computer Vision (2025 Fall)" course project. This project extends the original DragGAN by integrating advanced point tracking methods (**Weighted Context-Aware Tracker**) and mask preservation techniques (**Loss Scheduling**) to improve the stability and accuracy of interactive image editing.

## Overview

Our implementation enhances the original DragGAN framework with the following key features:

*   **Weighted Context-Aware Tracker (WCAT)**: A robust tracking mechanism that utilizes weighted context features to maintain accurate point correspondence during large deformations.
*   **Loss Scheduling Mask Handler**: A dynamic masking strategy that adapts the reconstruction loss weight over optimization steps, effectively preserving non-edited regions.
*   **Unified Experiment Framework**: A comprehensive script for batch evaluation and comparison of different tracking and masking configurations.

### Demo Video (WCAT vs Baseline)

<div align="center">
  <video src="https://github.com/skywalkjian/DragGAN-cv-project/blob/main/demo.mp4?raw=true" controls="controls" muted="muted" style="max-width: 100%;"></video>
  <p><i>WCAT demonstrates superior robustness in complex texture scenarios (e.g., Lion case) compared to the baseline.</i></p>
</div>

---

## Table of Contents
1. [Installation](#1-installation)
2. [Interactive Editing](#2-interactive-editing)
3. [Batch Experiments](#3-batch-experiments)
4. [Project Structure](#4-project-structure)
5. [Acknowledgments](#5-acknowledgments)

---

## 1. Installation

This project is optimized for performance using NVIDIA GPUs and CUDA. While it may run on other platforms, we strongly recommend using an NVIDIA GPU for the best experience.

### Prerequisites
- **NVIDIA GPU**: Required for CUDA acceleration.
- **CUDA Toolkit**: 11.7 or newer.
- **C++ Compiler**: 
  - **Linux**: `gcc` (usually pre-installed).
  - **Windows**: Visual Studio 2019 or newer with "Desktop development with C++" workload.

### Environment Setup
We recommend using Conda:
```bash
conda env create -f environment.yml
conda activate draggan_project
```

### PyTorch Installation (CUDA Recommended)
Install the version of PyTorch that matches your CUDA toolkit:
```bash
# Example for CUDA 12.1
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# Example for CUDA 11.8
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

### Download Pre-trained Models
```bash
python scripts/download_model.py
```
This will download checkpoints (e.g., `stylegan2_lions_512_pytorch.pkl`) to the `checkpoints` directory.

---

## 2. Interactive Editing

We provide a user-friendly Gradio interface for interactive image manipulation.

Launch the visualizer:
```bash
python visualizer_drag_gradio.py
```
Open your browser and navigate to the local URL (usually `http://127.0.0.1:7860`).

### Basic Usage
1.  **Select Model**: Choose a pre-trained model (e.g., Lion, Cat, Dog) from the "Pretrained Model" dropdown.
2.  **Add Points**: Click on the image to add control points. The **Blue** point represents the handle (start) point, and the **Red** point represents the target point.
3.  **Start Dragging**: Click the "Start" button to begin the optimization process.

### Advanced Features
*   **Tracking Method Selection**:
    Use the `Tracking Method` dropdown to switch between:
    *   `Baseline`: The original point tracking algorithm.
    *   `WCAT`: The Weighted Context-Aware Tracker for improved robustness.
*   **Masking Method Selection**:
    Use the `Masking Method` dropdown to switch between:
    *   `Baseline`: Standard binary masking.
    *   `Loss Scheduling`: Advanced handler that dynamically adjusts loss weights.
*   **Flexible Area Editing**:
    Click "Edit Flexible Area" to define the region allowed to move. The unmasked region remains fixed.

---

## 3. Batch Experiments

To evaluate performance quantitatively, use the unified experiment script:

```bash
# Exp 1: Compare Baseline vs. WCAT on DragBench
python experiments/run_experiments.py --exp 1 --steps 50

# Exp 3: Evaluate Mask Stability on specific samples
python experiments/run_experiments.py --exp 3 --steps 50
```

Results, including generated images and comparison collages, are saved in `experiments/results/`.

---

## 4. Project Structure

```text
DragGAN-CV25Fall/
├── visualizer_drag_gradio.py  # Main Gradio application
├── demo.mp4                   # Supplementary material video
├── calculate_fid.py           # FID evaluation utility
├── experiments/               # Experiment scripts and data
│   └── run_experiments.py     # Unified batch evaluation script
├── viz/                       # Core visualization and logic
│   ├── trackers.py            # Tracker implementations (Baseline, WCAT)
│   ├── mask_handlers.py       # Mask handler implementations (Loss Scheduling)
│   └── renderer.py            # Rendering loop
├── checkpoints/               # Pre-trained StyleGAN2 weights
└── gradio_utils/              # UI utility functions
```

---

## 5. Acknowledgments

This project is based on the official [DragGAN](https://github.com/XingangPan/DragGAN) repository. We acknowledge the authors for their excellent work and code release.
