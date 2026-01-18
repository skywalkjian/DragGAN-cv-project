DragGAN-CV25Fall: Enhanced Interactive Point-based Image Editing
================================================================

This repository contains the source code for the "Computer Vision (2025 Fall)" course project.
This project extends the original DragGAN by integrating advanced point tracking methods (Weighted Context-Aware Tracker) and mask preservation techniques (Loss Scheduling) to improve the stability and accuracy of interactive image editing.

Overview
--------
Our implementation enhances the original DragGAN framework with the following key features:
*   **Weighted Context-Aware Tracker (WCAT)**: A robust tracking mechanism that utilizes weighted context features to maintain accurate point correspondence during large deformations.
*   **Loss Scheduling Mask Handler**: A dynamic masking strategy that adapts the reconstruction loss weight over optimization steps, effectively preserving non-edited regions while allowing flexible deformations in the target area.
*   **Unified Experiment Framework**: A comprehensive script for batch evaluation and comparison of different tracking and masking configurations.

1. Installation
2. Interactive Editing
3. Batch Experiments
4. Project Structure
5. Supplementary Material
6. Acknowledgments

1. Installation
---------------
This project is optimized for performance using NVIDIA GPUs and CUDA. While it may run on other platforms, we strongly recommend using an NVIDIA GPU for the best experience.

### **Prerequisites**
- **NVIDIA GPU**: Required for CUDA acceleration.
- **CUDA Toolkit**: 11.7 or newer.
- **C++ Compiler**: 
  - **Linux**: `gcc` (usually pre-installed).
  - **Windows**: Visual Studio 2019 or newer with "Desktop development with C++" workload.

### **Environment Setup**
We recommend using Conda:
```bash
conda env create -f environment.yml
conda activate draggan_project
```

### **PyTorch Installation (CUDA Recommended)**
Install the version of PyTorch that matches your CUDA toolkit:
```bash
# Example for CUDA 12.1
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# Example for CUDA 11.8
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

### **Download Pre-trained Models**
```bash
python scripts/download_model.py
```
This will download checkpoints (e.g., stylegan2_lions_512_pytorch.pkl) to the `checkpoints` directory.

2. Interactive Editing
----------------------
We provide a user-friendly Gradio interface for interactive image manipulation.

Launch the visualizer:
```bash
python visualizer_drag_gradio.py
```
Open your browser and navigate to the local URL (usually `http://127.0.0.1:7860`).

### Basic Usage
1.  **Select Model**: Choose a pre-trained model (e.g., Lion, Cat, Dog) from the "Pretrained Model" dropdown.
2.  **Add Points**: Click on the image to add control points. The **Blue** point represents the handle (start) point, and the **Red** point represents the target point.
3.  **Start Dragging**: Click the "Start" button to begin the optimization process. The image will deform to move the handle points towards the target points.

### Advanced Features
Our enhanced UI offers additional controls for precise editing:

*   **Tracking Method Selection**:
    Located in the "Drag" section, use the `Tracking Method` dropdown to switch between:
    *   `Baseline`: The original point tracking algorithm.
    *   `WCAT`: The Weighted Context-Aware Tracker for improved robustness.

*   **Masking Method Selection**:
    Located in the "Mask" section, use the `Masking Method` dropdown to switch between:
    *   `Baseline`: Standard binary masking.
    *   `Loss Scheduling`: Our advanced handler that dynamically adjusts loss weights to better preserve background details.

*   **Flexible Area Editing**:
    Click the "Edit Flexible Area" button to switch to mask editing mode. Draw on the image to define the region allowed to move (the flexible area). The unmasked region will remain fixed.

3. Batch Experiments
--------------------
To evaluate the performance of different trackers and mask handlers quantitatively, we provide a unified experiment script.

Run experiments using `experiments/run_experiments.py`:

```bash
# Exp 1: Compare Baseline vs. WCAT on DragBench
python experiments/run_experiments.py --exp 1 --steps 50

# Exp 3: Evaluate Mask Stability on specific samples
python experiments/run_experiments.py --exp 3 --steps 50
```

The results, including generated images and comparison collages, will be saved in `experiments/results/`.

4. Project Structure
--------------------
```
DragGAN-CV25Fall/
├── visualizer_drag_gradio.py  # Main Gradio application
├── demo.mp4                   # Supplementary material video
├── experiments/               # Experiment scripts and data
│   └── run_experiments.py     # Unified batch evaluation script
├── viz/                       # Core visualization and logic
│   ├── trackers.py            # Tracker implementations (Baseline, WCAT)
│   ├── mask_handlers.py       # Mask handler implementations (Loss Scheduling)
│   └── renderer.py            # Rendering loop
├── checkpoints/               # Pre-trained StyleGAN2 weights
└── gradio_utils/              # UI utility functions
```

5. Supplementary Material
--------------------------
We provide a demonstration video `demo.mp4` as supplementary material. 

The video showcases the robustness of our **Weighted Context-Aware Tracker (WCAT)** compared to the baseline in challenging scenarios with complex textures.

6. Acknowledgments
------------------
This project is based on the official [DragGAN](https://github.com/XingangPan/DragGAN) repository. We acknowledge the authors for their excellent work and code release.
