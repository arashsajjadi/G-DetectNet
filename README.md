
# G-DetectNet: A Homegrown HOI-GNN Pipeline

**Status:** Experimental / Research Project

---

## Table of Contents

1. [Introduction and Motivation](#introduction-and-motivation)
2. [Scientific and Technical Background](#scientific-and-technical-background)
3. [Project Structure](#project-structure)
4. [Data Preparation](#data-preparation)
   - [Video Downloading](#video-downloading)
   - [Frame Extraction](#frame-extraction)
   - [CSV Annotations and Data Loader](#csv-annotations-and-data-loader)
5. [Feature Extraction Stage](#feature-extraction-stage)
   - [YOLOv11 for Object Detection](#YOLOv11-for-object-detection)
   - [Keypoint Extraction (Placeholder)](#keypoint-extraction-placeholder)
   - [ResNet18 for ROI Feature Extraction](#resnet18-for-roi-feature-extraction)
6. [Graph Construction and Advanced GNN Model](#graph-construction-and-advanced-gnn-model)
   - [Graph Nodes and Edge Formation](#graph-nodes-and-edge-formation)
   - [Advanced Hybrid GNN Architecture](#advanced-hybrid-gnn-architecture)
7. [Training, Validation, and Inference](#training-validation-and-inference)
   - [Loss Functions and Optimization](#loss-functions-and-optimization)
   - [Training Procedure](#training-procedure)
   - [Validation and Evaluation](#validation-and-evaluation)
   - [Real-Time Inference](#real-time-inference)
   - [Visualization and Result Analysis](#visualization-and-result-analysis)
8. [Known Challenges and Future Improvements](#known-challenges-and-future-improvements)
9. [How to Run](#how-to-run)
10. [Debugging and Logging](#debugging-and-logging)
11. [Contributing](#contributing)
12. [License](#license)

---

## 1. Introduction and Motivation

Human-Object Interaction (HOI) detection aims to determine **how humans interact with objects** in images or video frames (e.g., "holding," "riding," "cutting"). This repository presents a pipeline that combines state-of-the-art object detection, keypoint extraction, and deep feature extraction with a Graph Neural Network (GNN) to classify interactions.

The project was developed as a personal research and experimentation effort. Although the current results show that the model sometimes collapses to predicting only one class (likely due to over-smoothing or class imbalance), it serves as a valuable starting point for further exploration.

---

## 2. Scientific and Technical Background

- **Object Detection and Keypoint Extraction:**  
  I used YOLOv11 for object detection and a placeholder for keypoint extraction (which can later be replaced by a real OpenPose or HRNet implementation).

- **Feature Extraction with Deep CNNs:**  
  A ResNet18 (with its final classification layer removed) converts cropped regions of interest (ROIs) into 512-dimensional feature vectors.

- **Graph Neural Networks (GNNs):**  
  GNNs enable modeling the relationships among detected entities by constructing graphs where nodes represent detections and edges capture spatial or semantic closeness. Our advanced hybrid model combines:
  - A **deep GNN branch** (6 layers, residual connections, dropout, and layer normalization) for global relational reasoning.
  - A **parallel CNN branch** (using 1D convolutions) that processes node features as a sequence to capture local interactions.
  - A **fusion module** employing multi-head self-attention to blend the outputs of both branches.

- **Loss Function:**  
  I used cross-entropy loss for classification, which compares the predicted class probabilities against the true class labels.

---

## 3. Project Structure

```
G-DetectNet/
├── configs/                 # YAML configuration files
├── datasets/                # Data-related files:
│   ├── ava_images/          # Extracted frames from videos
│   ├── ava_kinetics_v1_0/   # AVA CSV annotations and related files
│   ├── ...                  # Other dataset directories
├── models/
│   ├── feature_extractor.py # YOLOv11, OpenPose/HRNet (placeholder), ResNet18 extraction
│   └── gnn_model.py         # AdvancedHybridHOIGNN model architecture
├── utils/
│   ├── training_utils.py    # Training and evaluation helper functions
│   ├── graph_utils.py       # Graph construction utilities
│   └── visualization.py     # Visualization functions for results
├── checkpoints/             # Directory for saved model weights and loss history
├── train.py                 # Training script (with TensorBoard and Matplotlib plotting)
├── evaluate.py              # Evaluation script (updated to use real AVADataset)
├── inference.py             # Real-time inference script
├── visualize_random_results.py  # Visualize predictions on random train/val images
├── main.py                  # CLI to switch between modes (train, evaluate, inference)
└── README.md                # This documentation file
```

---

## 4. Data Preparation

### Video Downloading

- **Script:** `datasets/download_videos.py`  
- **Function:** Reads CSVs (e.g., `ava_train_v2.2.csv`) for YouTube video IDs and uses tools like `yt-dlp` to download videos into `datasets/ava_videos/`.

### Frame Extraction

- **Script:** `datasets/extract_frames.py`  
- **Function:** Uses OpenCV to extract frames from videos at specified timestamps and saves them as `videoID_timestamp.jpg` in `datasets/ava_images/`.

### CSV Annotations and Data Loader

- **File:** `datasets/ava_dataset.py`  
- **Function:**  
  - Reads CSV files with annotations.
  - Checks for the existence of corresponding images.
  - Filters rows based on available images and allowed action classes.
  - Maps raw action IDs to contiguous class indices.
  - Uses ResNet18 to extract 512-dimensional features from each ROI.

---

## 5. Feature Extraction Stage

### YOLOv11 for Object Detection

- **File:** `models/feature_extractor.py`  
- **Function:**  
  - Uses a pre-trained YOLOv11 model (`yolo11x.pt`) to detect bounding boxes and classify objects.
  - Outputs bounding boxes with confidence scores and class labels.

### Keypoint Extraction (Placeholder)

- **File:** `models/feature_extractor.py`  
- **Function:**  
  - Currently a placeholder (`OpenPoseExtractor`) that returns dummy keypoints.
  - Can be upgraded to use an actual keypoint detection model (e.g., OpenPose or HRNet).

### ResNet18 for ROI Feature Extraction

- **File:** `models/feature_extractor.py`  
- **Function:**  
  - Crops each detected ROI, resizes it to 224×224, and extracts a 512-dimensional feature vector using ResNet18.
  - Enhanced with dropout and layer normalization to prevent overfitting and improve generalization.

---

## 6. Graph Construction and Advanced GNN Model

### Graph Nodes and Edges

- **File:** `utils/graph_utils.py`  
- **Function:**  
  - Each detection becomes a node.
  - Edges are formed between nodes whose bounding box centroids are within a certain Euclidean distance threshold.

### Advanced Hybrid GNN Architecture

- **File:** `models/gnn_model.py`  
- **Model:** `AdvancedHybridHOIGNN`  
- **Architecture Overview:**
  - **Deep GNN Branch:**  
    - Consists of 6 layers (using GraphSAGE or GAT).
    - Uses residual connections, dropout, and layer normalization to address over-smoothing and gradient vanishing.
  - **Parallel CNN Branch:**  
    - Processes node features as a 1D sequence using multiple 1D convolutional layers.
    - Captures local interactions among nodes.
  - **Fusion Module:**  
    - Concatenates outputs from both branches.
    - Applies multi-head self-attention to blend features.
    - A final linear projection produces the output logits.
  - **Justification:**  
    This fusion of global (GNN) and local (CNN) features aims to capture both the holistic relationships in the scene and fine-grained details—essential for robust HOI detection.

---

## 7. Training, Validation, and Inference

### Loss Functions and Optimization

- **Loss:** Cross-Entropy Loss for classification.
- **Optimizer:** AdamW.
- **Learning Rate Scheduler:** StepLR (steps every 5 epochs).

### Training Procedure

- Uses `train.py` to run training:
  - Loads training and validation datasets using `AVADataset`.
  - Performs forward passes through the Advanced Hybrid GNN model.
  - Logs loss values using TensorBoard and enhanced Matplotlib plots.
  - Saves checkpoints and loss history (combined with previous runs).

### Validation Procedure

- Uses `evaluate.py` (now updated to use the real AVADataset) to compute loss, confusion matrix, and classification metrics.

### Real-Time Inference

- Uses `inference.py` for real-time (webcam) or offline video inference.
- Visualizes bounding boxes and predicted interactions overlaid on frames.

### Visualization and Result Analysis

- **TensorBoard:**  
  Launch using:
  ```bash
  tensorboard --logdir=runs
  ```
  This displays live logging of training and validation loss.
- **Matplotlib Plots:**  
  The training script saves professional loss curves (including previous runs) in the `checkpoints/` folder.  
- **Random Image Visualization:**  
  Run `python visualize_random_results.py` to see predictions on 8 random images from the training set and 8 from the validation set arranged in a 4×4 grid.

---

## 8. Known Challenges and Future Improvements

### Current Issues

- **Constant Validation Loss & Class Collapse:**  
  The model sometimes collapses to predicting only one class (e.g., class 6). This results in a nearly constant validation loss.  
  Possible reasons include:
  - **Severe Class Imbalance:** Majority classes dominate training.
  - **Over-Smoothing:** Even with residuals and dropout, the GNN branch might still over-smooth, causing loss of discriminative features.
  - **Insufficient Feature Discrimination:** Bounding box–based features might not be rich enough to capture complex interactions.

### Future Improvements

- **Enhanced Loss Functions:**  
  Implement weighted cross-entropy or focal loss to handle class imbalance.
- **Data Augmentation and Sampling Strategies:**  
  Improve training by augmenting images and balancing mini-batches.
- **Advanced Keypoint Integration:**  
  Replace the dummy keypoint extractor with a state-of-the-art method (e.g., OpenPose or HRNet).
- **GNN Architecture Tweaks:**  
  Experiment with other architectures (Graph Transformers, more attention layers) to better handle high-dimensional visual features.
- **Fusion Module Enhancements:**  
  Further refine the multi-head self-attention mechanism to better blend global and local features.

---

## 9. How to Run

### Environment Setup

1. **Create/Activate Conda Environment:**
   ```bash
   conda create -n gdetectnet python=3.9
   conda activate gdetectnet
   ```
2. **Install Dependencies:**
   ```bash
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   pip install torch-geometric ultralytics opencv-python pillow pandas matplotlib seaborn
   ```

### Commands

- **Download Videos (if needed):**
  ```bash
  python datasets/download_videos.py
  ```
- **Extract Frames:**
  ```bash
  python datasets/extract_frames.py
  ```
- **Train the Model:**
  ```bash
  python main.py --mode train --config configs/config.yaml
  ```
- **Evaluate the Model:**
  ```bash
  python main.py --mode evaluate --config configs/config.yaml
  ```
  or directly:
  ```bash
  python evaluate.py
  ```
- **Real-Time Inference:**
  ```bash
  python main.py --mode inference --config configs/config.yaml
  ```
  or directly:
  ```bash
  python inference.py
  ```
- **Visualize Random Results:**
  ```bash
  python visualize_random_results.py
  ```
- **Launch TensorBoard:**
  ```bash
  tensorboard --logdir=runs
  ```

---

## 10. Debugging and Logging

- **Logging:**  
  All training logs are saved to `training.log` and printed to the console.
- **Matplotlib Loss History:**  
  Loss curves (including previous run data) are saved as PNG files in the `checkpoints/` folder.
- **Common Issues:**  
  - Mismatches between CSV annotations and available images.
  - Warnings regarding deprecated functions or missing dependencies (e.g., torch-scatter, torch-sparse).  
  If you encounter such warnings, ensure that you have installed the latest versions of these packages and check for compatibility issues.

---

## 11. Contributing

This project is a **home research experiment**. I welcome contributions that help:
- Improve the GNN architecture (e.g., advanced fusion techniques, better handling of over-smoothing).
- Enhance feature extraction (e.g., replace dummy keypoint extraction with a real model).
- Tackle class imbalance and add new loss functions.
- Optimize data loading and graph construction methods.

If you have ideas, please feel free to open an issue or submit a pull request.

---

## 12. Conclusion and Future Work

G-DetectNet represents an exploratory effort to fuse advanced deep learning techniques with graph-based reasoning for HOI detection. While the current model has challenges—most notably, collapsing to a single class during validation—the project has laid the groundwork for:

- **Advanced Hybrid Architectures:** Combining a deep GNN branch with a parallel CNN branch using multi-head self-attention.
- **Enhanced Data Processing:** Improved filtering and feature extraction techniques.
- **Extensibility:** A modular design that allows for easy integration of new models and techniques.

Future work will focus on addressing class imbalance, refining the fusion mechanism, and incorporating more robust keypoint detection. If you have insights or improvements, your contributions will be highly appreciated.

---

## 13. License

This repository is released under the **MIT License**. See the `LICENSE` file for more details.

---

*Thank you for checking out G-DetectNet! If you have any questions or suggestions, please open an issue. I am eager to collaborate and improve this research project further.*
```

### How to Run the Updated Scripts with the New Model

The new GNN model is now called `AdvancedHybridHOIGNN`. In your `visualize_random_results.py` (and anywhere else using the model) make sure to import it as follows:

```python
from models.gnn_model import AdvancedHybridHOIGNN
```

Then run the script as usual:
```bash
python visualize_random_results.py
```

For training and evaluation, you can run:
```bash
python main.py --mode train --config configs/config.yaml
python main.py --mode evaluate --config configs/config.yaml
```

---

## Author

**Arash Sajjadi**

Thank you for exploring **G-DetectNet**! If you have any questions, suggestions, or contributions, feel free to open an issue or submit a pull request. Let's continue improving this research project together!

---

