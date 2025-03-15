
# G-DetectNet: A Homegrown HOI-GNN Pipeline

**Status:** Experimental / Research Exploration

---

## Overview

This repository attempts to build a **Human-Object Interaction (HOI)** detection system by combining:
- **Feature Extraction** (via YOLOv8 for object bounding boxes, an HRNet/OpenPose placeholder for human keypoints, and a ResNet-based CNN for ROI features).
- **Graph Neural Network** (GNN) approaches to model relationships between detected entities in the scene.
- **Advanced Hybrid GNN Model** that merges a deep GNN branch with a CNN branch to better capture both global (graph-level) and local (convolutional) patterns.

It was developed as a **personal research project** to explore new ways of fusing image-based features with graph-based relational reasoning. **Please note** that this is not a production-level codebase and is *highly experimental*.

---

## Motivation

The **core idea** is to:
1. Detect humans and objects in an image (or frame).
2. Build a graph where each detection is a node, and edges are formed between detections that are close in the image space.
3. Run a **GNN** (with advanced architecture) to classify the types of interactions among these nodes.

We want a pipeline that:
- Takes a raw image or video frame.
- Extracts bounding boxes and features (via YOLOv8 and a CNN).
- Constructs a graph connecting bounding boxes based on their proximity.
- Classifies the HOI relationships (who is interacting with what, and how).

**However**, results so far show the model frequently collapses to predicting a single class for almost every sample, indicating deeper issues like:
- **Data imbalance** in the AVA dataset (some classes are overrepresented).
- **Over-smoothing** or insufficient feature richness.
- **Possible mismatch** between GNN architecture and the complexities of image data.

---

## Repository Structure

```bash
.
├── configs/                 # YAML configs (model hyperparams, dataset paths, etc.)
├── datasets/
│   ├── ava_images/          # Where extracted frames from AVA videos are stored
│   ├── ava_kinetics_v1_0/   # AVA annotation CSVs
│   ├── ...
│   └── ava_dataset.py       # Defines AVADataset
├── models/
│   ├── feature_extractor.py # YOLOv8 + ResNet-based feature extraction
│   └── gnn_model.py         # AdvancedHybridHOIGNN or other GNN-based models
├── utils/
│   ├── training_utils.py    # train_one_epoch, evaluate_model, etc.
│   ├── graph_utils.py       # construct_graph from detections
│   └── visualization.py     # overlay bounding boxes, interactions
├── checkpoints/             # Model weights are saved here
├── train.py                 # Training entry script
├── evaluate.py              # Evaluation script using real AVA data
├── inference.py             # Real-time inference (webcam or video)
├── visualize_random_results.py # Script to visualize predictions on random images
├── main.py                  # CLI to load data, train, evaluate, or run inference
└── README.md                # (This file)
```

---

## Data Structures and Flow

1. **AVA Dataset**  
   - Each row in `ava_train_v2.2.csv` or `ava_val_v2.2.csv` denotes a `(video_id, timestamp, action_id, …)`.
   - Frames are extracted and named `videoID_timestamp.jpg`.  
   - `AVADataset` loads each image, resizes, normalizes, and extracts a 512-dim feature using ResNet18.

2. **Feature Extraction**  
   - **YOLOv8** for bounding boxes (object/person).
   - **OpenPose** or **HRNet** placeholder for keypoints (currently just a dummy).
   - **ResNet** to get a high-level embedding for each bounding box region.

3. **Graph Construction**  
   - For each image, bounding boxes become nodes, edges formed if centroids of boxes are within some distance threshold.

4. **GNN Forward Pass**  
   - The advanced `AdvancedHybridHOIGNN` architecture:
     - **Deep GNN Branch:** (6-layer SAGEConv or GATConv) with residual connections, dropout, and LayerNorm.
     - **CNN Branch:** Treat node feature matrix as a “1D sequence” and apply 1D Convolutions for local smoothing.
     - **Fusion:** Concatenate GNN and CNN features → multi-head self-attention → final classification layer.

5. **Training**  
   - The pipeline calls `train_one_epoch` with cross-entropy loss.
   - Checkpoints are saved every `config["checkpoint_interval"]` epochs in `checkpoints/`.

6. **Evaluation**  
   - The script uses the same `AVADataset` but with `ignore_availability_filter=True` to skip strict image existence checks.
   - Reports validation loss, confusion matrix, classification report, etc.

---

## Running the Code

Here are some **sample commands** to run the pipeline (assuming you have the appropriate environment set up):

1. **Install Dependencies (Example):**
   ```bash
   conda create -n gdetectnet python=3.9
   conda activate gdetectnet
   pip install -r requirements.txt
   # Make sure to install PyTorch Geometric, YOLOv8, etc.
   ```

2. **Train the Model:**
   ```bash
   python main.py --mode train --config configs/config.yaml
   ```
   - This uses `train.py` behind the scenes, reading hyperparams from `configs/config.yaml`.

3. **Evaluate on Validation Set:**
   ```bash
   python main.py --mode evaluate --config configs/config.yaml
   ```
   - Or directly:
   ```bash
   python evaluate.py
   ```

4. **Run Real-Time Inference with Webcam:**
   ```bash
   python main.py --mode inference --config configs/config.yaml
   ```
   - Or directly:
   ```bash
   python inference.py
   ```

5. **Visualize TensorBoard Logs:**
   ```bash
   tensorboard --logdir=runs
   ```

6. **Visualize Random Results on Train/Val Images:**
   ```bash
   python visualize_random_results.py
   ```

---

## Current Challenges

1. **Class Collapse**  
   The model (both the simpler GNN and the advanced `AdvancedHybridHOIGNN`) often converges to predicting a single class (e.g., class 6). The confusion matrix shows nearly all samples are predicted as that class, which yields a nearly constant validation loss.

2. **Potential Reasons**  
   - **Severe Class Imbalance**: The AVA dataset can be skewed, leading the model to prefer majority classes.  
   - **Over-Smoothing**: GNNs can struggle with deep architectures on image-based data. Even with residuals, advanced techniques, the final output overfits.  
   - **Hyperparameter Mismatch**: Possibly suboptimal learning rate, dropout, or weight decay.  
   - **Insufficient Negative Samples** or incorrectly filtered dataset rows.  

3. **Model Complexity vs. Data**  
   - The advanced GNN approach might require even more data or more specialized graph construction to capture the complexities of HOI.  
   - The CNN + GNN fusion architecture is large but may still fail if the bounding box-level features do not discriminate the interactions well.

---

## Future Directions

- **Address Imbalance**  
  - Weighted cross-entropy or focal loss.  
  - Balanced sampling in mini-batches.

- **Further Architecture Tweaks**  
  - Experiment with GAT + different attention heads, or other GNN variants (like Graph Transformer networks).  
  - Tuning the multi-head self-attention fusion to better merge GNN and CNN branches.

- **Better Keypoint Integration**  
  - Right now, keypoints are placeholders. Incorporating real pose estimates might help the model differentiate actions more robustly.

- **Stronger Regularization or Data Augmentation**  
  - Could mitigate overfitting to a single class.

- **Community Contributions**  
  - If you have better ideas on fusing image features with GNN-based relational reasoning, we would **love** your help!

---

## Conclusion & Disclaimer

This project is a **home research experiment** aimed at exploring advanced HOI detection with GNNs. **It is not production-ready** and still faces significant performance issues, notably collapsing to a single predicted class. It’s possible that GNN-based HOI detection with purely bounding-box-level features is not sufficient or requires more complex architectures and significant engineering.

We hope the code might serve as a **starting point** for others to investigate new ways of combining image-based detection with graph-based reasoning. If you have any suggestions or if you’d like to contribute solutions (especially around data imbalance, advanced fusion modules, or improved bounding box detection pipelines), please **open an Issue or Pull Request**. We would be happy to collaborate and learn from your insights.

**Thank you** for your interest in G-DetectNet!

---
