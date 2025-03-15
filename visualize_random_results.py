import os
import cv2
import random
import torch
import yaml
import matplotlib.pyplot as plt

from models.feature_extractor import HOIFeatureExtractor
from models.gnn_model import AdvancedHybridHOIGNN
from utils.graph_utils import construct_graph
from utils.visualization import overlay_detections


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_random_image_paths(root, num_samples=8, file_exts=(".jpg", ".png")):
    files = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(file_exts)]
    return random.sample(files, min(len(files), num_samples))


def run_inference_on_image(image_path, feature_extractor, model, device):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Failed to load image: {image_path}")
        return None

    # Feature extraction
    try:
        output = feature_extractor.process_frame(image)
    except Exception as e:
        print(f"⚠️ Feature extraction failed: {e}")
        return None

    detections = output.get("detections", [])
    features = output.get("features", [])
    keypoints = output.get("keypoints", [])

    if len(detections) == 0 or len(features) == 0:
        print(f"⚠️ No detections or features in: {image_path}")
        return None

    # Build graph
    node_features, edge_index = construct_graph(detections, keypoints, features, threshold=50)
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(node_features, edge_index)
    preds = logits.argmax(dim=1).cpu().tolist()

    # Overlay results
    return overlay_detections(image, detections, interactions=[f"Interaction {p}" for p in preds])


def visualize_results():
    # Load config
    config = load_config("configs/config.yaml")
    model_cfg = config.get("model", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Using device: {device}")

    # Initialize feature extractor
    feature_extractor = HOIFeatureExtractor(device=device)

    # Initialize model with config
    model = AdvancedHybridHOIGNN(
        in_channels=512,
        hidden_channels=model_cfg.get("hidden_dim", 256),
        out_channels=10,
        num_layers=model_cfg.get("num_layers", 6),
        gnn_type=model_cfg.get("gnn_type", "GraphSAGE"),
        num_heads=model_cfg.get("num_heads", 4)
    ).to(device)

    # Load checkpoint
    checkpoint_path = os.path.join("checkpoints", "hoi_gnn_best.pth")
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("✅ Model loaded from checkpoint.")
        except RuntimeError as e:
            print("❌ Checkpoint loading failed due to architecture mismatch.")
            print(str(e))
            return
    else:
        print("❌ Checkpoint not found.")
        return

    # Get image samples
    image_root = "datasets/ava_images"
    sample_paths = load_random_image_paths(image_root, num_samples=16)

    print(f"🖼️ Running inference on {len(sample_paths)} random images...")

    results = [run_inference_on_image(p, feature_extractor, model, device) for p in sample_paths]

    # Show results
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle("Random Sample Inference Results", fontsize=18)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(results) and results[i] is not None:
            img_rgb = cv2.cvtColor(results[i], cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"Sample {i+1}", fontsize=10)
        else:
            ax.set_title("No result", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    visualize_results()
