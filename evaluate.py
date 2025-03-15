import torch
import yaml
import numpy as np
from torch_geometric.data import DataLoader  # Consider replacing with loader.DataLoader in future
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_class_names(pbtxt_path):
    """
    Parses a pbtxt file (e.g., ava_action_list_v2.2.pbtxt) to extract class names.
    Returns a dictionary mapping class IDs to class names.
    """
    class_names = {}
    with open(pbtxt_path, "r") as f:
        lines = f.readlines()

    current_id = None
    current_name = None
    for line in lines:
        line = line.strip()
        if line.startswith("id:"):
            try:
                current_id = int(line.split("id:")[1].strip())
            except ValueError:
                continue
        elif line.startswith("name:") or line.startswith("display_name:"):
            current_name = line.split(":", 1)[1].strip().strip('"')
        if current_id is not None and current_name is not None:
            class_names[current_id] = current_name
            current_id, current_name = None, None
    return class_names

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                     xticklabels=classes, yticklabels=classes, cbar=True)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix\n(Misclassification counts; lower is better)", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_classification_metrics(report_dict, class_names):
    # Extract precision, recall, and f1-score for each class
    classes = list(report_dict.keys())[:-3]  # Skip 'accuracy', 'macro avg', 'weighted avg'
    precisions = [report_dict[c]["precision"] for c in classes]
    recalls = [report_dict[c]["recall"] for c in classes]
    f1_scores = [report_dict[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precisions, width, label="Precision", color="skyblue")
    plt.bar(x, recalls, width, label="Recall", color="lightgreen")
    plt.bar(x + width, f1_scores, width, label="F1-Score", color="salmon")
    plt.xticks(x, [class_names.get(int(c), f"Class {c}") for c in classes], rotation=45, fontsize=10)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Classification Metrics per Class\n(Higher is better)", fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

# Use the real validation dataset.
from datasets.ava_dataset import AVADataset
from utils.training_utils import evaluate_model
from models.gnn_model import AdvancedHybridHOIGNN

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and load checkpoint (adjust checkpoint path as needed)
    model = AdvancedHybridHOIGNN(
        in_channels=512,
        hidden_channels=config["model"]["hidden_dim"],
        out_channels=10,
        num_layers=config["model"]["num_layers"],
        gnn_type=config["model"]["gnn_type"]
    ).to(device)
    checkpoint_path = "checkpoints/hoi_gnn_best.pth"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_path} not found. Please run training first.")
        return
    model.eval()

    # Instantiate the real validation dataset.
    dataset = AVADataset(
        root="datasets/ava_images",
        csv_file="datasets/ava_kinetics_v1_0/ava_val_v2.2.csv",
        ignore_availability_filter=True,
        max_classes=config.get("max_classes", 10),
        skip_missing_files=True
    )
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    val_loss, preds, labels = evaluate_model(model, data_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")

    # Compute confusion matrix.
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)

    # Load class names from pbtxt file.
    class_names = load_class_names("datasets/ava_kinetics_v1_0/ava_action_list_v2.2.pbtxt")
    print("Confusion Matrix with Class Names:")
    for i, row in enumerate(cm):
        print(f"{class_names.get(i, f'Class {i}')}: {row}")

    # Plot the confusion matrix.
    plot_confusion_matrix(cm, classes=[class_names.get(i, f"Class {i}") for i in range(10)])

    # Compute and plot additional metrics.
    report = classification_report(labels, preds, output_dict=True)
    print("\nClassification Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    plot_classification_metrics(report, class_names)

if __name__ == "__main__":
    main()
