# train.py
"""
Training script for the HOI detection system using the AVA dataset.
This version supports dynamic training with resumption, checkpoint saving,
advanced visualization using TensorBoard and matplotlib, and detailed debugging.
It dynamically sets the number of classes based on the AVA dataset.
"""

import os
import glob
import yaml
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np  # For saving/loading loss history
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.training_utils import train_one_epoch, evaluate_model
# ONLY change the model import here:
from models.gnn_model import AdvancedHybridHOIGNN
from datasets.ava_dataset import AVADataset

# Set up logging.
log_filename = "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode='a')
    ]
)


def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "hoi_gnn_epoch*.pth"))
    if not checkpoint_files:
        return None, 0
    latest = max(checkpoint_files, key=lambda x: int(x.split("hoi_gnn_epoch")[-1].split(".pth")[0]))
    epoch = int(latest.split("hoi_gnn_epoch")[-1].split(".pth")[0])
    return latest, epoch


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and config["gpu"] else "cpu")

    # Create training dataset (for training we want filtering)
    train_dataset = AVADataset(
        root="datasets/ava_images",
        csv_file="datasets/ava_kinetics_v1_0/ava_train_v2.2.csv",
        ignore_availability_filter=False,
        max_classes=config.get("max_classes", 10),
        skip_missing_files=False
    )
    # For validation, bypass filtering (or use dummy samples) so that we can evaluate even if validation images are missing.
    val_dataset = AVADataset(
        root="datasets/ava_images",
        csv_file="datasets/ava_kinetics_v1_0/ava_val_v2.2.csv",
        ignore_availability_filter=True,   # bypass available video ID check
        max_classes=config.get("max_classes", 10),
        skip_missing_files=True             # return dummy samples for missing images
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    sample_data = train_dataset.get(0)
    in_channels = sample_data.x.size(1)

    num_classes = len(train_dataset.action_to_idx)
    print(f"Total number of classes (unique actions): {num_classes}")

    # ONLY change model instantiation here:
    model = AdvancedHybridHOIGNN(
        in_channels=in_channels,
        hidden_channels=config["model"]["hidden_dim"],
        out_channels=num_classes,
        num_layers=config["model"]["num_layers"],
        gnn_type=config["model"]["gnn_type"]
    ).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = GradScaler() if config["mixed_precision"] else None

    os.makedirs("checkpoints", exist_ok=True)
    start_epoch = 1
    best_val_loss = float('inf')
    checkpoint_dir = "checkpoints"
    latest_ckpt, last_epoch = get_latest_checkpoint(checkpoint_dir)
    if latest_ckpt is not None:
        logging.info(f"Resuming training from checkpoint: {latest_ckpt}")
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        start_epoch = last_epoch + 1
    else:
        logging.info("No checkpoint found. Starting training from scratch.")

    writer = SummaryWriter(log_dir="checkpoints/runs")

    # Initialize loss history lists for the current run.
    epochs_list = []
    train_losses = []
    val_losses = []

    # Load previous run's loss history if available.
    history_file = os.path.join("checkpoints", "loss_history.npz")
    if os.path.exists(history_file):
        prev_data = np.load(history_file)
        prev_epochs = prev_data["epochs"].tolist()
        prev_train_losses = prev_data["train_losses"].tolist()
        prev_val_losses = prev_data["val_losses"].tolist()
    else:
        prev_epochs, prev_train_losses, prev_val_losses = [], [], []

    # Create a professional Matplotlib figure.
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    num_epochs = config["num_epochs"]
    checkpoint_interval = config.get("checkpoint_interval", 5)
    for epoch in range(start_epoch, num_epochs + 1):
        logging.info(f"Epoch {epoch} starting...")
        train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler, epoch, scheduler)
        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        val_loss, preds, labels = evaluate_model(model, val_loader, device)
        logging.info(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Combine previous run data with current run data.
        combined_epochs = prev_epochs + epochs_list
        combined_train_losses = prev_train_losses + train_losses
        combined_val_losses = prev_val_losses + val_losses

        # Update the Matplotlib plot with enhanced aesthetics.
        ax.clear()
        ax.plot(combined_epochs, combined_train_losses, label="Train Loss", marker="o", linestyle="-", color="blue",
                markersize=6, linewidth=2)
        ax.plot(combined_epochs, combined_val_losses, label="Validation Loss", marker="s", linestyle="--", color="red",
                markersize=6, linewidth=2)
        # Set title and labels with font size and improved aesthetics
        ax.set_title("Training and Validation Loss Over Epochs", fontsize=16, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_xlim(left=0)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray',
                alpha=0.7)  # Grid lines for both major and minor ticks
        ax.minorticks_on()  # Enable minor ticks for finer grid
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray',
                alpha=0.5)  # Minor grid lines for a more refined look
        ax.legend(fontsize=12, loc='upper right')
        ax.text(0.5, 0.02, "Note: Lower loss is better", transform=ax.transAxes, fontsize=12, ha="center")
        plt.tight_layout()
        # Save the plot after the last epoch or after certain conditions
        plt.savefig(os.path.join("checkpoints", f"loss_curve_latest.png"))
        plt.show()
        plt.pause(0.01)

        if epoch % checkpoint_interval == 0:
            ckpt_path = os.path.join("checkpoints", f"hoi_gnn_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Saved checkpoint: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = os.path.join("checkpoints", "hoi_gnn_best.pth")
            torch.save(model.state_dict(), best_ckpt)
            logging.info(f"New best model saved: {best_ckpt}")

        if epoch % 5 == 0:
            logging.info(f"Epoch {epoch} sample predictions (first 5):")
            logging.info(f"Predicted: {preds[:5]}")
            logging.info(f"Actual: {labels[:5]}")

    # Save the combined loss history for future runs.
    np.savez(history_file, epochs=np.array(combined_epochs), train_losses=np.array(combined_train_losses),
             val_losses=np.array(combined_val_losses))

    plt.ioff()
    plt.show()
    writer.close()
    logging.info("Training completed.")


if __name__ == "__main__":
    main()
