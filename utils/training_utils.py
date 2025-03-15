# utils/training_utils.py
"""
Training helper functions including mixed precision training, logging, and learning rate scheduling.
This version works with PyTorch Geometric Data objects.
"""

import torch
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, optimizer, data_loader, device, scaler, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        optimizer.zero_grad()
        batch = batch.to(device)  # Batch now is a PyG Batch object
        with autocast():
            outputs = model(batch.x, batch.edge_index)
            loss = torch.nn.functional.cross_entropy(outputs, batch.y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        # Optionally log batch-level loss here
        # e.g., print(f"Epoch {epoch} Batch Loss: {loss.item():.4f}")
    if scheduler is not None:
        scheduler.step()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index)
            loss = torch.nn.functional.cross_entropy(outputs, batch.y)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    return avg_loss, all_preds, all_labels
