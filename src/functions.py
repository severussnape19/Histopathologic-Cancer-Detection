# 🧠 Core Libraries
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 📊 Metrics
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    roc_curve
)

import os
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    accuracy_score, confusion_matrix
)


def train_and_validate(model, train_loader, val_loader, loss_fn, optimizer,
                       device, epochs, medical_priority_score, plot_roc_curve):
    print(f"🚀 Using device: {device}")
    if device.type == 'cuda':
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")

    best_mps = 0
    scaler = GradScaler()

    # Metric trackers
    f1_list, recall_list, precision_list, medical_priority_list, roc_auc_list = [], [], [], [], []

    # Folder setup
    os.makedirs("graphs/v1", exist_ok=True)
    os.makedirs("models/v1", exist_ok=True)

    for epoch in range(epochs):
        print(f"\n📘 Epoch {epoch+1}")
        model.train()
        train_loader_tqdm = tqdm(train_loader, desc="🔁 Training", leave=True)

        for images, labels in train_loader_tqdm:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            with autocast():  # Mixed precision context
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loader_tqdm.set_postfix(loss=loss.item())

        # Validation loop
        model.eval()
        total_val_loss = 0
        y_true, y_pred, y_prob = [], [], []

        val_loader_tqdm = tqdm(val_loader, desc="🧪 Validating", leave=True)
        with torch.inference_mode():
            for images, labels in val_loader_tqdm:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                total_val_loss += loss.item()
                y_true += labels.cpu().numpy().flatten().tolist()
                y_pred += preds.cpu().numpy().flatten().tolist()
                y_prob += probs.cpu().numpy().flatten().tolist()

                val_loader_tqdm.set_postfix(loss=loss.item())

        avg_loss_val = total_val_loss / len(val_loader)

        # Metrics
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        roc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        priority = medical_priority_score(precision, recall, roc)

        # Store metrics
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        roc_auc_list.append(roc)
        medical_priority_list.append(priority)

        # Logging
        print(f"\n📊 Epoch: {epoch+1} | Priority Medical Score: {priority:.4f} | "
              f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | "
              f"ROC AUC: {roc:.4f} | Accuracy: {acc:.4f} | Loss: {avg_loss_val:.4f}")
        print("🧾 Confusion Matrix:\n", cm)

        # Save ROC curve
        roc_path = f"graphs/v1/roc_epoch_{epoch+1:02d}.png"
        plot_roc_curve(y_true, y_prob, roc, epoch=epoch+1, save_path=roc_path)

        # Save best model
        if priority > best_mps:
            best_mps = priority
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'f1_score': f1,
                'roc_score': roc,
                'precision': precision,
                'Priority': priority
            }, "models/v1/ResNet_Model.pth")
            print("✅ Best model saved (↑ Medical Priority Score)")

    return {
        "f1": f1_list,
        "precision": precision_list,
        "recall": recall_list,
        "roc_auc": roc_auc_list,
        "medical_priority": medical_priority_list,
        "best_priority_score": best_mps
    }


# 🔒 Set reproducible seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 🧪 Custom metric prioritizing recall (used in medical tasks)
def medical_priority_score(precision, recall, auc, beta=2):
    weighted_f1 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-8)
    return 0.6 * weighted_f1 + 0.4 * auc

# 📈 ROC Curve Plotter
def plot_roc_curve(y_true, y_prob, auc, epoch):
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color='dodgerblue', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random", alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Epoch {epoch})")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # ✅ Always save to graphs/v1/
    save_dir = "graphs/v1"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch}_roc.png")
    plt.savefig(save_path)
    plt.close()


