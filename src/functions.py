import os, time, gc, random, csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
import cv2
import numpy as np
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

import torch
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix
)

class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model.eval()
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[:, class_idx]
        target.backward()

        gradients = self.gradients
        activations = self.activations

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        weighted_activations = activations.squeeze(0) * pooled_gradients[:, None, None]
        cam = weighted_activations.sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_gradcam(image_tensor, cam, alpha=0.4):
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(255 * img)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    return overlay

def generate_gradcam_samples(model, test_loader, model_name, save_dir, device):
    import os
    model.eval()

    # Customize for your model structure
    if hasattr(model.model, "layer4"):
        target_layer = model.model.layer4[-1]
    else:
        target_layer = list(model.model.children())[-2]  # fallback

    cam_generator = GradCAM(model, target_layer, device)
    os.makedirs(save_dir, exist_ok=True)

    for i, (img, label) in enumerate(test_loader):
        if i >= 5: break

        cam = cam_generator.generate(img[0].to(device))
        overlay = overlay_gradcam(img[0], cam)

        out_img = ToPILImage()(torch.tensor(overlay).permute(2, 0, 1).float() / 255)
        out_img.save(os.path.join(save_dir, f"{model_name}_sample{i}.png"))

    print(f"✅ GradCAMs saved to {save_dir}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def medical_priority_score(precision, recall, beta=2):
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-8)

def plot_metrics_over_epochs(metrics_dict, save_dir="graphs/metrics"):
    os.makedirs(save_dir, exist_ok=True)
    for metric, values in metrics_dict.items():
        plt.figure()
        plt.plot(values, label=metric, color='tab:blue')
        plt.title(f"{metric.capitalize()} Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{metric}_plot.png"))
        plt.close()

def train_and_validate_transformers(
    model, train_loader, val_loader, loss_fn, optimizer,
    device, epochs, medical_priority_score,
    model_name="Model", graph_dir="graphs", model_dir="models",
    patience=5, scheduler=None, lr_scheduler_per_batch=False,
    threshold_range=(0.1, 0.95), beta=2, use_amp=True
):
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    log_path = os.path.join(graph_dir, f"{model_name}_epoch_log.csv")
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'precision', 'recall', 'f1', 'priority', 'threshold', 'val_loss', 'lr'])

    best_mps, best_smoothed_mps, wait = 0, 0, 0
    mps_window = deque(maxlen=3)
    scaler = GradScaler(enabled=use_amp)
    
    f1_list, recall_list, precision_list, medical_priority_list = [], [], [], []
    best_conf_matrix, best_threshold = None, None
    epoch_times, history = [], []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if scheduler and not lr_scheduler_per_batch:
            scheduler.step()

        model.eval()
        total_val_loss = 0
        y_true, y_prob = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                with autocast(device_type="cuda", enabled=use_amp):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)

                probs = torch.sigmoid(outputs).squeeze(1)
                total_val_loss += loss.item()
                y_true.extend(labels.squeeze(1).cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        avg_loss_val = total_val_loss / len(val_loader)
        y_true_np = np.array(y_true)
        y_prob_np = np.array(y_prob)

        best_priority = -1
        search_range = np.arange(*threshold_range, 0.01)
        for threshold in search_range:
            y_pred = (y_prob_np > threshold).astype(np.float32)
            precision = precision_score(y_true_np, y_pred, zero_division=0)
            recall = recall_score(y_true_np, y_pred, zero_division=0)
            f1 = f1_score(y_true_np, y_pred)
            priority = medical_priority_score(precision, recall, beta=beta)

            if priority > best_priority:
                best_priority = priority
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'priority': priority,
                    'accuracy': accuracy_score(y_true_np, y_pred),
                    'conf_matrix': confusion_matrix(y_true_np, y_pred)
                }

        precision_list.append(best_metrics['precision'])
        recall_list.append(best_metrics['recall'])
        f1_list.append(best_metrics['f1'])
        medical_priority_list.append(best_metrics['priority'])

        history.append({
            "epoch": epoch + 1,
            **best_metrics,
            "threshold": best_threshold,
            "loss": avg_loss_val,
            "lr": optimizer.param_groups[0]['lr']
        })

        print(
            f"📍 Epoch {epoch+1}/{epochs} | "
            f"Val Loss: {avg_loss_val:.4f} | "
            f"F1: {best_metrics['f1']:.3f} | "
            f"Precision: {best_metrics['precision']:.3f} | "
            f"Recall: {best_metrics['recall']:.3f} | "
            f"MPS: {best_metrics['priority']:.3f} | "
            f"Threshold: {best_threshold:.2f}"
        )

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                best_metrics['precision'],
                best_metrics['recall'],
                best_metrics['f1'],
                best_metrics['priority'],
                best_threshold,
                avg_loss_val,
                optimizer.param_groups[0]['lr']
            ])

        mps_window.append(best_metrics['priority'])
        smoothed_mps = np.mean(mps_window)

        if smoothed_mps > best_smoothed_mps:
            best_smoothed_mps = smoothed_mps
            best_mps = best_metrics['priority']
            best_conf_matrix = best_metrics['conf_matrix']
            wait = 0

            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'priority': best_metrics['priority'],
                'f1': best_metrics['f1'],
                'precision': best_metrics['precision'],
                'recall': best_metrics['recall'],
                'threshold': best_threshold
            }, os.path.join(model_dir, f"{model_name}_epoch{epoch+1}_priority{best_priority:.4f}.pth"))

            with open(os.path.join(model_dir, f"{model_name}_best_threshold.txt"), "w") as f:
                f.write(f"Best Threshold: {best_threshold:.4f}\n")
                f.write(f"Smoothed MPS: {smoothed_mps:.4f}\n")
        else:
            wait += 1
            if wait >= patience:
                print("⏹️ Early stopping triggered.")
                break

        epoch_times.append(time.time() - start_time)
        torch.cuda.empty_cache()
        gc.collect()

    plot_metrics_over_epochs({
        "f1": f1_list,
        "precision": precision_list,
        "recall": recall_list,
        "medical_priority": medical_priority_list
    }, save_dir=graph_dir)

    return {
        "f1": f1_list,
        "precision": precision_list,
        "recall": recall_list,
        "medical_priority": medical_priority_list,
        "best_priority_score": best_mps,
        "conf_matrix": best_conf_matrix,
        "epoch_times": epoch_times,
        "history": history
    }

