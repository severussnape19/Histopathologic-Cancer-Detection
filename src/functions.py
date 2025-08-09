import os, time, gc, random, csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix
)
from src.Models import (CancerClassifierResNet18, CancerClassifierResNet50, TimmEfficientNet, 
                        CancerClassifierCoaTLiteTiny, CancerClassifierSwinTiny, CancerClassifierConvNeXTTiny)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### GRAD-CAM ###


def compute_gradcam(model, input_tensor, target_layer, threshold=0.5):
    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    input_tensor = input_tensor.unsqueeze(0)
    output = model(input_tensor)
    confidence = torch.sigmoid(output).item()
    pred_label = int(confidence >= threshold)

    model.zero_grad()
    output.backward()
    
    act = activations[0].detach()
    grad = gradients[0].detach()
    pooled_grads = torch.mean(grad, dim=[0, 2, 3])
    for i in range(act.shape[1]):
        act[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(act, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= heatmap.max() + 1e-8
    heatmap = heatmap.cpu().numpy()

    fwd_handle.remove()
    bwd_handle.remove()

    return heatmap, confidence, pred_label

def plot_gradcam_with_conf(image_pil, heatmap, confidence, pred_label, true_label=None, title="GradCAM"):
    image_np = np.array(image_pil.resize((224, 224))) / 255.0
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(np.uint8(image_np * 255), 0.6, heatmap_color, 0.4, 0)

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    
    label_str = "Cancerous" if pred_label else "Non-Cancerous"
    gt_str = f"GT: {true_label} | " if true_label is not None else ""
    plt.title(f"{gt_str}Pred: {label_str} ({confidence*100:.2f}% conf)", fontsize=11)
    plt.axis("off")
    plt.show()

def run_gradcam_for_model(MODELS, model_name, CSV_PATH, IMG_DIR, seed = 123):
    config = MODELS[model_name]
    model = config["model"]()
    target_layer = config["target_layer"](model)
    threshold = config["threshold"]

    df = pd.read_csv(CSV_PATH).sample(n=18, random_state=seed)
    results = []

    for idx, row in df.iterrows():
        fname = row['filename']
        true_label = row['label']
        img_path = IMG_DIR / fname

        try:
            image_pil, tensor = load_image(img_path)

            if "CoaT" in model_name or "Swin" in model_name:
                heatmap, confidence, pred_label = compute_transformer_gradcam(
                    model, tensor, target_layer, threshold=threshold
                )
            else:
                heatmap, confidence, pred_label = compute_gradcam(
                    model, tensor, target_layer, threshold=threshold
                )

            image_np = np.array(image_pil.resize((224, 224))) / 255.0
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_TURBO)
            overlay = cv2.addWeighted(np.uint8(image_np * 255), 0.6, heatmap_color, 0.4, 0)

            results.append((overlay, confidence, pred_label, true_label))

        except Exception as e:
            print(f" Error processing {fname}: {e}")

    cols = 6
    rows = (len(results) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, (overlay, conf, pred, true) in enumerate(results):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(overlay)
        pred_str = "Cancerous" if pred else "Non-Cancerous"
        color = 'green' if pred == true else 'red'
        plt.title(f"{pred_str}\n{conf*100:.1f}% | GT: {true}", fontsize=10, color=color)
        plt.axis("off")

    plt.tight_layout()
    plt.suptitle(f"Grad-CAM for {model_name}", fontsize=16)
    plt.subplots_adjust(top=0.90)
    plt.show()

def compute_transformer_gradcam(model, input_tensor, target_layer, threshold=0.5):
    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor.unsqueeze(0)) 
    confidence = torch.sigmoid(output).item()
    pred_label = int(confidence >= threshold)

    model.zero_grad()
    output.backward()

    act = activations[0].detach()
    grad = gradients[0].detach()

    if act.ndim == 4:  # CNN like: [B, C, H, W]
        weights = grad.mean(dim=(2, 3), keepdim=True)  # /Global avg pooling
        cam = (weights * act).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
    elif act.ndim == 3:  # Transformer like: [B, L, C]
        weights = grad.mean(dim=1, keepdim=True)  # [B, 1, C]
        cam = (weights * act).sum(dim=-1)    # [B, L]
        cam = F.relu(cam).squeeze()
        cam = cam / (cam.max() + 1e-8)
        num_tokens = cam.numel()
        side = int(num_tokens ** 0.5)
        cam = cam[:side * side].reshape(side, side).cpu().numpy()
    else:
        raise ValueError(f"Unsupported activation shape: {act.shape}")

    fwd_handle.remove()
    bwd_handle.remove()

    return cam, confidence, pred_label

def overlay_heatmap(image_pil, heatmap, alpha=0.6):
    image_np = np.array(image_pil.resize((224, 224))) / 255.0
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(np.uint8(image_np * 255), alpha, heatmap_color, 1 - alpha, 0)


def compare_gradcams_on_image(filename, IMG_DIR, MODELS, df, transform):
    """
    Compares GradCAM overlays across multiple models for a given image.

    Args:
        filename (str): Image filename to compare.
        IMG_DIR (Path): Directory containing images.
        MODELS (dict): Dictionary with model config, threshold, and target layer function.
        df (pd.DataFrame): DataFrame containing 'filename' and 'label' columns.
        transform (callable): Image transform to apply (should match training).
    """
    row = df[df["filename"] == filename].iloc[0]
    true_label = row["label"]
    img_path = IMG_DIR / filename
    image_pil, tensor = load_image(img_path, transform)

    model_names = list(MODELS.keys())
    overlays = [("Original", np.array(image_pil.resize((224, 224))), None, None, None)]

    for model_name in model_names:
        config = MODELS[model_name]
        model = config["model"]()
        threshold = config["threshold"]
        target_layer = config["target_layer"](model)

        try:
            ### GradCAM: transformer or CNN #### IMP
            if any(k in model_name for k in ["CoaT", "Swin", "ConvNeXT"]):
                heatmap, confidence, pred_label = compute_transformer_gradcam(
                    model, tensor, target_layer, threshold=threshold
                )
            else:
                heatmap, confidence, pred_label = compute_gradcam(
                    model, tensor, target_layer, threshold=threshold
                )

            overlay = overlay_heatmap(image_pil, heatmap)
            overlays.append((model_name, overlay, confidence, pred_label, true_label))

        except Exception as e:
            print(f"⚠️ {model_name} failed: {e}")
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            overlays.append((model_name, blank, 0.0, None, true_label))

    # Plot
    n = len(overlays)
    plt.figure(figsize=(4 * n, 4))
    for i, (title, img, conf, pred, gt) in enumerate(overlays):
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        if conf is None:
            plt.title("Original", fontsize=12)
        else:
            pred_str = "Cancer" if pred else "Non-Cancer"
            correct = pred == gt
            color = "green" if correct else "red"
            plt.title(f"{title}\n{pred_str} ({conf * 100:.1f}%)", color=color, fontsize=12)
        plt.axis("off")

    plt.suptitle(f"GradCAM Comparison | Ground Truth: {true_label}", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()








def load_image(img_path, transform=None):
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).to(DEVICE)
    return image, tensor

def load_model(cls, weights_path):
    model = cls()
    checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    state_dict = {k.replace("base_model.", "base."): v for k, v in checkpoint["model_state"].items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

def load_effnet_model(weights_path, model = TimmEfficientNet()):
    # model = TimmEfficientNet()
    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()
    return model

def load_transformer_model(cls, weights_path):
    model = cls(pretrained=False)
    checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()
    return model

def get_image_tensor_and_label(row, transform, image_dir):
    img_path = os.path.join(image_dir, row["filename"])
    label = int(row["label"])

    img = Image.open(img_path).convert("RGB")
    image_tensor = transform(img).unsqueeze(0).to(DEVICE)

    return image_tensor, label


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
                print("Early stopping triggered.")
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














########### GRAD-CAM #########


def run_gradcam_for_model(model_name, MODELS, CSV_PATH, IMG_DIR, load_image, sample_size=18, seed=123):
    config = MODELS[model_name]
    model = config["model"]()
    model.eval()
    target_layer = config["target_layer"](model)
    threshold = config["threshold"]

    def compute_gradcam(model, input_tensor, target_layer, threshold=0.5):
        activations, gradients = [], []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        input_tensor = input_tensor.unsqueeze(0)
        output = model(input_tensor)
        confidence = torch.sigmoid(output).item()
        pred_label = int(confidence >= threshold)

        model.zero_grad()
        output.backward()
        
        act = activations[0].detach()
        grad = gradients[0].detach()
        pooled_grads = torch.mean(grad, dim=[0, 2, 3])
        for i in range(act.shape[1]):
            act[:, i, :, :] *= pooled_grads[i]

        heatmap = torch.mean(act, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= heatmap.max() + 1e-8
        heatmap = heatmap.cpu().numpy()

        fwd_handle.remove()
        bwd_handle.remove()

        return heatmap, confidence, pred_label

    def compute_transformer_gradcam(model, input_tensor, target_layer, threshold=0.5):
        activations, gradients = [], []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        output = model(input_tensor.unsqueeze(0))
        confidence = torch.sigmoid(output).item()
        pred_label = int(confidence >= threshold)

        model.zero_grad()
        output.backward()

        act = activations[0].detach()
        grad = gradients[0].detach()

        if act.ndim == 4:
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam).squeeze().cpu().numpy()
            cam = cam / (cam.max() + 1e-8)
        elif act.ndim == 3:
            weights = grad.mean(dim=1, keepdim=True)
            cam = (weights * act).sum(dim=-1)
            cam = F.relu(cam).squeeze()
            cam = cam / (cam.max() + 1e-8)
            num_tokens = cam.numel()
            side = int(num_tokens ** 0.5)
            cam = cam[:side * side].reshape(side, side).cpu().numpy()
        else:
            raise ValueError(f"Unsupported activation shape: {act.shape}")

        fwd_handle.remove()
        bwd_handle.remove()

        return cam, confidence, pred_label

    df = pd.read_csv(CSV_PATH).sample(n=sample_size, random_state=seed)
    results = []

    for idx, row in df.iterrows():
        fname = row['filename']
        true_label = row['label']
        img_path = IMG_DIR / fname

        try:
            image_pil, tensor = load_image(img_path)

            if any(k in model_name for k in ["CoaT", "Swin", "ConvNeXT"]):
                heatmap, confidence, pred_label = compute_transformer_gradcam(
                    model, tensor, target_layer, threshold=threshold
                )
            else:
                heatmap, confidence, pred_label = compute_gradcam(
                    model, tensor, target_layer, threshold=threshold
                )

            image_np = np.array(image_pil.resize((224, 224))) / 255.0
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_TURBO)
            overlay = cv2.addWeighted(np.uint8(image_np * 255), 0.6, heatmap_color, 0.4, 0)

            results.append((overlay, confidence, pred_label, true_label))

        except Exception as e:
            print(f" Error processing {fname}: {e}")

    # Plotting
    cols = 6
    rows = (len(results) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, (overlay, conf, pred, true) in enumerate(results):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(overlay)
        pred_str = "Cancerous" if pred else "Non-Cancerous"
        color = 'green' if pred == true else 'red'
        plt.title(f"{pred_str}\n{conf*100:.1f}% | GT: {true}", fontsize=10, color=color)
        plt.axis("off")

    plt.tight_layout()
    plt.suptitle(f"Grad-CAM for {model_name}", fontsize=16)
    plt.subplots_adjust(top=0.90)
    plt.show()
