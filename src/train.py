from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix
import torch

def train_and_validate(model, train_loader, val_loader, loss_fn, optimizer, device, epochs, medical_priority_score, plot_roc_curve):
    print(f"🚀 Using device: {device}")
    if device.type == 'cuda':
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")

    best_mps = 0
    f1_list, recall_list, precision_list, medical_priority_list, roc_auc_list = [], [], [], [], []

    for epoch in range(epochs):
        print(f"\n📘 Epoch {epoch+1}")

        # Training loop
        model.train()
        train_loader_tqdm = tqdm(train_loader, desc="🔁 Training", leave=True)
        for images, labels in train_loader_tqdm:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

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
        Priority = medical_priority_score(precision, recall, roc)

        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        roc_auc_list.append(roc)
        medical_priority_list.append(Priority)

        print(f"\n📊 Epoch: {epoch+1} | Priority Medical Score: {Priority:.4f} | F1: {f1:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} | ROC AUC: {roc:.4f} | Accuracy: {acc:.4f} | Loss: {avg_loss_val:.4f}")
        print("🧾 Confusion Matrix:\n", cm)

        plot_roc_curve(y_true, y_prob, roc, epoch=epoch+1)

        if Priority > best_mps:
            best_mps = Priority
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'f1_score': f1,
                'roc_score': roc,
                'precision': precision,
                'Priority': Priority
            }, "ResNet_Model.pth")
            print("✅ Best model saved (↑ Medical Priority Score)")

    return {
        "f1": f1_list,
        "precision": precision_list,
        "recall": recall_list,
        "roc_auc": roc_auc_list,
        "medical_priority": medical_priority_list,
        "best_priority_score": best_mps
    }
