import os
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.nn.functional import sigmoid
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import seaborn as sns

from src.Models import (
    HistopathologyDataset,
    CancerClassifierResNet18_V1,
    CancerClassifierResNet50_V1,
    CancerClassifierCoaTLiteTiny,
    model_b0
)

# === Paths ===
train_csv = "/media/lak_05/Windows-SSD/ML/PathoDetect/Data/train_labels.csv"
train_img_dir = "/media/lak_05/Windows-SSD/ML/PathoDetect/Data/train"

# === Output Folder ===
os.makedirs("outputs", exist_ok=True)

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Dataset and Loader ===
full_dataset = HistopathologyDataset(train_csv, train_img_dir, transform)
train_size = int(0.88 * len(full_dataset))
val_size = len(full_dataset) - train_size
_, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Models ===
def load_model(ModelClass, path):
    model = ModelClass().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

model1 = load_model(CancerClassifierResNet18_V1, "Final_Models/ResNet18.pth")
model2 = load_model(CancerClassifierResNet50_V1, "Final_Models/ResNet50.pth")
model3 = load_model(CancerClassifierCoaTLiteTiny, "Final_Models/CoaTLite.pth")
model4 = load_model(model_b0, "Final_Models/EfficientNetB0.pth")

models = [model1, model2, model3, model4]
thresh = [0.33, 0.41, 0.37, 0.39]
weights = [0.9672, 0.9668, 0.9738, 0.968]
total_weight = sum(weights)
norm_weights = [w / total_weight for w in weights]

# === Evaluation Loop ===
all_preds, all_labels, ensemble_confidences = [], [], []

print("\n🔍 Running ensemble predictions...\n")

for i, (img_tensor, label) in enumerate(tqdm(val_loader)):
    img_tensor = img_tensor.to(device)
    label = label.to(device)

    preds, probs = [], []

    for idx, (model, t, w) in enumerate(zip(models, thresh, norm_weights)):
        with torch.no_grad():
            prob = sigmoid(model(img_tensor)).squeeze()
        preds.append((prob >= t).float())
        probs.append(prob.item())

    # Weighted ensemble
    weighted_sum = sum([preds[i] * norm_weights[i] for i in range(4)])
    final_pred = int(weighted_sum >= 0.5)
    confidence = weighted_sum.item()

    all_preds.append(final_pred)
    all_labels.append(label.item())
    ensemble_confidences.append(confidence)


cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# === Confidence Score Distribution ===
plt.figure(figsize=(6, 4))
sns.histplot(ensemble_confidences, bins=30, kde=True)
plt.title("Distribution of Ensemble Confidence Scores")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/confidence_distribution.png")
plt.close()
