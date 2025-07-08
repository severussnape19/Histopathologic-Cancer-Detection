import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights
)

# Dataset for Histopathology Images
class HistopathologyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        if not {'id', 'label'}.issubset(self.labels_df.columns):
            raise ValueError("CSV must contain 'id' and 'label' columns.")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_id = row['id']
        label = int(row['label'])

        img_path = os.path.join(self.img_dir, f"{img_id}.tif")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# 🧠 Basic CNN Classifier
class CancerClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 28 * 28, 128)  # assuming input 224x224
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 🔁 ResNet18 Binary Classifier
class CancerClassifierResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.base_model(x)

# 🔁 ResNet50 Variant V1 (basic head)
class CancerClassifierResNet50_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.base_model(x)

# 🔁 ResNet50 Variant V2 (dropout head)
class CancerClassifierResNet50_V2(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        return self.fc(x)

# 🔁 ResNet50 Variant V3 (stronger dropout)
class CancerClassifierResNet50_V3(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        return self.fc(x)

# Instantiate all models here if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_a = CancerClassifierCNN().to(device)
model_b = CancerClassifierResNet().to(device)
model_1 = CancerClassifierResNet50_V1().to(device)
model_2 = CancerClassifierResNet50_V2().to(device)
model_3 = CancerClassifierResNet50_V3().to(device)
