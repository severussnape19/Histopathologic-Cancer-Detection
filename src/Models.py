import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision import models
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights
)
from PIL import UnidentifiedImageError
import timm

# -------------------------
# 📁 Dataset
# -------------------------
class HistopathologyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.image_paths = [os.path.join(img_dir, fname) for fname in self.labels_df['filename'].values]

        if not {'filename', 'label'}.issubset(self.labels_df.columns):
            raise ValueError("CSV must contain 'filename' and 'label' columns.")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        label = int(row['label'])

        filename = str(row['filename']).strip()
        img_path = os.path.join(self.img_dir, filename)

        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError) as e:
            print(f"[⚠️ Skipping file] {img_path} -> {e}")
            # Use a black image placeholder
            image = Image.new("RGB", (96, 96))
            label = 0  # Or -1 to ignore this label later

        if self.transform:
            image = self.transform(image)

        return image, label


# -------------------------
# 🔬 Classic CNN (Toy Model)
# -------------------------
class CancerClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)  # Assumes 224x224 input
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# -------------------------
# 🏛️ ResNet Variants
# -------------------------
class CancerClassifierResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.base.fc = nn.Linear(self.base.fc.in_features, 1)

    def forward(self, x):
        return self.base(x)

class CancerClassifierResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.base.fc = nn.Linear(self.base.fc.in_features, 1)

    def forward(self, x):
        return self.base(x)

# -------------------------
# 🌱 DenseNet
# -------------------------
class CancerClassifierDenseNet121(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        self.base.classifier = nn.Linear(self.base.classifier.in_features, 1)

    def forward(self, x):
        return self.base(x)

# -------------------------
# ⚡ MobileNet
# -------------------------
class CancerClassifierMobileNetV3Large(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base = models.mobilenet_v3_large(pretrained=pretrained)
        in_features = self.base.classifier[0].in_features
        self.base.classifier = nn.Sequential(nn.Linear(in_features, 1))

    def forward(self, x):
        return self.base(x)

# -------------------------
# 🔁 EfficientNet Variants
# -------------------------
class TimmEfficientNet(nn.Module):
    def __init__(self, model_name="efficientnet_b0", pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=dropout)
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

def model_b0(): return TimmEfficientNet("efficientnet_b0")
def model_b1(): return TimmEfficientNet("efficientnet_b1")
def model_b2(): return TimmEfficientNet("efficientnet_b2")

# EfficientNetV2-S
class CancerClassifierEffNetV2S(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model("efficientnetv2_rw_s", pretrained=pretrained, num_classes=1)

    def forward(self, x):
        return self.model(x)

# -------------------------
# 🔬 Transformer-Based Models
# -------------------------
class CancerClassifierCoaTLiteTiny(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base_model = timm.create_model('coat_lite_tiny', pretrained=pretrained)
        self.base_model.reset_classifier(1)

    def forward(self, x):
        return self.base_model(x)

class CancerClassifierSwinTiny(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
        self.base_model.reset_classifier(1)

    def forward(self, x):
        return self.base_model(x)

class CancerClassifierConvNeXTTiny(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base_model = timm.create_model('convnext_tiny', pretrained=pretrained)
        self.base_model.reset_classifier(1)

    def forward(self, x):
        return self.base_model(x)

class CancerClassifierMaxViTTiny(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base_model = timm.create_model('maxvit_tiny_rw_224', pretrained=pretrained)
        self.base_model.reset_classifier(1)

    def forward(self, x):
        return self.base_model(x)

class CancerClassifierEfficientFormerL1(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base_model = timm.create_model("efficientformer_l1", pretrained=pretrained, num_classes=0)
        in_features = self.base_model.num_features
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.base_model(x)
        return self.classifier(x)

# -------------------------
# 📦 Model Exports
# -------------------------
__all__ = [
    "HistopathologyDataset",
    "CancerClassifierCNN",
    "CancerClassifierResNet18",
    "CancerClassifierResNet50",
    "CancerClassifierDenseNet121",
    "CancerClassifierMobileNetV3Large",
    "CancerClassifierEffNetV2S",
    "TimmEfficientNet",
    "model_b0", "model_b1", "model_b2",
    "CancerClassifierCoaTLiteTiny",
    "CancerClassifierSwinTiny",
    "CancerClassifierConvNeXTTiny",
    "CancerClassifierMaxViTTiny",
    "CancerClassifierEfficientFormerL1"
]
