# main.py
import os
import gc
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from src.functions import (
    set_seed,
    train_and_validate,
    medical_priority_score,
    plot_roc_curve
)
from src.Models import HistopathologyDataset, model_b  # Change model_b to model_1 etc. as needed


def main():
    set_seed(42)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    # Load labels
    df = pd.read_csv(r'C:\ML\PathoDetect\Data\train_labels.csv')
    print(f"🧾 Total images: {len(df)}")
    print("📊 Class Distribution:\n", df['label'].value_counts())

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    full_dataset = HistopathologyDataset(
        csv_file=r'C:\ML\PathoDetect\Data\train_labels.csv',
        img_dir=r'C:\ML\PathoDetect\Data\train',
        transform=transform
    )

    # Train/Val split
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.1,
        stratify=full_dataset.labels_df['label'],
        random_state=42
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        pin_memory=True, num_workers=8, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        pin_memory=True, num_workers=8, persistent_workers=True
    )

    # Quick sanity check
    for images, labels in tqdm([next(iter(train_loader))], desc="🔍 Sample Batch"):
        print("📦 Batch shape:", images.shape)
        print("🧷 First 5 labels:", labels[:5].tolist())

    # Model, loss, optimizer
    model = model_b.to(device)  # Use model_1, model_2 etc. to swap architectures
    pos_weight = torch.tensor([130908 / 89117], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train
    results = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=10,
        medical_priority_score=medical_priority_score,
        plot_roc_curve=plot_roc_curve
    )

    # (Optional) Save results
    pd.DataFrame(results).to_csv("graphs/v1/metrics.csv", index=False)
    print("📁 Metrics saved to graphs/v1/metrics.csv")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()