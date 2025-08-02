import os, gc, time, random
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup

from src.functions import (
    set_seed,
    train_and_validate_transformers,
    medical_priority_score
)

from src.Models import (
    HistopathologyDataset,
    CancerClassifierConvNeXTTiny,
    CancerClassifierCoaTLiteTiny,
    CancerClassifierSwinTiny,
    CancerClassifierEffNetV2S
)
from torchvision.transforms import InterpolationMode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_worker(worker_id):
    seed = 1
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def prepare_dataloaders(train_csv, val_csv, test_csv, train_dir, val_dir, test_dir, batch_size):
    train_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
    
    val_test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


    train_dataset = HistopathologyDataset(train_csv, train_dir, train_transform)
    val_dataset = HistopathologyDataset(val_csv, val_dir, val_test_transform)
    test_dataset = HistopathologyDataset(test_csv, test_dir, val_test_transform)

    g = torch.Generator().manual_seed(1)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                   num_workers=8, pin_memory=True, persistent_workers=True,
                   prefetch_factor=2, worker_init_fn=seed_worker, generator=g),

        DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                   num_workers=8, pin_memory=True, persistent_workers=True,
                   prefetch_factor=2, worker_init_fn=seed_worker, generator=g),

        DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                   num_workers=4, pin_memory=True, persistent_workers=True,
                   prefetch_factor=2, worker_init_fn=seed_worker, generator=g)
    )


def main():
    set_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    gc.collect()

    base_path = r"/media/lak_05/Windows-SSD/ML/PathoDetect/Data"
    train_csv = os.path.join(base_path, "train_labels.csv")
    val_csv = os.path.join(base_path, "val_labels.csv")
    test_csv = os.path.join(base_path, "test_labels.csv")

    train_img_dir = os.path.join(base_path, "train")
    val_img_dir = os.path.join(base_path, "val")
    test_img_dir = os.path.join(base_path, "test")

    batch_size = 64

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv, val_csv, test_csv, train_img_dir, val_img_dir, test_img_dir, batch_size)

    print("✅ Data loaders ready.")

    pos_weight = torch.tensor([130908 / 89117], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model_configs = [
        {
            "name": "SwinTiny",
            "model": CancerClassifierSwinTiny().to(device),
            "lr": 2.5e-4,
            "epochs": 60
        },
        {
            "name": "ConvNext",
            "model": CancerClassifierConvNeXTTiny().to(device),
            "lr": 3e-4,
            "epochs": 60
        },
        {
            "name": "EffNetV2S",
            "model": CancerClassifierEffNetV2S().to(device),
            "lr": 2e-4,
            "epochs": 40
        },
        {
            "name": "CoaTLite",
            "model": CancerClassifierCoaTLiteTiny().to(device),
            "lr": 2e-4,
            "epochs": 40
        }
    ]

    for cfg in model_configs:
        print(f"\n🚀 Training: {cfg['name']}")
        try:
            model = cfg["model"]
            optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.05)
            total_steps = len(train_loader) * cfg["epochs"]
            warmup_steps = int(0.1 * total_steps)

            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

            start = time.time()
            results = train_and_validate_transformers(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epochs=cfg["epochs"],
                model_name=cfg["name"],
                graph_dir=f"graphs/{cfg['name']}",
                model_dir=f"models/{cfg['name']}",
                threshold_range=(0.1, 0.95),
                beta=2,
                patience=6,
                lr_scheduler_per_batch=False,
                use_amp=True,
                medical_priority_score=medical_priority_score
            )

            # Save training metrics
            pd.DataFrame(results["history"]).to_csv(f"graphs/{cfg['name']}/metrics.csv", index=False)

            # Save best model checkpoint
            torch.save(model.state_dict(), f"models/{cfg['name']}/best_model.pth")

            print(f"✅ Saved: {cfg['name']} | Best Priority Score: {results['best_priority_score']:.4f}")
            print(f"⏱️ Time: {(time.time() - start) / 60:.2f} mins")

            del model, optimizer, scheduler, results
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"❌ Failed: {cfg['name']} | Error: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
