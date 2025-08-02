import os
import shutil
import random
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
SOURCE_FOLDER = "/media/lak_05/Windows-SSD/ML/Data/Histopathology_cancer/Data/train"     # Folder with all images
CSV_PATH = "/media/lak_05/Windows-SSD/ML/Data/Histopathology_cancer/Data/train_labels.csv" # CSV with 'id' and 'label' columns
OUTPUT_FOLDER = "/media/lak_05/Windows-SSD/ML/PathoDetect/Data"                      # Destination for train/, val/, test/

# Load CSV
df = pd.read_csv(CSV_PATH)

# 🔍 Match image files with available extensions
id_to_file = {}
all_files = os.listdir(SOURCE_FOLDER)
for file in all_files:
    name, ext = os.path.splitext(file)
    id_to_file[name] = file  # e.g., "12345678": "12345678.png"

# Filter only available images
df = df[df['id'].astype(str).isin(id_to_file.keys())].reset_index(drop=True)
df['filename'] = df['id'].astype(str).map(id_to_file)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# === Split ===
n = len(df)
train_df = df.iloc[:int(0.8 * n)]
val_df   = df.iloc[int(0.8 * n):int(0.9 * n)]
test_df  = df.iloc[int(0.9 * n):]

splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df
}

# === Create folders ===
for split in splits:
    os.makedirs(os.path.join(OUTPUT_FOLDER, split), exist_ok=True)

# === Copy files and save CSVs ===
for split_name, split_df in splits.items():
    split_folder = os.path.join(OUTPUT_FOLDER, split_name)
    csv_path = os.path.join(OUTPUT_FOLDER, f"{split_name}_labels.csv")

    records = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
        src = os.path.join(SOURCE_FOLDER, row['filename'])
        dst = os.path.join(split_folder, row['filename'])
        shutil.copyfile(src, dst)
        records.append({"filename": row['filename'], "label": row['label']})

    pd.DataFrame(records).to_csv(csv_path, index=False)

print("\n✅ Done. Splits created with CSVs and images.")
