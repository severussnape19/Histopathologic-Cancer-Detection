import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

# ---------------------
# CONFIGURATION
# ---------------------
CSV_PATH = "/media/lak_05/Windows-SSD/ML/PathoDetect/Data/train_labels.csv"  # Your full metadata CSV
IMAGE_ROOT = "/media/lak_05/Windows-SSD/ML/PathoDetect/Data/train"  # Folder where all images are stored
OUTPUT_DIR = "/media/lak_05/Windows-SSD/ML/PathoDetect/Split_data"  # Where to save train/val/test folders
SEED = 42

# ---------------------
# LOAD METADATA
# ---------------------
df = pd.read_csv(CSV_PATH)

# Optional: ensure filenames and labels exist
assert 'id' in df.columns and 'label' in df.columns

# ---------------------
# STRATIFIED SPLIT
# ---------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label'],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['label'],
    random_state=SEED
)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ---------------------
# FUNCTION TO COPY IMAGES
# ---------------------
def copy_images(df_split, split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Copying {split_name}"):
        src = os.path.join(IMAGE_ROOT, row['id'] + '.tif')
        dst = os.path.join(split_dir, row['id'])
        shutil.copyfile(src, dst)

# ---------------------
# COPY IMAGES TO SPLIT FOLDERS
# ---------------------
copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")

# ---------------------
# SAVE SPLIT CSVs
# ---------------------
train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("✅ Dataset successfully split and saved!")
