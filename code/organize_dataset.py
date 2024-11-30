import os
import shutil
from sklearn.model_selection import train_test_split

def reorganize_all_files(raw_dir, processed_dir, train_split=0.7, val_split=0.15, test_split=0.15):
    # Ensure splits sum to 1
    assert train_split + val_split + test_split == 1.0, "Splits must sum to 1.0"

    # Define classes and initialize file lists
    classes = ["NORMAL", "PNEUMONIA"]
    all_files = {cls: [] for cls in classes}

    # Collect all files from the raw dataset across train, val, and test folders
    for split_folder in ["train", "val", "test"]:
        for cls in classes:
            cls_dir = os.path.join(raw_dir, split_folder, cls)
            if os.path.exists(cls_dir):
                all_files[cls].extend(
                    [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
                )

    # Ensure the reordered directory structure exists
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(processed_dir, split, cls), exist_ok=True)

    # Split and distribute files into train/val/test
    for cls in classes:
        train_files, temp_files = train_test_split(all_files[cls], test_size=(val_split + test_split), random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=(test_split / (val_split + test_split)), random_state=42)

        # Move files to the reordered directory
        for file_set, split in zip([train_files, val_files, test_files], ["train", "val", "test"]):
            for file in file_set:
                dest_dir = os.path.join(processed_dir, split, cls)
                shutil.copy(file, dest_dir)

    print(f"Dataset reorganized into {processed_dir} with splits: train={train_split}, val={val_split}, test={test_split}")

# Example usage
raw_dir = "../data/raw"
processed_dir = "../data/reordered"
reorganize_all_files(raw_dir, processed_dir)
