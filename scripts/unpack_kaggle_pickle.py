#!/usr/bin/env python3
"""Unpack Kaggle preprocessed pickles (train.pickle, valid.pickle, test.pickle) into folders of images."""

import os
import pickle
import cv2
import numpy as np
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw" / "kaggle_preprocessed"
    output_dir = raw_dir / "train"

    print(f"Target extraction directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pickle_files = ["train.pickle", "valid.pickle", "test.pickle"]

    for p_file in pickle_files:
        p_path = raw_dir / p_file
        if not p_path.exists():
            print(f"Skipping {p_file} because it does not exist.")
            continue
            
        print(f"Loading {p_file}...")
        with open(p_path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        
        features = d["features"]
        labels = d["labels"]
        
        print(f"Extracting {len(features)} images from {p_file}...")
        
        # Save every 500th iteration log to not pollute stdout too much
        total = len(features)
        for idx in range(total):
            img = features[idx]
            label = int(labels[idx])
            
            # Convert RGB to BGR for OpenCV cv2.imwrite
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            class_dir = output_dir / str(label)
            class_dir.mkdir(parents=True, exist_ok=True)
            
            img_name = f"{p_file.split('.')[0]}_{idx}.png"
            cv2.imwrite(str(class_dir / img_name), img_bgr)
            
            if (idx + 1) % 5000 == 0 or (idx + 1) == total:
                print(f"  Processed {idx + 1}/{total} images...")

    print("Unpacking completed successfully!")

if __name__ == "__main__":
    main()
