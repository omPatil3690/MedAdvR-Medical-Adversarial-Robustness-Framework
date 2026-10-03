"""
MedAdvR Dataset Downloader & Mock Data Generator
================================================
Utility script to either:
1. Automatically download the Brain Tumor MRI Segmentation dataset from Kaggle via kagglehub.
2. Generate a synthetic/mock tumor dataset for quick local testing without large downloads.
"""

import os
import shutil
import argparse
import numpy as np
from PIL import Image


def download_kaggle_dataset(target_dir="./data/glioma"):
    """
    Downloads the 'indk214/brain-tumor-dataset-segmentation-and-classification' dataset from Kaggle.
    """
    print("=" * 60)
    print("📥 DOWNLOADING DATASET FROM KAGGLE")
    print("=" * 60)
    
    try:
        import kagglehub
    except ImportError:
        print("[!] 'kagglehub' is not installed.")
        print("    Installing kagglehub or running via: uv pip install kagglehub")
        os.system("pip install -q kagglehub")
        import kagglehub

    print("[INFO] Fetching 'indk214/brain-tumor-dataset-segmentation-and-classification' via kagglehub...")
    download_path = kagglehub.dataset_download('indk214/brain-tumor-dataset-segmentation-and-classification')
    print(f"[INFO] Dataset downloaded to cache: {download_path}")

    # Look for DATASET/Segmentation/Glioma
    source_segmentation = os.path.join(download_path, "DATASET", "Segmentation", "Glioma")
    if not os.path.exists(source_segmentation):
        # Fallback check
        source_segmentation = download_path

    os.makedirs(target_dir, exist_ok=True)
    print(f"[INFO] Copying / linking files to: {target_dir}")
    
    # Copy or symlink files to target_dir
    file_count = 0
    for root, _, files in os.walk(source_segmentation):
        for f in files:
            if f.endswith(('.png', '.jpg', '.jpeg')):
                src_file = os.path.join(root, f)
                dst_file = os.path.join(target_dir, f)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                file_count += 1

    print(f"\n✅ [SUCCESS] {file_count} image/mask files available in: {target_dir}")
    print(f"👉 You can now run:\n   uv run main.py --dataset_path {target_dir} --dataset_type tumor")


def create_synthetic_dataset(target_dir="./data/synthetic_tumor", num_samples=30, img_size=128):
    """
    Generates a small synthetic dataset of realistic-looking brain slices and tumor masks for smoke testing.
    """
    print("=" * 60)
    print(f"🧪 GENERATING SYNTHETIC TEST DATASET ({num_samples} samples)")
    print("=" * 60)
    
    os.makedirs(target_dir, exist_ok=True)

    y, x = np.ogrid[:img_size, :img_size]
    center_y, center_x = img_size // 2, img_size // 2
    brain_radius = img_size * 0.42

    brain_mask = ((x - center_x)**2 + (y - center_y)**2) <= brain_radius**2

    for i in range(num_samples):
        # Base brain background
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Add brain tissue texture
        brain_tissue = np.random.randint(60, 140, (img_size, img_size), dtype=np.uint8)
        for c in range(3):
            img[:, :, c] = np.where(brain_mask, brain_tissue + np.random.randint(-15, 15, (img_size, img_size)), 0)

        # Random tumor blob inside brain
        tumor_radius = np.random.randint(img_size // 12, img_size // 6)
        t_cy = center_y + np.random.randint(-int(brain_radius * 0.5), int(brain_radius * 0.5))
        t_cx = center_x + np.random.randint(-int(brain_radius * 0.5), int(brain_radius * 0.5))
        
        tumor_dist = np.sqrt((x - t_cx)**2 + (y - t_cy)**2)
        tumor_mask = (tumor_dist <= tumor_radius) & brain_mask
        
        # High intensity for tumor area
        img[tumor_mask] = np.clip(img[tumor_mask] + 100, 0, 255)

        # Binary mask (0 or 255)
        mask = (tumor_mask * 255).astype(np.uint8)

        # Save image and mask pair
        img_filename = os.path.join(target_dir, f"sample_{i:04d}.png")
        mask_filename = os.path.join(target_dir, f"sample_{i:04d}_mask.png")

        Image.fromarray(img).save(img_filename)
        Image.fromarray(mask).save(mask_filename)

    print(f"✅ [SUCCESS] Generated {num_samples} synthetic MRI scans and masks in: {target_dir}")
    print(f"👉 You can immediately test training with:\n   uv run main.py --dataset_path {target_dir} --dataset_type tumor --pretrain_epochs 2 --cycles 2")


def main():
    parser = argparse.ArgumentParser(description="MedAdvR Dataset Downloader & Generator")
    parser.add_argument("--source", choices=["kaggle", "synthetic"], default="kaggle",
                        help="Choose 'kaggle' to download real dataset or 'synthetic' for fast dummy testing")
    parser.add_argument("--target_dir", type=str, default="./data/glioma",
                        help="Target output directory")
    parser.add_argument("--num_samples", type=int, default=30,
                        help="Number of synthetic samples (if source=synthetic)")
    args = parser.parse_args()

    if args.source == "kaggle":
        download_kaggle_dataset(target_dir=args.target_dir)
    else:
        create_synthetic_dataset(target_dir=args.target_dir, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
