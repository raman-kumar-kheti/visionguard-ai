import os
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def generate_synthetic_dataset(download_dir: str, category: str):
    print("Generating synthetic mock dataset for testing...")
    base_dir = Path(download_dir) / category
    train_dir = base_dir / "train" / "good"
    test_good_dir = base_dir / "test" / "good"
    test_anomaly_dir = base_dir / "test" / "anomaly"
    
    for d in [train_dir, test_good_dir, test_anomaly_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    def create_image(path, is_anomaly=False):
        # Create a light gray background
        img = Image.new('RGB', (224, 224), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        
        # Draw a normal object (circle)
        draw.ellipse((62, 62, 162, 162), fill=(100, 100, 100))
        
        if is_anomaly:
            # Draw a red scratch/defect
            draw.line((90, 90, 130, 130), fill=(255, 0, 0), width=5)
            
        img.save(str(path))

    for i in range(20):
        create_image(train_dir / f"{i:03d}.png")
    for i in range(5):
        create_image(test_good_dir / f"{i:03d}.png")
        create_image(test_anomaly_dir / f"{i:03d}.png", is_anomaly=True)
        
    print(f"Synthetic dataset created at {base_dir}")

def download_mvtec_category(category: str = "bottle", download_dir: str = "dataset"):
    """
    Downloads a single category from the MVTec AD dataset to save space.
    """
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    
    url = f"https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/42093737-1629951468/{category}.tar.xz"
    tar_path = os.path.join(download_dir, f"{category}.tar.xz")
    category_dir = os.path.join(download_dir, category)

    if os.path.exists(category_dir):
        print(f"Dataset category '{category}' already exists at {category_dir}")
        return

    print(f"Attempting to download {category} dataset from {url}...")
    try:
        # Need user-agent for mydrive
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(tar_path, 'wb') as out_file:
            out_file.write(response.read())
            
        print(f"Downloaded to {tar_path}. Extracting...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=download_dir)
            
        print("Extraction complete.")
        os.remove(tar_path)
    except Exception as e:
        print(f"Failed to download the dataset: {e}")
        generate_synthetic_dataset(download_dir, category)

if __name__ == "__main__":
    download_mvtec_category()

