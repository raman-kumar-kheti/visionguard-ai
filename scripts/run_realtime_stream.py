import os
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.patchcore import PatchCore
from src.pipeline.visualizer import Visualizer
import torchvision.transforms as transforms

def preprocess_image(image_path, image_size):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((image_size, image_size))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(img_resized).unsqueeze(0)
    return np.array(img_resized), tensor

def main():
    print("Running Pure PyTorch Inference (Bypassing OpenVINO/OpenCV Windows DLL restrictions)...")
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    image_size = config['dataset']['image_size']
    threshold = config['inference']['threshold']
    
    print("Loading PatchCore Memory Bank...")
    model = PatchCore(backbone_name=config['model']['backbone'])
    model.load("output")
    model.feature_extractor.eval()
    
    visualizer = Visualizer(alpha=0.6)
    
    test_dir = Path("dataset/bottle/test")
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return
        
    output_dir = Path("output/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_file in test_dir.rglob("*.png"):
        print(f"Processing {img_file.name}...")
        img_np, input_tensor = preprocess_image(str(img_file), image_size)
        
        image_scores, heatmaps = model.predict(input_tensor)
        heatmap = heatmaps[0]
        
        result_np = visualizer.apply_heatmap(img_np, heatmap, threshold)
        
        out_path = output_dir / f"result_{img_file.name}"
        Image.fromarray(result_np).save(str(out_path))
        print(f"Saved inference result to {out_path}")
        
    print("Inference complete! Check the output/results folder for anomaly heatmaps.")

if __name__ == "__main__":
    main()
