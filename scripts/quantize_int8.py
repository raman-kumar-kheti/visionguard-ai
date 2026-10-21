import os
import sys
import yaml
import nncf
import openvino as ov
from openvino.runtime import Core

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.dataset_loader import get_dataloader

def transform_fn(data_item):
    """Callback function for NNCF to fetch input data from the dataloader."""
    images, _ = data_item
    return images.numpy()

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    dataset_path = config['dataset']['path']
    category = config['dataset']['category']
    image_size = config['dataset']['image_size']
    
    output_dir = "output"
    ir_path = os.path.join(output_dir, "feature_extractor.xml")
    int8_ir_path = os.path.join(output_dir, "feature_extractor_int8.xml")
    
    if not os.path.exists(ir_path):
        raise FileNotFoundError(f"Missing {ir_path}. Run train_and_export.py first.")
        
    print("Loading OpenVINO FP32 model...")
    core = Core()
    ov_model = core.read_model(ir_path)
    
    print("Preparing calibration dataset...")
    # Use a small subset of normal training data for quantization calibration
    train_loader = get_dataloader(dataset_path, category, split="train", image_size=image_size, batch_size=1)
    
    calibration_dataset = nncf.Dataset(train_loader, transform_fn)
    
    print("Starting INT8 Post-Training Quantization (PTQ)...")
    quantized_model = nncf.quantize(
        ov_model, 
        calibration_dataset,
        preset=nncf.QuantizationPreset.PERFORMANCE,
        subset_size=300, # Use up to 300 images
        fast_bias_correction=True
    )
    
    print(f"Saving quantized model to {int8_ir_path}")
    ov.save_model(quantized_model, int8_ir_path)
    print("Quantization complete!")

if __name__ == "__main__":
    main()
