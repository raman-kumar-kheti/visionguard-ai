import os
import torch
import openvino as ov
from openvino.runtime import Core

class OpenVINOEngine:
    def __init__(self, model_path: str, device: str = "CPU"):
        self.core = Core()
        self.model_path = model_path
        self.device = device
        
        if os.path.exists(model_path):
            self.model = self.core.read_model(model=self.model_path)
            self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device)
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
        else:
            self.model = None

    def export_pytorch_to_onnx(self, pt_model, dummy_input, onnx_path):
        """Exports a PyTorch feature extractor to ONNX."""
        print(f"Exporting PyTorch model to ONNX: {onnx_path}")
        torch.onnx.export(
            pt_model, 
            dummy_input, 
            onnx_path, 
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

    def convert_onnx_to_openvino(self, onnx_path, ir_path):
        """Converts ONNX to OpenVINO IR format (FP32 or FP16)."""
        print(f"Converting ONNX to OpenVINO IR: {ir_path}")
        ov_model = ov.convert_model(onnx_path)
        ov.save_model(ov_model, ir_path)
        self.model_path = ir_path
        self.model = self.core.read_model(model=self.model_path)
        self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
    def predict(self, x_numpy):
        """Run inference on OpenVINO engine."""
        res = self.compiled_model([x_numpy])[self.output_layer]
        return res
