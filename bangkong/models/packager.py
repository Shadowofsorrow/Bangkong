"""
Model packager for Bangkong LLM Training System
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Any, List, Optional
from ..config.schemas import BangkongConfig
from ..hardware.detector import HardwareDetector
from ..utils.path_manager import PathManager
from ..utils.dynamic_importer import DynamicImporter
from .quantization import quantize_model_weights


class ModelPackager:
    """Packages models in multiple formats with environment metadata."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize the model packager.
        
        Args:
            config: Bangkong configuration.
        """
        self.config = config
        self.path_manager = PathManager()
    
    def package_model(self, model: torch.nn.Module, tokenizer: Any, 
                     output_path: str, formats: Optional[List[str]] = None):
        """
        Package model in multiple formats dynamically.
        
        Args:
            model: PyTorch model to package.
            tokenizer: Tokenizer for the model.
            output_path: Path to save the packaged model.
            formats: List of formats to package the model in.
        """
        output_path = self.path_manager.resolve_path(output_path)
        
        # Apply quantization if specified in config
        quantization_precision = getattr(self.config.packaging.quantization, 'default_precision', 'none')
        if quantization_precision != 'none':
            model = quantize_model_weights(model, self.config)
            print(f"Applied {quantization_precision} quantization to model")
        
        # Save default format (PyTorch)
        self._save_pytorch_format(model, tokenizer, output_path)
        
        # Save metadata
        self._save_metadata(model, tokenizer, output_path)
        
        # Convert to additional formats if requested
        if formats:
            for fmt in formats:
                if fmt == "onnx":
                    self._save_onnx_format(model, tokenizer, output_path)
                elif fmt == "safetensors":
                    self._save_safetensors_format(model, tokenizer, output_path)
                elif fmt == "gguf":
                    self._save_gguf_format(model, tokenizer, output_path)
    
    def _save_pytorch_format(self, model: torch.nn.Module, tokenizer: Any, output_path: Path):
        """
        Save model in PyTorch format.
        
        Args:
            model: PyTorch model to save.
            tokenizer: Tokenizer for the model.
            output_path: Path to save the model.
        """
        # Save model
        model_path = output_path / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save tokenizer (if it has a save_pretrained method)
        if hasattr(tokenizer, 'save_pretrained'):
            tokenizer.save_pretrained(output_path)
        
        print(f"Model saved in PyTorch format to {output_path}")
    
    def _save_onnx_format(self, model: torch.nn.Module, tokenizer: Any, output_path: Path):
        """
        Save model in ONNX format.
        
        Args:
            model: PyTorch model to convert.
            tokenizer: Tokenizer for the model.
            output_path: Path to save the model.
        """
        onnx_module = DynamicImporter.safe_import("onnx", "ONNX")
        torch_onnx = DynamicImporter.safe_import("torch.onnx", "PyTorch ONNX export")
        if not onnx_module or not torch_onnx:
            print("ONNX or PyTorch ONNX export not available, skipping ONNX format")
            return
        
        try:
            onnx_path = output_path / "model.onnx"
            
            # Set the model to evaluation mode
            model.eval()
            
            # Create a dummy input for tracing (adjust dimensions as needed)
            dummy_input = torch.randint(
                0, 
                self.config.model.vocab_size or 50257, 
                (1, self.config.model.sequence_length or 1024)
            )
            
            # Export the model to ONNX
            torch_onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size', 1: 'sequence'},
                    'output': {0: 'batch_size', 1: 'sequence'}
                }
            )
            
            print(f"Model saved in ONNX format to {onnx_path}")
        except Exception as e:
            print(f"Failed to save model in ONNX format: {e}")
    
    def _save_safetensors_format(self, model: torch.nn.Module, tokenizer: Any, output_path: Path):
        """
        Save model in SafeTensors format.
        
        Args:
            model: PyTorch model to convert.
            tokenizer: Tokenizer for the model.
            output_path: Path to save the model.
        """
        safetensors_torch = DynamicImporter.safe_import("safetensors.torch", "SafeTensors PyTorch")
        if not safetensors_torch:
            print("SafeTensors PyTorch not available, skipping SafeTensors format")
            return
        
        try:
            safetensors_path = output_path / "model.safetensors"
            
            # Save model state dict using SafeTensors
            safetensors_torch.save_file(model.state_dict(), safetensors_path)
            
            print(f"Model saved in SafeTensors format to {safetensors_path}")
        except Exception as e:
            print(f"Failed to save model in SafeTensors format: {e}")
    
    def _save_gguf_format(self, model: torch.nn.Module, tokenizer: Any, output_path: Path):
        """
        Save model in GGUF format.
        
        Args:
            model: PyTorch model to convert.
            tokenizer: Tokenizer for the model.
            output_path: Path to save the model.
        """
        gguf_module = DynamicImporter.safe_import("gguf", "GGUF")
        if not gguf_module:
            print("GGUF not available, skipping GGUF format")
            return
        
        try:
            # Create GGUF writer
            gguf_path = output_path / "model.gguf"
            gguf_writer = gguf_module.GGUFWriter(gguf_path, "bangkong")
            
            # Add model architecture info
            gguf_writer.add_architecture()
            gguf_writer.add_name("Bangkong LLM")
            
            # Add model parameters
            state_dict = model.state_dict()
            for name, tensor in state_dict.items():
                # Convert tensor to the appropriate format for GGUF
                gguf_writer.add_tensor(name, tensor.numpy())
            
            # Add tokenizer information if available
            if tokenizer:
                gguf_writer.add_tokenizer_model("llama")
                # Add vocabulary if tokenizer has it
                if hasattr(tokenizer, 'vocab') and hasattr(tokenizer, 'get_vocab'):
                    vocab = tokenizer.get_vocab()
                    gguf_writer.add_token_list(list(vocab.keys()))
            
            # Write the GGUF file
            gguf_writer.write_header_to_file()
            gguf_writer.write_kv_data_to_file()
            gguf_writer.write_tensors_to_file()
            gguf_writer.close()
            
            print(f"Model saved in GGUF format to {gguf_path}")
        except Exception as e:
            print(f"Failed to save model in GGUF format: {e}")
    
    def _save_metadata(self, model: torch.nn.Module, tokenizer: Any, output_path: Path):
        """
        Save environment-agnostic metadata.
        
        Args:
            model: PyTorch model.
            tokenizer: Tokenizer for the model.
            output_path: Path to save the metadata.
        """
        metadata = {
            "bangkong_version": "0.1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "hardware_used": HardwareDetector.get_available_resources(),
            "config": self.config.dict(),
            "model_architecture": self.config.model.architecture,
            "model_size": self.config.model.size,
            "tokenizer_type": type(tokenizer).__name__ if tokenizer else "unknown"
        }
        
        metadata_path = output_path / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model metadata saved to {metadata_path}")