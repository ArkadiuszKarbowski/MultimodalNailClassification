import os
import sys
from typing import Any, Union, Dict, Optional
import argparse
import json
import importlib.util
import torch
import torch.nn as nn


def get_model(
    file_path: str, 
    model_name: Optional[str] = None, 
    model_args: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None
) -> Union[Any, nn.Module]:
    try:
        abs_path = os.path.abspath(file_path)
        
        if not os.path.exists(abs_path):
            print(f"Error: File '{abs_path}' does not exist")
            return None
        
        module_name = os.path.splitext(os.path.basename(abs_path))[0]
        
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None:
            print(f"Error: Could not load specification from '{abs_path}'")
            return None
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Find the model class
        model_class = None
        if model_name and hasattr(module, model_name):
            model_class = getattr(module, model_name)
        elif hasattr(module, "Model"):
            model_class = getattr(module, "Model")
        
        if model_class is None:
            print(f"Error: No model definition found in '{abs_path}'")
            return None
        
        # Instantiate the model
        model_args = model_args or {}
        
        # Handle both function-style models and class-based models
        if callable(model_class) and not isinstance(model_class, type):
            model = model_class(**model_args)
        else:
            model = model_class(**model_args)
            
        # Move model to device if specified
        if device is not None and hasattr(model, 'to'):
            model = model.to(device)
            
        return model
            
    except Exception as e:
        print(f"Error loading model from '{file_path}': {str(e)}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load model architecture from a Python file")
    parser.add_argument("file_path", help="Path to the Python file containing the model architecture")
    parser.add_argument("--model_name", help="Name of the model class/function in the file")
    parser.add_argument("--model_args", help="JSON string of arguments to pass to the model constructor")
    parser.add_argument("--device", help="Device to move the model to (e.g., 'cuda', 'cpu')")
    args = parser.parse_args()
    
    model_args = None
    if args.model_args:
        try:
            model_args = json.loads(args.model_args)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format for model_args: {args.model_args}")
            sys.exit(1)
    
    model = get_model(args.file_path, args.model_name, model_args, args.device)
    if model is not None:
        print(f"Successfully loaded model: {type(model).__name__}")