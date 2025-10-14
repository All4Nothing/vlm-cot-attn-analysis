"""
Configuration file for LLaVA-1.6-Vicuna-7B inference
"""
import os
from typing import Dict, Any, Optional

class Config:
    """Configuration class for LLaVA inference"""
    
    # Model configuration
    # MODEL_NAME = "liuhaotian/llava-v1.6-vicuna-7b"  # LLaVA-1.6 (NeXT) - 프로세서 파일 문제
    MODEL_NAME = "llava-hf/llava-v1.6-vicuna-7b-hf"  # LLaVA-1.6 (HF 공식 버전)
    # MODEL_NAME = "llava-hf/llava-1.5-7b-hf"  # LLaVA-1.5 (대안)
    MODEL_PATH = None  # Set to local path if model is downloaded locally
    
    # Device configuration
    DEVICE = "cuda:0"  # "auto", "cuda", "cpu", or specific GPU like "cuda:0"
    CUDA_VISIBLE_DEVICES = "0"  # Set specific GPU(s), e.g., "0" or "0,1"
    TORCH_DTYPE = "float16"  # or "bfloat16", "float32"
    
    # Generation parameters
    GENERATION_CONFIG = {
        "max_new_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": 0,
    }
    
    # Image processing
    IMAGE_SIZE = (336, 336)  # Default size for LLaVA-1.6
    
    # Cache directory
    CACHE_DIR = os.path.expanduser("~/.cache/huggingface/transformers")
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            "model_name": cls.MODEL_NAME,
            "model_path": cls.MODEL_PATH,
            "device": cls.DEVICE,
            "torch_dtype": cls.TORCH_DTYPE,
            "cache_dir": cls.CACHE_DIR,
        }
    
    @classmethod
    def get_generation_config(cls) -> Dict[str, Any]:
        """Get generation configuration dictionary"""
        return cls.GENERATION_CONFIG.copy()
    
    @classmethod
    def set_cuda_device(cls, device_id: Optional[str] = None):
        """
        Set CUDA device programmatically
        
        Args:
            device_id: GPU device ID(s) to use, e.g., "0" or "0,1"
                      If None, uses environment variable or auto-detection
        """
        if device_id is not None:
            cls.CUDA_VISIBLE_DEVICES = device_id
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            cls.DEVICE = f"cuda:{device_id.split(',')[0]}" if "," not in device_id else "cuda"
        elif cls.CUDA_VISIBLE_DEVICES is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cls.CUDA_VISIBLE_DEVICES
            cls.DEVICE = f"cuda:{cls.CUDA_VISIBLE_DEVICES.split(',')[0]}" if "," not in cls.CUDA_VISIBLE_DEVICES else "cuda"
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Get current device configuration info"""
        import torch
        
        cuda_available = torch.cuda.is_available()
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
        
        info = {
            "cuda_available": cuda_available,
            "cuda_visible_devices": cuda_visible_devices,
            "configured_device": cls.DEVICE,
            "configured_cuda_devices": cls.CUDA_VISIBLE_DEVICES,
        }
        
        if cuda_available:
            info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
                "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            })
        
        return info
