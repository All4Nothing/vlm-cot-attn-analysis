"""
LLaVA-1.6-Vicuna-7B inference package

This package provides a complete inference system for LLaVA-1.6-Vicuna-7B model
with support for image processing, model loading, and flexible GPU configuration.

Example usage:
    from models.llava import LLaVAInferenceEngine
    
    engine = LLaVAInferenceEngine()
    engine.load_model()
    result = engine.generate_response("image.jpg", "What do you see?")
"""

from .inference import LLaVAInferenceEngine
from .model_loader import LLaVAModelLoader
from .image_processor import ImageProcessor
from .config import Config

def create_engine(gpu_device=None):
    """
    Create a LLaVA inference engine with optional GPU specification
    
    Args:
        gpu_device: GPU device ID (e.g., "0" or "0,1")
    
    Returns:
        LLaVAInferenceEngine instance
    """
    if gpu_device:
        Config.set_cuda_device(gpu_device)
    return LLaVAInferenceEngine()

def get_version():
    """Get package version"""
    return __version__


__all__ = [
    'LLaVAInferenceEngine',
    'LLaVAModelLoader', 
    'ImageProcessor',
    'Config',
    'create_engine',
    'get_version'
]

__version__ = '1.0.0'
__author__ = 'Assistant'
__description__ = 'LLaVA-1.6-Vicuna-7B inference system'
