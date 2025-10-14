"""
Model loader for LLaVA-1.6-Vicuna-7B
"""
import torch
# LLaVA-1.6 with AutoProcessor (자동 감지)
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from typing import Tuple, Optional
import logging
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLaVAModelLoader:
    """Class to handle loading and managing LLaVA model and processor"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the model loader
        
        Args:
            config: Configuration object, uses default Config if None
        """
        self.config = config or Config()
        self.model = None
        self.processor = None
        self.device = None
        
    def load_model(self) -> Tuple[LlavaNextForConditionalGeneration, AutoProcessor]:
        """
        Load the LLaVA model and processor
        
        Returns:
            Tuple of (model, processor)
        """
        try:
            logger.info("Loading LLaVA-1.6-Vicuna-7B model...")
            
            # Set CUDA device if specified
            self.config.set_cuda_device()
            
            # Determine device
            if self.config.DEVICE == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.DEVICE
            
            # Log device information
            device_info = self.config.get_device_info()
            logger.info(f"Using device: {self.device}")
            logger.info(f"CUDA available: {device_info['cuda_available']}")
            if device_info['cuda_available']:
                logger.info(f"CUDA_VISIBLE_DEVICES: {device_info['cuda_visible_devices']}")
                logger.info(f"Available GPUs: {device_info['device_names']}")
            
            # Set torch dtype
            if self.config.TORCH_DTYPE == "float16":
                torch_dtype = torch.float16
            elif self.config.TORCH_DTYPE == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
            
            # Load processor with AutoProcessor
            logger.info("Loading processor with AutoProcessor...")
            self.processor = AutoProcessor.from_pretrained(
                self.config.MODEL_PATH or self.config.MODEL_NAME,
                cache_dir=self.config.CACHE_DIR,
                use_fast=True  # Fast tokenizer 사용 강제
            )
            
            # 실제 로드된 프로세서 타입 확인
            logger.info(f"Loaded processor type: {type(self.processor).__name__}")
            logger.info(f"Processor class: {self.processor.__class__}")
            
            # 프로세서의 구성 요소 확인
            if hasattr(self.processor, 'image_processor'):
                logger.info(f"Image processor: {type(self.processor.image_processor).__name__}")
            if hasattr(self.processor, 'tokenizer'):
                logger.info(f"Tokenizer: {type(self.processor.tokenizer).__name__}")
            
            # Load model
            logger.info("Loading model...")
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.config.MODEL_PATH or self.config.MODEL_NAME,
                torch_dtype=torch_dtype,
                device_map=self.device if self.device != "cpu" else None,
                cache_dir=self.config.CACHE_DIR,
                low_cpu_mem_usage=True,
                attn_implementation="eager"  # Enable attention extraction
            )
            
            # Move to device if CPU
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Model and processor loaded successfully!")
            return self.model, self.processor
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self.config.MODEL_NAME,
            "device": self.device,
            "dtype": str(self.model.dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def unload_model(self):
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded successfully!")
