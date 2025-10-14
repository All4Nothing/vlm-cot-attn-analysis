"""
Image processing utilities for LLaVA inference
"""
import cv2
import numpy as np
from PIL import Image
from typing import Union, List, Optional
import requests
from io import BytesIO
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Class to handle image preprocessing for LLaVA model"""
    
    def __init__(self, target_size: tuple = (336, 336)):
        """
        Initialize the image processor
        
        Args:
            target_size: Target size for image resizing (width, height)
        """
        self.target_size = target_size
    
    def load_image_from_path(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load image from local file path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            image = Image.open(image_path)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"Loaded image from {image_path} with size {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {str(e)}")
            raise
    
    def load_image_from_url(self, url: str, timeout: int = 10) -> Image.Image:
        """
        Load image from URL
        
        Args:
            url: URL of the image
            timeout: Request timeout in seconds
            
        Returns:
            PIL Image object
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"Loaded image from URL with size {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image from URL {url}: {str(e)}")
            raise
    
    def load_image_from_array(self, image_array: np.ndarray) -> Image.Image:
        """
        Load image from numpy array
        
        Args:
            image_array: Numpy array representing the image
            
        Returns:
            PIL Image object
        """
        try:
            # Handle different array formats
            if image_array.dtype != np.uint8:
                # Normalize to 0-255 range if needed
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            # Handle different channel orders
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 3:  # RGB or BGR
                    # Assume BGR (OpenCV format) and convert to RGB
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                elif image_array.shape[2] == 4:  # RGBA
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGB)
            
            image = Image.fromarray(image_array)
            logger.info(f"Loaded image from array with size {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image from array: {str(e)}")
            raise
    
    def resize_image(self, image: Image.Image, maintain_aspect_ratio: bool = True) -> Image.Image:
        """
        Resize image to target size
        
        Args:
            image: PIL Image object
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized PIL Image object
        """
        try:
            if maintain_aspect_ratio:
                # Calculate new size maintaining aspect ratio
                original_width, original_height = image.size
                target_width, target_height = self.target_size
                
                # Calculate scaling factor
                scale = min(target_width / original_width, target_height / original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                
                # Resize image
                resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create new image with target size and paste resized image
                final_image = Image.new('RGB', self.target_size, (255, 255, 255))
                paste_x = (target_width - new_width) // 2
                paste_y = (target_height - new_height) // 2
                final_image.paste(resized_image, (paste_x, paste_y))
                
                return final_image
            else:
                # Direct resize without maintaining aspect ratio
                return image.resize(self.target_size, Image.Resampling.LANCZOS)
                
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            raise
    
    def preprocess_image(self, 
                        image_input: Union[str, Path, Image.Image, np.ndarray],
                        resize: bool = True,
                        maintain_aspect_ratio: bool = True) -> Image.Image:
        """
        Preprocess image for LLaVA model
        
        Args:
            image_input: Image input (path, URL, PIL Image, or numpy array)
            resize: Whether to resize the image
            maintain_aspect_ratio: Whether to maintain aspect ratio when resizing
            
        Returns:
            Preprocessed PIL Image object
        """
        try:
            # Load image based on input type
            if isinstance(image_input, (str, Path)):
                if str(image_input).startswith(('http://', 'https://')):
                    image = self.load_image_from_url(str(image_input))
                else:
                    image = self.load_image_from_path(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            elif isinstance(image_input, np.ndarray):
                image = self.load_image_from_array(image_input)
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Resize if requested
            if resize:
                image = self.resize_image(image, maintain_aspect_ratio)
            
            logger.info(f"Image preprocessed successfully. Final size: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def batch_preprocess_images(self, 
                               image_inputs: List[Union[str, Path, Image.Image, np.ndarray]],
                               resize: bool = True,
                               maintain_aspect_ratio: bool = True) -> List[Image.Image]:
        """
        Preprocess multiple images
        
        Args:
            image_inputs: List of image inputs
            resize: Whether to resize the images
            maintain_aspect_ratio: Whether to maintain aspect ratio when resizing
            
        Returns:
            List of preprocessed PIL Image objects
        """
        processed_images = []
        
        for i, image_input in enumerate(image_inputs):
            try:
                processed_image = self.preprocess_image(
                    image_input, resize, maintain_aspect_ratio
                )
                processed_images.append(processed_image)
                logger.info(f"Processed image {i+1}/{len(image_inputs)}")
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {str(e)}")
                raise
        
        return processed_images
