"""
Batch Image Analyzer

Step 1-2: Generate descriptions for all images and save lightweight metadata
No attention data is stored - only descriptions and metadata for efficiency.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging
from PIL import Image
import sys
import torch

# Add models path to sys.path for importing model modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'llava'))
from inference import LLaVAInferenceEngine
from config import Config

logger = logging.getLogger(__name__)

class BatchImageAnalyzer:
    """
    Batch analyzer for generating image descriptions
    
    This class handles Step 1-2 of the workflow:
    1. Process all images in a directory
    2. Generate descriptions using inference.py
    3. Save lightweight metadata (no attention data)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the batch analyzer
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.inference_engine = LLaVAInferenceEngine(self.config)
        self.results = []
        # Prepare attention cache directory for storing large tensors
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.attention_cache_dir = os.path.join(project_root, 'outputs', 'attention_cache')
        os.makedirs(self.attention_cache_dir, exist_ok=True)
        
    def load_model(self):
        """Load the LLaVA model"""
        logger.info("Loading model for batch processing...")
        self.inference_engine.load_model()
        
    def unload_model(self):
        """Unload the model to free memory"""
        self.inference_engine.unload_model()
    
    def get_image_files(self, data_dir: str) -> List[str]:
        """
        Get all image files from the data directory
        
        Args:
            data_dir: Path to directory containing images
            
        Returns:
            List of image file paths
        """
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        try:
            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(filename.lower())
                    if ext in supported_formats:
                        image_files.append(file_path)
            
            image_files.sort()  # Sort for consistent ordering
            logger.info(f"Found {len(image_files)} image files in {data_dir}")
            return image_files
            
        except Exception as e:
            logger.error(f"Error reading directory {data_dir}: {str(e)}")
            return []
    
    def process_single_image(self, 
                           image_path: str,
                           description_prompt: str = "Describe this image in detail, mentioning all visible objects, their locations, and the overall scene.") -> Dict[str, Any]:
        """
        Process a single image to generate description
        
        Args:
            image_path: Path to the image file
            description_prompt: Prompt for description generation
            
        Returns:
            Dictionary containing image analysis results
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        try:
            logger.info(f"Processing: {os.path.basename(image_path)}")
            
            # Generate description WITH attention data for later analysis
            result = self.inference_engine.generate_response(
                image=image_path,
                question=description_prompt,
                return_attentions=True  # Key: Store attention data for Step 4
            )
            
            if 'error' in result:
                return {
                    "image_path": image_path,
                    "error": result['error'],
                    "processed_at": datetime.now().isoformat()
                }
            
            attention_cache = {
                "attentions": result.get("attentions"),
                "generated_ids": result.get("generated_ids")
            }
            cache_path = os.path.join(self.attention_cache_dir, f"{base_name}_cache.pt")
            torch.save(attention_cache, cache_path)
            print(f"attention cache saved: {cache_path}")

            result_data = {
                "metadata": {
                    "model_config": {
                        "model_name": "liuhaotian/llava-v1.6-vicuna-7b",
                        "model_path": "./models/llava/model_weights",
                        "generation_config": self.config.get_generation_config() if self.results else {}
                    },
                },

                "results": [{
                    "image_path": image_path,
                    "image_filename": os.path.basename(image_path),
                    "description": result["response"],
                    "inference_time": result.get("inference_time", 0),
                    "token_info": {
                        "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                        "total_tokens": result["total_tokens"]
                    },
                    "generation_config": result["generation_config"],
                    "attention_data": {
                        "cache_path": cache_path
                    }
                }]
            }
            
            json_path = os.path.join(self.attention_cache_dir, f"{base_name}_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=4)
            print(f"metadata saved: {json_path}")
            
            
            logger.info(f"Processed {os.path.basename(image_path)} in {result['inference_time']:.2f}s")
            return result_data
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return {
                "image_path": image_path,
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }
    
    def analyze_all_images(self, 
                          data_dir: str,
                          description_prompt: str = "Describe this image in detail, mentioning all visible objects, their locations, and the overall scene.",
                          resume_from_existing: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze all images in the data directory
        
        Args:
            data_dir: Directory containing images
            description_prompt: Prompt for description generation
            resume_from_existing: Whether to skip already processed images
            
        Returns:
            List of analysis results
        """
        try:
            # Get all image files
            image_files = self.get_image_files(data_dir)
            
            if not image_files:
                logger.warning(f"No image files found in {data_dir}")
                return []
            
            # Load existing results if resuming
            processed_images = set()
            if resume_from_existing and self.results:
                processed_images = {result.get('image_path') for result in self.results if 'error' not in result}
                logger.info(f"Resuming: {len(processed_images)} images already processed")
            
            # Process images
            total_images = len(image_files)
            new_results = []
            
            print(f"\nProcessing {total_images} images...")
            print("="*60)
            
            for i, image_path in enumerate(image_files, 1):
                # Skip if already processed
                if image_path in processed_images:
                    print(f"[{i:3d}/{total_images}] Skipping {os.path.basename(image_path)} (already processed)")
                    continue
                
                print(f"[{i:3d}/{total_images}] Processing {os.path.basename(image_path)}...")
                
                # Process image
                result = self.process_single_image(image_path, description_prompt)
                new_results.append(result)
            
            
            print("="*60)
            print(f"Batch processing completed!")
            print(f"   Total: {len(new_results)}")
            
            return new_results
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            return []
    
    def save_results_to_json(self, 
                           output_path: str,
                           include_metadata: bool = True) -> bool:
        """
        Save analysis results to JSON file
        
        Args:
            output_path: Path to save the JSON file
            include_metadata: Whether to include metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Prepare data structure
            output_data = {}
            
            if include_metadata:
                successful_results = [r for r in self.results if 'error' not in r]
                failed_results = [r for r in self.results if 'error' in r]
                
                output_data["metadata"] = {
                    "created_at": datetime.now().isoformat(),
                    "total_images": len(self.results),
                    "successful_images": len(successful_results),
                    "failed_images": len(failed_results),
                    "model_config": {
                        "model_name": "liuhaotian/llava-v1.6-vicuna-7b",
                        "model_path": "./models/llava/model_weights",
                        "generation_config": self.config.get_generation_config() if self.results else {}
                    },
                    "workflow_step": "1-2 (Description Generation)",
                    "next_step": "Add key_objects to each result, then run Step 4 (Attention Visualization)"
                }
            
            output_data["results"] = self.results
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_path}")
            print(f"Results saved to: {output_path}")
            print(f"Next step: Edit the JSON file to add 'key_objects' for each image")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False
    
    def load_results_from_json(self, input_path: str) -> bool:
        """
        Load existing results from JSON file
        
        Args:
            input_path: Path to the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.results = data.get("results", [])
            
            logger.info(f"Loaded {len(self.results)} results from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of analysis results
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.results:
            return {"error": "No results available"}
        
        successful = [r for r in self.results if 'error' not in r]
        failed = [r for r in self.results if 'error' in r]
        
        summary = {
            "total_images": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results) * 100
        }
        
        if successful:
            inference_times = [r['inference_time'] for r in successful]
            output_tokens = [r['token_info']['output_tokens'] for r in successful]
            
            summary.update({
                "avg_inference_time": sum(inference_times) / len(inference_times),
                "total_inference_time": sum(inference_times),
                "avg_output_tokens": sum(output_tokens) / len(output_tokens),
                "total_output_tokens": sum(output_tokens)
            })
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of results"""
        summary = self.get_summary()
        
        if "error" in summary:
            print(f"{summary['error']}")
            return
        
        print("\nBATCH ANALYSIS SUMMARY")
        print("="*40)
        print(f"Total Images: {summary['total_images']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['successful'] > 0:
            print(f"\nPerformance:")
            print(f"  Avg Inference Time: {summary['avg_inference_time']:.2f}s")
            print(f"  Total Processing Time: {summary['total_inference_time']:.1f}s")
            print(f"  Avg Output Tokens: {summary['avg_output_tokens']:.0f}")
            print(f"  Total Output Tokens: {summary['total_output_tokens']}")
        
        print("="*40)
