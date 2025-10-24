"""
Inference engine for LLaVA-1.6-Vicuna-7B
"""
import torch
from PIL import Image
from typing import List, Dict, Any, Optional, Union
import logging
import time
from model_loader import LLaVAModelLoader
from image_processor import ImageProcessor
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLaVAInferenceEngine:
    """Main inference engine for LLaVA model"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the inference engine
        
        Args:
            config: Configuration object, uses default Config if None
        """
        self.config = config or Config()
        self.model_loader = LLaVAModelLoader(self.config)
        self.image_processor = ImageProcessor(self.config.IMAGE_SIZE)
        self.model = None
        self.processor = None
        self.is_loaded = False
    
    def load_model(self):
        """Load the model and processor"""
        if not self.is_loaded:
            logger.info("Loading LLaVA model...")
            self.model, self.processor = self.model_loader.load_model()
            
            # Set to evaluation mode for inference
            self.model.eval()
            logger.info("Model set to evaluation mode for inference")
            
            self.is_loaded = True
            logger.info("Model loaded successfully!")
        else:
            logger.info("Model is already loaded.")
    
    def unload_model(self):
        """Unload the model to free memory"""
        if self.is_loaded:
            self.model_loader.unload_model()
            self.model = None
            self.processor = None
            self.is_loaded = False
            logger.info("Model unloaded successfully!")
    
    def prepare_conversation(self, 
                           image: Image.Image, 
                           question: str, 
                           system_prompt: Optional[str] = None) -> str:
        """
        Prepare conversation format for LLaVA
        
        Args:
            image: PIL Image object
            question: User question about the image
            system_prompt: Optional system prompt
            
        Returns:
            Formatted conversation string
        """
        if system_prompt:
            conversation = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        else:
            conversation = ""
        
        conversation += f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
        return conversation
    
    def generate_response(self, 
                         image: Union[str, Image.Image], 
                         question: str,
                         system_prompt: Optional[str] = None,
                         generation_config: Optional[Dict[str, Any]] = None,
                         return_attentions: bool = True) -> Dict[str, Any]:
        """
        Generate response for image and question
        
        Args:
            image: Image input (path, URL, or PIL Image)
            question: Question about the image
            system_prompt: Optional system prompt
            generation_config: Optional generation configuration
            
        Returns:
            Dictionary containing response and metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        
        try:
            start_time = time.time()
            
            # Preprocess image
            logger.info("Preprocessing image...")
            processed_image = self.image_processor.preprocess_image(image)
            
            # Prepare LLaVA-Next standard prompt format
            if system_prompt:
                prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt = f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
            
            # Process inputs with proper format for LLaVA-Next
            logger.info("Processing inputs...")
            inputs = self.processor(
                text=prompt,
                images=processed_image, 
                return_tensors="pt"
            )
            
            # Move inputs to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Set generation config
            gen_config = self.config.get_generation_config()
            if generation_config:
                gen_config.update(generation_config)
            
            # Generate response
            logger.info("Generating response...")
            with torch.no_grad():
                if return_attentions:
                    outputs = self.model.generate(
                        **inputs,
                        **gen_config,
                        output_attentions=True,
                        return_dict_in_generate=True,
                        use_cache=True
                    )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        **gen_config,
                        use_cache=True
                    )

            # Process outputs and decode response
            if return_attentions:
                sequences = outputs.sequences
                attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
                scores = outputs.scores if hasattr(outputs, 'scores') else None
                
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = sequences[0][input_length:]
                response = self.processor.decode(generated_tokens, skip_special_tokens=True)
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                logger.info(f"Response generated in {inference_time:.2f} seconds")
                
                # Basic attention logging (detailed analysis moved to AttentionAnalyzer)
                if attentions:
                    logger.info(f"Extracted attention from {len(attentions)} generation steps")
                else:
                    logger.warning("No attention data extracted")
                
                return {
                    "response": response.strip(),
                    "inference_time": inference_time,
                    "input_tokens": input_length,
                    "output_tokens": len(generated_tokens),
                    "total_tokens": len(sequences[0]),
                    "generation_config": gen_config,
                    "attentions": attentions,
                    "scores": scores,
                    "input_ids": inputs["input_ids"],
                    "generated_ids": sequences
                }
            else:
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]
                response = self.processor.decode(generated_tokens, skip_special_tokens=True)
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                logger.info(f"Response generated in {inference_time:.2f} seconds")
                
                return {
                    "response": response.strip(),
                    "inference_time": inference_time,
                    "input_tokens": input_length,
                    "output_tokens": len(generated_tokens),
                    "total_tokens": len(outputs[0]),
                    "generation_config": gen_config
                }
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise
    
    def batch_generate_responses(self, 
                                image_question_pairs: List[Dict[str, Any]],
                                system_prompt: Optional[str] = None,
                                generation_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple image-question pairs
        
        Args:
            image_question_pairs: List of dictionaries with 'image' and 'question' keys
            system_prompt: Optional system prompt
            generation_config: Optional generation configuration
            
        Returns:
            List of response dictionaries
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        
        results = []
        total_pairs = len(image_question_pairs)
        
        logger.info(f"Processing {total_pairs} image-question pairs...")
        
        for i, pair in enumerate(image_question_pairs):
            try:
                logger.info(f"Processing pair {i+1}/{total_pairs}")
                
                result = self.generate_response(
                    image=pair["image"],
                    question=pair["question"],
                    system_prompt=system_prompt,
                    generation_config=generation_config
                )
                
                result["pair_index"] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing pair {i+1}: {str(e)}")
                results.append({
                    "pair_index": i,
                    "error": str(e),
                    "response": None
                })
        
        return results
    
    def interactive_chat(self):
        """
        Start an interactive chat session
        """
        if not self.is_loaded:
            self.load_model()
        
        print("LLaVA Interactive Chat")
        print("Commands:")
        print("  /load <image_path> - Load an image")
        print("  /quit - Exit the chat")
        print("  /help - Show this help message")
        print("-" * 50)
        
        current_image = None
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['/quit', '/exit']:
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == '/help':
                    print("Commands:")
                    print("  /load <image_path> - Load an image")
                    print("  /quit - Exit the chat")
                    print("  /help - Show this help message")
                    continue
                
                elif user_input.startswith('/load'):
                    try:
                        image_path = user_input.split(' ', 1)[1]
                        current_image = self.image_processor.preprocess_image(image_path)
                        print(f"Image loaded: {image_path}")
                        continue
                    except IndexError:
                        print("Please provide an image path: /load <image_path>")
                        continue
                    except Exception as e:
                        print(f"Error loading image: {str(e)}")
                        continue
                
                elif user_input.startswith('/'):
                    print("Unknown command. Type /help for available commands.")
                    continue
                
                else:
                    if current_image is None:
                        print("Please load an image first using: /load <image_path>")
                        continue
                    
                    print("Assistant: Thinking...")
                    result = self.generate_response(current_image, user_input)
                    print(f"Assistant: {result['response']}")
                    print(f"(Generated in {result['inference_time']:.2f}s)")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return self.model_loader.get_model_info()
