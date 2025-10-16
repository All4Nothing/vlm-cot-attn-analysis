"""
Basic usage examples for LLaVA inference

This module demonstrates basic usage of the LLaVA model for inference
without advanced attention analysis.
"""

import sys
import os

# Add models path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'llava'))

from inference import LLaVAInferenceEngine
from config import Config

def example_single_inference():
    """Example of single image inference"""
    print("="*60)
    print("BASIC LLAVA INFERENCE EXAMPLE")
    print("="*60)
    
    # Initialize inference engine
    engine = LLaVAInferenceEngine()
    
    try:
        # Load model
        print("Loading model...")
        engine.load_model()
        
        # Example with a local image file
        image_path = "/workspace/yongjoo/vlm-cot-attn-analysis/data/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657114112404.jpg"
        question = "What do you see in this image? Describe the scene in detail."
        
        print(f"Image: {image_path}")
        print(f"Question: {question}")
        print("\nGenerating response...")
        
        # Generate response with attention extraction
        result = engine.generate_response(
            image=image_path,
            question=question,
            system_prompt="You are a helpful assistant that describes images in detail.",
            return_attentions=True
        )
        
        print(f"\nResponse: {result['response']}")
        print(f"Inference time: {result['inference_time']:.2f} seconds")
        print(f"Tokens - Input: {result['input_tokens']}, Output: {result['output_tokens']}")
        
        if 'attentions' in result and result['attentions']:
            print(f"\nBasic Attention Info:")
            print(f"Number of generation steps: {len(result['attentions'])}")
            print(f"Attention data type: {type(result['attentions'])}")
            print("(For detailed analysis, use the LLaVAAnalyzer from src.analyzers)")
        else:
            print("\nNo attention data extracted")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        engine.unload_model()

def example_batch_inference():
    """Example of batch inference"""
    print("="*60)
    print("BATCH INFERENCE EXAMPLE")
    print("="*60)
    
    engine = LLaVAInferenceEngine()
    
    try:
        print("Loading model...")
        engine.load_model()
        
        # Prepare batch data
        image_question_pairs = [
            {
                "image": "/workspace/yongjoo/vlm-cot-attn-analysis/data/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657114112404.jpg",
                "question": "What do you see in this image?"
            },
            {
                "image": "/workspace/yongjoo/vlm-cot-attn-analysis/data/n015-2018-10-02-10-50-40+0800__CAM_FRONT__1538448755012460.jpg",
                "question": "Describe the main objects in this scene."
            }
        ]
        
        print(f"Processing {len(image_question_pairs)} image-question pairs...")
        
        results = engine.batch_generate_responses(
            image_question_pairs=image_question_pairs,
            system_prompt="You are a helpful assistant that describes images accurately."
        )
        
        print(f"\nBatch Results:")
        for i, result in enumerate(results):
            if 'error' not in result:
                print(f"\nPair {i+1}:")
                print(f"Response: {result['response'][:100]}...")
                print(f"Time: {result['inference_time']:.2f}s")
            else:
                print(f"\nPair {i+1}: Error - {result['error']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        engine.unload_model()

def example_custom_config():
    """Example with custom configuration"""
    print("="*60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("="*60)
    
    # Create custom config
    config = Config()
    config.GENERATION_CONFIG["max_new_tokens"] = 256
    config.GENERATION_CONFIG["temperature"] = 0.1
    
    engine = LLaVAInferenceEngine(config)
    
    try:
        print("Loading model with custom config...")
        engine.load_model()
        
        image_path = "/workspace/yongjoo/vlm-cot-attn-analysis/data/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657114112404.jpg"
        question = "What is the main object in this image?"
        
        print(f"Image: {image_path}")
        print(f"Question: {question}")
        print(f"Custom config: max_tokens={config.GENERATION_CONFIG['max_new_tokens']}, temp={config.GENERATION_CONFIG['temperature']}")
        
        result = engine.generate_response(
            image=image_path,
            question=question,
            return_attentions=False  # Skip attention for faster inference
        )
        
        print(f"\nResponse: {result['response']}")
        print(f"Inference time: {result['inference_time']:.2f} seconds")
        print(f"Generated tokens: {result['output_tokens']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        engine.unload_model()

def main():
    """Run all basic examples"""
    print("LLaVA Basic Usage Examples")
    print("="*80)
    
    try:
        # Run examples
        example_single_inference()
        
        print("\n" + "="*80)
        example_batch_inference()
        
        print("\n" + "="*80)
        example_custom_config()
        
        print("\n" + "="*60)
        print("All basic examples completed!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

if __name__ == "__main__":
    # Run single inference by default
    example_single_inference()
    
    # Uncomment to run all examples
    # main()
