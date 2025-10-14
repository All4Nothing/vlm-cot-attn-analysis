"""
Example usage of LLaVA-1.6-Vicuna-7B inference system
"""
import json
from pathlib import Path
from inference_engine import LLaVAInferenceEngine
from config import Config

def example_single_inference():
    """Example of single image inference"""
    print("="*60)
    print("EXAMPLE 1: Single Image Inference")
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
        
        # Display results
        print(f"\nResponse: {result['response']}")
        print(f"Inference time: {result['inference_time']:.2f} seconds")
        print(f"Tokens - Input: {result['input_tokens']}, Output: {result['output_tokens']}")
        
        # Display attention information
        if 'attentions' in result and result['attentions']:
            print(f"\nAttention Analysis:")
            print(f"Number of layers with attention: {len(result['attentions'])}")
            print(f"Attention data type: {type(result['attentions'])}")
            
            # Safely display attention shapes
            for i, layer_attn in enumerate(result['attentions'][:5]):  # Show first 5 layers only
                try:
                    if hasattr(layer_attn, 'shape'):
                        print(f"Layer {i}: {layer_attn.shape} (batch, heads, seq_len, seq_len)")
                    elif isinstance(layer_attn, (list, tuple)):
                        print(f"Layer {i}: {type(layer_attn)} with {len(layer_attn)} elements")
                        if len(layer_attn) > 0 and hasattr(layer_attn[0], 'shape'):
                            print(f"  First element shape: {layer_attn[0].shape}")
                    else:
                        print(f"Layer {i}: {type(layer_attn)}")
                except Exception as e:
                    print(f"Layer {i}: Error accessing - {str(e)}")
            
            if len(result['attentions']) > 5:
                print(f"... and {len(result['attentions']) - 5} more layers")
        else:
            print("\nNo attention data extracted")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Unload model
        engine.unload_model()

def example_batch_inference():
    """Example of batch inference"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Inference")
    print("="*60)
    
    # Create sample batch data with local images
    batch_data = [
        {
            "image": "/workspace/yongjoo/vlm-cot-attn-analysis/data/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657114112404.jpg",
            "question": "What type of scene is shown in this image?"
        },
        {
            "image": "/workspace/yongjoo/vlm-cot-attn-analysis/data/n015-2018-10-02-10-50-40+0800__CAM_FRONT__1538448755012460.jpg",
            "question": "Describe the driving environment in this image."
        }
    ]
    
    # Save batch data to file
    batch_file = "example_batch.json"
    with open(batch_file, 'w') as f:
        json.dump(batch_data, f, indent=2)
    
    print(f"Created batch file: {batch_file}")
    print(f"Number of image-question pairs: {len(batch_data)}")
    
    # Initialize inference engine
    engine = LLaVAInferenceEngine()
    
    try:
        # Load model
        print("Loading model...")
        engine.load_model()
        
        # Process batch
        print("Processing batch...")
        results = engine.batch_generate_responses(
            batch_data,
            system_prompt="You are a helpful assistant that analyzes images."
        )
        
        # Display results
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Response: {result['response']}")
                print(f"Time: {result['inference_time']:.2f}s")
        
        # Save results
        results_file = "example_batch_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Unload model
        engine.unload_model()
        
        # Clean up example files
        try:
            Path(batch_file).unlink()
            Path(results_file).unlink()
            print("Cleaned up example files")
        except:
            pass

def example_custom_config():
    """Example with custom configuration"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Configuration")
    print("="*60)
    
    # Create custom config
    class CustomConfig(Config):
        # Override generation parameters
        GENERATION_CONFIG = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": 0,
        }
        
        # Override image size
        IMAGE_SIZE = (224, 224)
    
    # Initialize with custom config
    engine = LLaVAInferenceEngine(CustomConfig())
    
    try:
        # Display model info
        print("Loading model with custom configuration...")
        engine.load_model()
        
        model_info = engine.get_model_info()
        print("\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Example inference with custom config
        image_path = "/workspace/yongjoo/vlm-cot-attn-analysis/data/n015-2018-10-02-10-50-40+0800__CAM_FRONT__1538448755012460.jpg"
        question = "Write a short poem about this scene."
        
        print(f"\nImage: {image_path}")
        print(f"Question: {question}")
        print("Generating creative response with custom config...")
        
        result = engine.generate_response(
            image=image_path,
            question=question
        )
        
        print(f"\nResponse: {result['response']}")
        print(f"Generation config used: {result['generation_config']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        engine.unload_model()

def example_error_handling():
    """Example of error handling"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Error Handling")
    print("="*60)
    
    engine = LLaVAInferenceEngine()
    
    # Test 1: Invalid image path
    print("Test 1: Invalid image path")
    try:
        result = engine.generate_response(
            image="nonexistent_image.jpg",
            question="What do you see?"
        )
    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}: {str(e)}")
    
    # Test 2: Model not loaded
    print("\nTest 2: Model not loaded")
    try:
        result = engine.generate_response(
            image="/workspace/yongjoo/vlm-cot-attn-analysis/data/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657114112404.jpg",
            question="What do you see?"
        )
    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}: {str(e)}")
    
    print("\nError handling examples completed.")

def main():
    """Run all examples"""
    print("LLaVA-1.6-Vicuna-7B Inference Examples")
    print("Note: These examples require internet connection for sample images")
    print("and sufficient GPU memory to load the model.")
    
    try:
        # Run examples
        example_single_inference()
        example_batch_inference()
        example_custom_config()
        example_error_handling()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

if __name__ == "__main__":
    example_single_inference()
    # main()
