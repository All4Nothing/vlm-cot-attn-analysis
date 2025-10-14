"""
Main script for LLaVA-1.6-Vicuna-7B inference
"""
import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

from inference_engine import LLaVAInferenceEngine
from config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def single_inference(args):
    """Perform single image inference"""
    try:
        # Set GPU device if specified
        if hasattr(args, 'gpu_device') and args.gpu_device:
            Config.set_cuda_device(args.gpu_device)
        
        # Initialize inference engine
        engine = LLaVAInferenceEngine()
        
        # Load model
        logger.info("Loading model...")
        engine.load_model()
        
        # Generate response
        logger.info(f"Processing image: {args.image}")
        logger.info(f"Question: {args.question}")
        
        result = engine.generate_response(
            image=args.image,
            question=args.question,
            system_prompt=args.system_prompt
        )
        
        # Print results
        print("\n" + "="*50)
        print("RESPONSE:")
        print("="*50)
        print(result["response"])
        print("\n" + "="*50)
        print("METADATA:")
        print("="*50)
        print(f"Inference time: {result['inference_time']:.2f} seconds")
        print(f"Input tokens: {result['input_tokens']}")
        print(f"Output tokens: {result['output_tokens']}")
        print(f"Total tokens: {result['total_tokens']}")
        
        # Save results if output file specified
        if args.output:
            output_data = {
                "image": str(args.image),
                "question": args.question,
                "system_prompt": args.system_prompt,
                "result": result
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {args.output}")
        
        # Unload model
        engine.unload_model()
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        sys.exit(1)

def batch_inference(args):
    """Perform batch inference from JSON file"""
    try:
        # Set GPU device if specified
        if hasattr(args, 'gpu_device') and args.gpu_device:
            Config.set_cuda_device(args.gpu_device)
        
        # Load batch data
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        if not isinstance(batch_data, list):
            raise ValueError("Batch file should contain a list of image-question pairs")
        
        # Initialize inference engine
        engine = LLaVAInferenceEngine()
        
        # Load model
        logger.info("Loading model...")
        engine.load_model()
        
        # Process batch
        logger.info(f"Processing {len(batch_data)} image-question pairs...")
        results = engine.batch_generate_responses(
            batch_data,
            system_prompt=args.system_prompt
        )
        
        # Print summary
        successful = sum(1 for r in results if "error" not in r)
        failed = len(results) - successful
        
        print(f"\nBatch processing completed:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        # Save results
        output_file = args.output or "batch_results.json"
        output_data = {
            "batch_file": str(args.batch_file),
            "system_prompt": args.system_prompt,
            "summary": {
                "total": len(results),
                "successful": successful,
                "failed": failed
            },
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_file}")
        
        # Unload model
        engine.unload_model()
        
    except Exception as e:
        logger.error(f"Error during batch inference: {str(e)}")
        sys.exit(1)

def interactive_mode(args):
    """Start interactive chat mode"""
    try:
        # Set GPU device if specified
        if hasattr(args, 'gpu_device') and args.gpu_device:
            Config.set_cuda_device(args.gpu_device)
        
        engine = LLaVAInferenceEngine()
        engine.interactive_chat()
    except Exception as e:
        logger.error(f"Error in interactive mode: {str(e)}")
        sys.exit(1)

def model_info(args):
    """Display model information"""
    try:
        # Set GPU device if specified
        if hasattr(args, 'gpu_device') and args.gpu_device:
            Config.set_cuda_device(args.gpu_device)
        
        engine = LLaVAInferenceEngine()
        engine.load_model()
        
        info = engine.get_model_info()
        
        print("\n" + "="*50)
        print("MODEL INFORMATION:")
        print("="*50)
        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        engine.unload_model()
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        sys.exit(1)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="LLaVA-1.6-Vicuna-7B Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single inference
  python main.py single --image path/to/image.jpg --question "What do you see in this image?"
  
  # Batch inference
  python main.py batch --batch-file batch_data.json --output results.json
  
  # Interactive mode
  python main.py interactiv
  
  # Model information
  python main.py info
        """
    )
    
    # Global arguments
    parser.add_argument('--gpu', '--cuda-device', dest='gpu_device', 
                       help='GPU device to use (e.g., "0" or "0,1"). Overrides CUDA_VISIBLE_DEVICES')
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Single inference
    single_parser = subparsers.add_parser('single', help='Single image inference')
    single_parser.add_argument('--image', required=True, help='Path to image file or URL')
    single_parser.add_argument('--question', required=True, help='Question about the image')
    single_parser.add_argument('--system-prompt', help='Optional system prompt')
    single_parser.add_argument('--output', help='Output JSON file path')
    
    # Batch inference
    batch_parser = subparsers.add_parser('batch', help='Batch inference from JSON file')
    batch_parser.add_argument('--batch-file', required=True, help='JSON file with image-question pairs')
    batch_parser.add_argument('--system-prompt', help='Optional system prompt')
    batch_parser.add_argument('--output', help='Output JSON file path')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Interactive chat mode')
    
    # Model info
    info_parser = subparsers.add_parser('info', help='Display model information')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        single_inference(args)
    elif args.mode == 'batch':
        batch_inference(args)
    elif args.mode == 'interactive':
        interactive_mode(args)
    elif args.mode == 'info':
        model_info(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
