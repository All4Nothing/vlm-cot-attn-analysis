"""
Batch Analysis Example - Steps 1-2

This example demonstrates:
1. Processing all images in the data directory
2. Generating descriptions using inference.py
3. Saving lightweight results to JSON (no attention data)

The JSON file can then be manually edited to add key_objects for Step 4.
"""

import sys
import os

# Add src path to sys.path for importing analyzers
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from analyzers.batch_image_analyzer import BatchImageAnalyzer

def example_batch_description_generation():
    """
    Example of batch processing all images for description generation
    """
    print("BATCH IMAGE DESCRIPTION GENERATION")
    print("="*60)
    
    # Initialize analyzer
    analyzer = BatchImageAnalyzer()
    
    try:
        # Load model
        print("Loading LLaVA model...")
        analyzer.load_model()
        
        # Set paths
        data_dir = "/workspace/yongjoo/vlm-cot-attn-analysis/data"
        output_path = "/workspace/yongjoo/vlm-cot-attn-analysis/outputs/batch_analysis_results.json"
        
        print(f"Data directory: {data_dir}")
        print(f"Output file: {output_path}")
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            return
        
        # Custom description prompt for detailed analysis
        description_prompt = """Describe this image in detail, mentioning:
1. All visible objects and their locations
2. The overall scene and setting
3. Colors, lighting, and atmosphere
4. Any people, vehicles, or structures
5. Spatial relationships between objects"""
        
        # Process all images
        print(f"\nStarting batch processing...")
        """results = analyzer.analyzer.analyze_all_images(
            data_dir=data_dir,
            description_prompt=description_prompt,
            resume_from_existing=True  # Skip already processed images
        )"""
        results = analyzer.process_single_image( # analyzer.analyze_all_images
            image_path=data_dir + "/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657114112404.jpg",
            description_prompt=description_prompt,
        )
        if not results:
            print("No results generated")
            return
        
        if results:
            # Print summary
            analyzer.print_summary()
        else:
            print("Failed to save results")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        print(f"\nUnloading model...")
        analyzer.unload_model()
        print("Done!")

def example_resume_processing():
    """
    Example of resuming interrupted batch processing
    """
    print("RESUME BATCH PROCESSING")
    print("="*60)
    
    analyzer = BatchImageAnalyzer()
    
    try:
        # Load existing results
        existing_results_path = "/workspace/yongjoo/vlm-cot-attn-analysis/outputs/batch_analysis_results.json"
        
        if os.path.exists(existing_results_path):
            print(f"Loading existing results from: {existing_results_path}")
            analyzer.load_results_from_json(existing_results_path)
            print(f"Loaded {len(analyzer.results)} existing results")
        else:
            print("No existing results found, starting fresh")
        
        # Load model and continue processing
        analyzer.load_model()
        
        # Continue processing (will skip already processed images)
        data_dir = "/workspace/yongjoo/vlm-cot-attn-analysis/data"
        results = analyzer.analyze_all_images(data_dir, resume_from_existing=True)
        
        # Save updated results
        analyzer.save_results_to_json(existing_results_path)
        analyzer.print_summary()
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        analyzer.unload_model()

def example_custom_prompt():
    """
    Example with custom description prompt for specific analysis needs
    """
    print("CUSTOM PROMPT BATCH PROCESSING")
    print("="*60)
    
    analyzer = BatchImageAnalyzer()
    
    try:
        analyzer.load_model()
        
        # Custom prompt for autonomous driving analysis
        driving_prompt = """Analyze this street scene for autonomous driving:
1. Identify all vehicles (cars, trucks, buses, motorcycles)
2. Locate pedestrians and cyclists
3. Describe traffic signs, signals, and road markings
4. Note road conditions and lane structure
5. Identify potential hazards or obstacles
6. Describe weather and lighting conditions"""
        
        data_dir = "/workspace/yongjoo/vlm-cot-attn-analysis/data"
        output_path = "/workspace/yongjoo/vlm-cot-attn-analysis/outputs/driving_analysis_results.json"
        
        print(f"Using autonomous driving analysis prompt")
        print(f"Processing: {data_dir}")
        
        results = analyzer.analyze_all_images(
            data_dir=data_dir,
            description_prompt=driving_prompt,
            resume_from_existing=False
        )
        
        analyzer.save_results_to_json(output_path)
        analyzer.print_summary()
        
        print(f"\nSpecialized analysis completed!")
        print(f"Results saved to: {output_path}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        analyzer.unload_model()

def example_quick_test():
    """
    Quick test with just a few images
    """
    print("QUICK TEST - FIRST 3 IMAGES")
    print("="*60)
    
    analyzer = BatchImageAnalyzer()
    
    try:
        analyzer.load_model()
        
        # Get first 3 images only
        data_dir = "/workspace/yongjoo/vlm-cot-attn-analysis/data"
        all_images = analyzer.get_image_files(data_dir)
        
        if not all_images:
            print(f"No images found in {data_dir}")
            return
        
        # Process only first 3 images for quick test
        test_images = all_images[:3]
        print(f"Testing with {len(test_images)} images:")
        for img in test_images:
            print(f"   - {os.path.basename(img)}")
        
        # Manually process each image (for demonstration)
        results = []
        for i, image_path in enumerate(test_images, 1):
            print(f"\n[{i}/3] Processing {os.path.basename(image_path)}...")
            
            result = analyzer.process_single_image(
                image_path,
                "Describe this image briefly, focusing on the main objects and scene."
            )
            
            results.append(result)
            
            if 'error' not in result:
                print(f"Success: {result['description'][:60]}...")
            else:
                print(f"Error: {result['error']}")
        
        # Save test results
        analyzer.results = results
        test_output = "/workspace/yongjoo/vlm-cot-attn-analysis/outputs/quick_test_results.json"
        analyzer.save_results_to_json(test_output)
        
        print(f"\nQuick test completed!")
        print(f"Test results: {test_output}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        analyzer.unload_model()

def main():
    """Run batch analysis examples"""
    print("Batch Image Analysis Examples")
    print("="*80)
    
    # Create output directory
    os.makedirs("/workspace/yongjoo/vlm-cot-attn-analysis/outputs", exist_ok=True)
    
    try:
        # Run main batch processing
        example_batch_description_generation()
        
        print("\n" + "="*80)
        print("Additional examples available:")
        print("- example_resume_processing(): Resume interrupted processing")
        print("- example_custom_prompt(): Use custom analysis prompts")
        print("- example_quick_test(): Quick test with few images")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

if __name__ == "__main__":
    # Run main batch processing by default
    example_batch_description_generation()
    
    # Uncomment to run quick test instead
    # example_quick_test()
    
    # Uncomment to run all examples
    # main()
