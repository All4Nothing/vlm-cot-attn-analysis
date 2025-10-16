#!/usr/bin/env python3
"""
Analyze the structure of model.generate() outputs
"""

# Import path setup is handled by __init__.py
import torch
from inference import LLaVAInferenceEngine
from config import Config
import logging
from PIL import Image
import numpy as np

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_outputs_structure():
    """Analyze the structure of model.generate() outputs"""
    
    print("="*80)
    print("MODEL.GENERATE() OUTPUTS STRUCTURE ANALYSIS")
    print("="*80)
    
    # Initialize the engine
    engine = LLaVAInferenceEngine()
    
    try:
        # Load the model
        print("\nLoading model...")
        engine.load_model()
        
        # Create a test image (if there is no actual image)
        print("\nCreating test image...")
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Test question
        question = "What do you see in this image?"
        
        print(f"\nQuestion: {question}")
        
        # 1. Basic generate (return_dict_in_generate=False)
        print("\n" + "="*60)
        print("1. BASIC GENERATE (return_dict_in_generate=False)")
        print("="*60)
        
        # Analyze the model directly
        processed_image = engine.image_processor.preprocess_image(test_image)
        prompt = f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = engine.processor(
            text=prompt,
            images=processed_image,
            return_tensors="pt"
        )
        
        device = next(engine.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        print("Analyzing basic generate output...")
        with torch.no_grad():
            basic_outputs = engine.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.2,
                do_sample=True,
                pad_token_id=0
            )
        
        print(f"Basic outputs type: {type(basic_outputs)}")
        print(f"Basic outputs shape: {basic_outputs.shape}")
        print(f"Basic outputs dtype: {basic_outputs.dtype}")
        print(f"Basic outputs device: {basic_outputs.device}")
        
        # Analyze the tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = basic_outputs[0][input_length:]
        
        print(f"\nInput tokens: {input_length}")
        print(f"Generated tokens: {len(generated_tokens)}")
        print(f"Total tokens: {basic_outputs.shape[1]}")
        
        # Decode
        response = engine.processor.decode(generated_tokens, skip_special_tokens=True)
        print(f"Response: {response[:100]}...")
        
        # 2. return_dict_in_generate=True)
        print("\n" + "="*60)
        print("2. DICT GENERATE (return_dict_in_generate=True)")
        print("="*60)
        
        print("Analyzing dict generate output...")
        with torch.no_grad():
            dict_outputs = engine.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.2,
                do_sample=True,
                pad_token_id=0,
                return_dict_in_generate=True
            )
        
        print(f"Dict outputs type: {type(dict_outputs)}")
        print(f"Dict outputs class: {dict_outputs.__class__}")
        
        # analyze the attributes
        print(f"\nAvailable attributes:")
        for attr in dir(dict_outputs):
            if not attr.startswith('_'):
                try:
                    value = getattr(dict_outputs, attr)
                    if not callable(value):
                        print(f"  {attr}: {type(value)}")
                        if hasattr(value, 'shape'):
                            print(f"      Shape: {value.shape}")
                        elif isinstance(value, (list, tuple)):
                            print(f"      Length: {len(value)}")
                except:
                    print(f"  {attr}: (error accessing)")
        
        # analyze the sequences
        if hasattr(dict_outputs, 'sequences'):
            print(f"\nSequences:")
            print(f"  Type: {type(dict_outputs.sequences)}")
            print(f"  Shape: {dict_outputs.sequences.shape}")
            print(f"  Device: {dict_outputs.sequences.device}")
            
            # extract the generated tokens
            generated_from_dict = dict_outputs.sequences[0][input_length:]
            response_from_dict = engine.processor.decode(generated_from_dict, skip_special_tokens=True)
            print(f"  Response: {response_from_dict[:100]}...")
        
        # analyze the scores (if available)
        if hasattr(dict_outputs, 'scores') and dict_outputs.scores is not None:
            print(f"\nScores:")
            print(f"  Type: {type(dict_outputs.scores)}")
            print(f"  Length: {len(dict_outputs.scores)}")
            if len(dict_outputs.scores) > 0:
                print(f"  First score shape: {dict_outputs.scores[0].shape}")
                print(f"  First score type: {type(dict_outputs.scores[0])}")
        
        # 3. output_attentions=True
        print("\n" + "="*60)
        print("3. WITH ATTENTIONS (output_attentions=True)")
        print("="*60)
        
        print("Analyzing attention output...")
        with torch.no_grad():
            attention_outputs = engine.model.generate(
                **inputs,
                max_new_tokens=30,  # save memory
                temperature=0.2,
                do_sample=True,
                pad_token_id=0,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        print(f"Attention outputs type: {type(attention_outputs)}")
        
        # analyze the attentions
        if hasattr(attention_outputs, 'attentions') and attention_outputs.attentions is not None:
            print(f"\nAttentions:")
            print(f"  Type: {type(attention_outputs.attentions)}")
            print(f"  Length (layers): {len(attention_outputs.attentions)}")
            
            # analyze the attention of each layer
            for i, layer_attention in enumerate(attention_outputs.attentions[:3]):  # analyze the first 3 layers only
                if layer_attention is not None:
                    print(f"  Layer {i}:")
                    if hasattr(layer_attention, 'shape'):
                        print(f"    Shape: {layer_attention.shape}")
                        print(f"    Type: {type(layer_attention)}")
                        print(f"    Device: {layer_attention.device}")
                        print(f"    Dtype: {layer_attention.dtype}")
                    elif isinstance(layer_attention, (list, tuple)):
                        print(f"    Type: {type(layer_attention)} with {len(layer_attention)} elements")
                        if len(layer_attention) > 0 and hasattr(layer_attention[0], 'shape'):
                            print(f"    First element shape: {layer_attention[0].shape}")
                else:
                    print(f"  Layer {i}: None")
            
            if len(attention_outputs.attentions) > 3:
                print(f"  ... and {len(attention_outputs.attentions) - 3} more layers")
        else:
            print(f"\nAttentions: None or not available")
        
        # 4. output_scores=True
        print("\n" + "="*60)
        print("4. WITH SCORES (output_scores=True)")
        print("="*60)
        
        print("Analyzing scores output...")
        with torch.no_grad():
            scores_outputs = engine.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.2,
                do_sample=True,
                pad_token_id=0,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        if hasattr(scores_outputs, 'scores') and scores_outputs.scores is not None:
            print(f"\nScores:")
            print(f"  Type: {type(scores_outputs.scores)}")
            print(f"  Length (generation steps): {len(scores_outputs.scores)}")
            
            if len(scores_outputs.scores) > 0:
                first_score = scores_outputs.scores[0]
                print(f"  First step score:")
                print(f"    Shape: {first_score.shape}")  # (batch_size, vocab_size)
                print(f"    Type: {type(first_score)}")
                print(f"    Device: {first_score.device}")
                print(f"    Dtype: {first_score.dtype}")
                
                # analyze the top tokens
                top_tokens = torch.topk(first_score[0], k=5)
                print(f"    Top 5 token IDs: {top_tokens.indices.tolist()}")
                print(f"    Top 5 scores: {top_tokens.values.tolist()}")
                
                # convert the tokens to text
                top_words = [engine.processor.tokenizer.decode([token_id]) for token_id in top_tokens.indices]
                print(f"    Top 5 words: {top_words}")
        
        # 5. All options together
        print("\n" + "="*60)
        print("5. ALL OPTIONS TOGETHER")
        print("="*60)
        
        print("Analyzing full output...")
        with torch.no_grad():
            full_outputs = engine.model.generate(
                **inputs,
                max_new_tokens=15,  # save memory
                temperature=0.2,
                do_sample=True,
                pad_token_id=0,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        print(f"Full outputs type: {type(full_outputs)}")
        print(f"Available attributes:")
        
        for attr in ['sequences', 'scores', 'attentions']:
            if hasattr(full_outputs, attr):
                value = getattr(full_outputs, attr)
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        print(f"  {attr}: {type(value)} {value.shape}")
                    elif isinstance(value, (list, tuple)):
                        print(f"  {attr}: {type(value)} length={len(value)}")
                    else:
                        print(f"  {attr}: {type(value)}")
                else:
                    print(f"  {attr}: None")
            else:
                print(f"  {attr}: Not available")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # clean up the memory
        if engine.is_loaded:
            engine.unload_model()

if __name__ == "__main__":
    analyze_outputs_structure()