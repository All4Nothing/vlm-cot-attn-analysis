import torch
import json
# import argparse

def verify_attention_cache(json_path: str):
    """
    Load attention cache connected to JSON file and
    verify that the number of generated tokens matches the length of attention records.
    """
    # 1. Load JSON metadata
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = data["results"][0]
    token_info = result["token_info"]
    cache_path = result["attention_data"]["cache_path"]

    num_generated_tokens = token_info["output_tokens"]
    
    print("--- JSON Metadata Info ---")
    print(f"JSON file: {json_path}")
    print(f"Image file: {result['image_path']}")
    print(f"Number of generated tokens: {num_generated_tokens}\n")

    # 2. Load attention cache (.pt) file
    try:
        cache_blob = torch.load(cache_path, map_location="cpu")
        attentions = cache_blob["attentions"]
    except FileNotFoundError:
        print(f"Error: Cache file not found! Path: {cache_path}")
        return
    
    # 3. Structure verification
    print("--- Attention Cache Structure Verification ---")
    
    # attentions should be Tuple type
    print(f"Data type: {type(attentions)}")

    # Check tuple length (number of steps)
    num_attention_steps = len(attentions)
    print(f"Number of recorded attention steps: {num_attention_steps}")
    
    # 4. Final comparison
    print("\n--- Final Verification Result ---")
    if num_generated_tokens == num_attention_steps:
        print(f"Match: Number of generated tokens ({num_generated_tokens}) equals number of attention steps ({num_attention_steps})")
    else:
        print(f"Mismatch: Number of generated tokens ({num_generated_tokens}) differs from number of attention steps ({num_attention_steps})")

    # Print detailed structure of first step (example)
    if num_attention_steps > 0:
        first_step_attentions = attentions[0]
        num_layers = len(first_step_attentions)
        first_layer_shape = first_step_attentions[0].shape
        print("\n--- Detailed Structure of First Step (Step 0) ---")
        print(f"Number of layers: {num_layers}")
        print(f"Attention tensor shape of first layer: {first_layer_shape}")
        print("(Batch, Heads, Query_Len, Key_Len)")


if __name__ == "__main__":
    verify_attention_cache("/workspace/yongjoo/vlm-cot-attn-analysis/outputs/attention_cache/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657114112404_results.json")