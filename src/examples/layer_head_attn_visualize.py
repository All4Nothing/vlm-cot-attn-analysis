import os
import json
from typing import List
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from transformers import AutoProcessor
import argparse

def make_heatmap_from_patch_vector(patch_vector: torch.Tensor, grid_size: int = 24) -> np.ndarray:
    """Creates a 2D heatmap from a 1D patch vector."""
    v = patch_vector.detach().float().cpu().numpy()
    v = v.reshape(grid_size, grid_size)
    maxv = v.max()
    if maxv > 0:
        v = v / maxv
    return v

def overlay_on_image(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays a heatmap on an image."""
    h, w = img.shape[:2]
    heat_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_color = plt.cm.jet(heat_resized)[:, :, :3]
    heat_color = (heat_color * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 1 - alpha, heat_color, alpha, 0)
    return overlay

def find_object_token_positions(object_name: str, generated_ids: torch.Tensor, tokenizer: any) -> List[int]:
    """Finds the last token positions of an object in the generated token ID sequence."""
    obj_ids: List[int] = tokenizer.encode(object_name, add_special_tokens=False)
    if not obj_ids: return []
    gen = generated_ids[0].tolist()
    positions: List[int] = []
    for i in range(len(gen) - len(obj_ids) + 1):
        if gen[i:i+len(obj_ids)] == obj_ids:
            positions.append(i + len(obj_ids) - 1)
    return positions

def visualize_across_layers(
    json_path: str,
    image_path: str,
    object_name: str,
    layers_to_viz: List[int],
    output_dir: str,
    image_token_end: int = 576
):
    """
    Visualizes attention for a specific object across specified layers.
    """
    # 1. Load JSON and cache info
    print(f"Starting analysis: Finding attention for object '{object_name}' in layers {layers_to_viz}")
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    result = data["results"][0]
    model_path = data["metadata"]["model_config"]["model_path"]

    cache_path = result["attention_data"]["cache_path"]
    cache_blob = torch.load(cache_path, map_location="cpu")
    attentions = cache_blob["attentions"]
    generated_ids = cache_blob["generated_ids"]
    input_length = result["token_info"]["input_tokens"]

    # 2. Load required tools
    tokenizer = AutoProcessor.from_pretrained(model_path).tokenizer
    img = np.array(Image.open(image_path).convert("RGB"))

    # 3. Find object token positions
    token_positions = find_object_token_positions(object_name, generated_ids, tokenizer)
    if not token_positions:
        print(f"Warning: Object '{object_name}' not found in generated text. Skipping analysis.")
        return

    # 4. Analyze each specified layer
    for layer in layers_to_viz:
        print(f"\n--- Analyzing Layer {layer} ---")
        
        # Get number of heads from attention tensor
        num_heads = attentions[0][layer].shape[1]
        
        for head_idx in range(num_heads):
            vectors = []
            for qpos in token_positions:
                # Core logic: Convert absolute token position to generation step index
                generation_step_idx = qpos - input_length
                if not (0 <= generation_step_idx < len(attentions)): continue

                # Get attention tensor for this step and current layer
                layer_attn_at_step = attentions[generation_step_idx][layer]
                
                # Extract attention vector from image patches to object token
                vec = layer_attn_at_step[0, head_idx, 0, :image_token_end]
                vectors.append(vec)

            if not vectors: continue
            
            # 5. Generate and save heatmap (including layer number in filename)
            mean_vec = torch.stack(vectors).mean(dim=0)
            heatmap = make_heatmap_from_patch_vector(mean_vec)
            overlay = overlay_on_image(img, heatmap)

            # Create output directory and set file path
            layer_output_dir = os.path.join(output_dir, f"layer_{layer}")
            os.makedirs(layer_output_dir, exist_ok=True)
            
            basename = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(layer_output_dir, f"{basename}_obj-{object_name}_L{layer}_H{head_idx}.png")
            plt.imsave(out_path, overlay)

        print(f"Completed visualization of all heads for Layer {layer}.")

if __name__ == "__main__":
    """parser = argparse.ArgumentParser(description="Visualize attention across multiple layers for a specific object.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the results JSON file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path or URL to the input image.")
    parser.add_argument("--object_name", type=str, required=True, help="Object to visualize attention for.")
    parser.add_argument("--layers", type=int, nargs='+', required=True, help="List of layer indices to visualize (e.g., 5 15 30).")
    parser.add_argument("--output_dir", type=str, default="./layer_analysis_maps", help="Base directory to save visualizations.")
    args = parser.parse_args()"""
    
    # Since argparse stores the layers argument as 'layers', we match the name in function call
    visualize_across_layers(
        json_path="./outputs/attention_cache/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657114112404_results.json",
        image_path="./data/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657114112404.jpg",
        object_name="car",
        layers_to_viz=range(0,4),
        output_dir="./outputs/layer_attention_visualization"
    )