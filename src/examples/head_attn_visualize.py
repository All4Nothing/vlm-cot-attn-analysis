# method1/visualize_attention.py
# (Modified version based on user-provided code with core logic changes)

import os
import json
from typing import List, Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from transformers import AutoProcessor
import argparse

# (make_heatmap_from_patch_vector, overlay_on_image functions are same as provided code)
def make_heatmap_from_patch_vector(patch_vector: torch.Tensor, grid_size: int = 24) -> np.ndarray:
    v = patch_vector.detach().float().cpu().numpy()
    v = v.reshape(grid_size, grid_size)
    maxv = v.max()
    if maxv > 0: v = v / maxv
    return v

def overlay_on_image(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = img.shape[:2]
    heat_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_color = plt.cm.jet(heat_resized)[:, :, :3]
    heat_color = (heat_color * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 1 - alpha, heat_color, alpha, 0)
    return overlay


def find_object_token_positions(object_name: str, generated_ids: torch.Tensor, tokenizer: Any) -> List[int]:
    obj_ids: List[int] = tokenizer.encode(object_name, add_special_tokens=False)
    if not obj_ids: return []
    gen = generated_ids[0].tolist()
    
    positions: List[int] = []
    for i in range(len(gen) - len(obj_ids) + 1):
        if gen[i:i+len(obj_ids)] == obj_ids:
            # Use the last token position of the object (has richest contextual info)
            positions.append(i + len(obj_ids) - 1)
    # print(f"obj_ids: {obj_ids}, positions: {positions}")
    return positions


def visualize(
    json_path: str,
    image_path: str,
    object_name: str,
    layer: int,
    output_dir: str,
    image_token_end: int = 576
):
    # 1. Load JSON and cache info
    with open(json_path, 'r', encoding='utf-8') as f: 
        data = json.load(f)
    # result = next(r for r in data["results"] if r["image_path"] == image_path)
    result = data["results"][0]
    model_path = data["metadata"]["model_config"]["model_path"]

    cache_path = result["attention_data"]["cache_path"]
    cache_blob = torch.load(cache_path, map_location="cpu")
    attentions = cache_blob["attentions"]
    generated_ids = cache_blob["generated_ids"]
    input_length = result["token_info"]["input_tokens"]

    # 2. Load required tools (tokenizer, image)
    tokenizer = AutoProcessor.from_pretrained(model_path, use_fast=True).tokenizer
    img = np.array(Image.open(image_path).convert("RGB"))

    # 3. Find object token positions
    token_positions = find_object_token_positions(object_name, generated_ids, tokenizer)
    if not token_positions:
        print(f"Warning: Object '{object_name}' not found in generated text.")
        return

    # 4. Visualize each head
    os.makedirs(f"{output_dir}/L{layer}", exist_ok=True)
    num_heads = attentions[0][layer].shape[1] # Get number of heads from first step attention
    
    for head_idx in range(num_heads):
        vectors = []
        for qpos in token_positions:
            # Core logic: Convert absolute token position (qpos) to generation step index
            generation_step_idx = qpos - input_length
            if not (0 <= generation_step_idx < len(attentions)): continue


            # decoded_token = tokenizer.decode(generated_ids[0, qpos])
            # print(f"Step {generation_step_idx} (absolute position={qpos}) | decoded token: '{decoded_token}' | target object: '{object_name}'")
            
            # Get attention tensor for this step (shape: [1,H,1,KeyLen])
            layer_attn_at_step = attentions[generation_step_idx][layer]
            
            # Extract attention vector from image patches to object token
            vec = layer_attn_at_step[0, head_idx, 0, :image_token_end]
            vectors.append(vec)

        if not vectors: continue
        
        # 5. Generate and save heatmap
        mean_vec = torch.stack(vectors).mean(dim=0)
        heatmap = make_heatmap_from_patch_vector(mean_vec)
        overlay = overlay_on_image(img, heatmap)

        basename = os.path.splitext(os.path.basename(image_path))[0][:4]
        out_path = os.path.join(output_dir, f"L{layer}", f"{basename}_obj-{object_name}_L{layer}_H{head_idx}.png")
        plt.imsave(out_path, overlay)

    print(f"Completed visualization of all heads in L{layer} for object '{object_name}'.")


if __name__ == "__main__":
    """parser = argparse.ArgumentParser(description="Visualize attention for a specific object.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the results JSON file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path or URL to the input image.")
    parser.add_argument("--object_name", type=str, required=True, help="Object to visualize attention for.")
    parser.add_argument("--layer", type=int, default=15, help="Layer index to visualize.")
    parser.add_argument("--output_dir", type=str, default="./attention_maps", help="Directory to save visualizations.")
    args = parser.parse_args()"""
    
    visualize(
        json_path="./outputs/attention_cache/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657114112404_results.json",
        image_path="./data/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657114112404.jpg",
        object_name="car",
        layer=15,
        output_dir="./outputs/attention_head_visualization2"
    )