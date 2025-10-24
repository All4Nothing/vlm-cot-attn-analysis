"""
Find attention heads that focus on the actual object location (center region)
Identifies heads with high attention scores in the central image region
"""

import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from transformers import AutoProcessor


def make_heatmap_from_patch_vector(patch_vector: torch.Tensor, grid_size: int = 24) -> np.ndarray:
    """Convert patch attention vector to 2D heatmap"""
    v = patch_vector.detach().float().cpu().numpy()
    v = v.reshape(grid_size, grid_size)
    maxv = v.max()
    if maxv > 0:
        v = v / maxv
    return v


def overlay_on_image(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay heatmap on image"""
    h, w = img.shape[:2]
    heat_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_color = plt.cm.jet(heat_resized)[:, :, :3]
    heat_color = (heat_color * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 1 - alpha, heat_color, alpha, 0)
    return overlay


def find_object_token_positions(object_name: str, generated_ids: torch.Tensor, tokenizer: Any) -> List[int]:
    """Find token positions where the object name appears"""
    obj_ids: List[int] = tokenizer.encode(object_name, add_special_tokens=False)
    if not obj_ids:
        return []
    gen = generated_ids[0].tolist()
    
    positions: List[int] = []
    for i in range(len(gen) - len(obj_ids) + 1):
        if gen[i:i+len(obj_ids)] == obj_ids:
            positions.append(i + len(obj_ids) - 1)
    return positions


def calculate_center_attention_score(heatmap: np.ndarray, 
                                     center_region_ratio: float = 0.3) -> float:
    """
    Calculate the average attention score in the center region of the heatmap
    
    Args:
        heatmap: 2D heatmap (normalized 0-1)
        center_region_ratio: Ratio of center region (0.3 = 30% of image centered)
    
    Returns:
        Average attention score in center region
    """
    h, w = heatmap.shape
    
    # Define center region
    center_h_start = int(h * (0.5 - center_region_ratio / 2))
    center_h_end = int(h * (0.5 + center_region_ratio / 2))
    center_w_start = int(w * (0.5 - center_region_ratio / 2))
    center_w_end = int(w * (0.5 + center_region_ratio / 2))
    
    # Extract center region
    center_region = heatmap[center_h_start:center_h_end, center_w_start:center_w_end]
    
    # Calculate average attention in center
    center_score = np.mean(center_region)
    
    return center_score


def analyze_all_heads(
    json_path: str,
    image_path: str,
    object_name: str,
    output_dir: str,
    center_region_ratio: float = 0.3,
    top_k: int = 10,
    image_token_end: int = 576,
    grid_size: int = 24
):
    """
    Analyze all layer-head combinations and rank by center attention score
    
    Args:
        json_path: Path to results JSON
        image_path: Path to image
        object_name: Object name to analyze
        output_dir: Output directory
        center_region_ratio: Ratio of center region to analyze
        top_k: Number of top heads to save
        image_token_end: End index of image tokens
        grid_size: Grid size for patches (24x24 = 576)
    """
    # 1. Load JSON and cache info
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result = data["results"][0]
    model_path = data["metadata"]["model_config"]["model_path"]
    cache_path = result["attention_data"]["cache_path"]
    
    cache_blob = torch.load(cache_path, map_location="cpu")
    attentions = cache_blob["attentions"]
    generated_ids = cache_blob["generated_ids"]
    input_length = result["token_info"]["input_tokens"]
    
    # 2. Load tokenizer and image
    print("Loading tokenizer and image...")
    tokenizer = AutoProcessor.from_pretrained(model_path, use_fast=True).tokenizer
    img = np.array(Image.open(image_path).convert("RGB"))
    
    # 3. Find object token positions
    print(f"Finding token positions for object '{object_name}'...")
    token_positions = find_object_token_positions(object_name, generated_ids, tokenizer)
    if not token_positions:
        print(f"Warning: Object '{object_name}' not found in generated text.")
        return
    
    print(f"Found {len(token_positions)} occurrences of '{object_name}'")
    
    # 4. Analyze all layer-head combinations
    num_layers = len(attentions[0])
    num_heads = attentions[0][0].shape[1]
    
    print(f"Analyzing {num_layers} layers x {num_heads} heads = {num_layers * num_heads} combinations...")
    
    head_scores = []  # List of (layer, head, center_score, heatmap, mean_vec)
    
    for layer in range(num_layers):
        if layer == 0: continue
        for head_idx in range(num_heads):
            vectors = []
            
            # Collect attention vectors for all object token occurrences
            for qpos in token_positions:
                generation_step_idx = qpos - input_length
                if not (0 <= generation_step_idx < len(attentions)):
                    continue
                
                # Get attention for this step
                layer_attn_at_step = attentions[generation_step_idx][layer]
                
                # Extract attention vector from image patches
                vec = layer_attn_at_step[0, head_idx, 0, :image_token_end]
                vectors.append(vec)
            
            if not vectors:
                continue
            
            # Calculate mean attention vector
            mean_vec = torch.stack(vectors).mean(dim=0)
            
            # Convert to heatmap
            heatmap = make_heatmap_from_patch_vector(mean_vec, grid_size)
            
            # Calculate center attention score
            center_score = calculate_center_attention_score(heatmap, center_region_ratio)
            
            head_scores.append({
                'layer': layer,
                'head': head_idx,
                'center_score': center_score,
                'heatmap': heatmap,
                'mean_vec': mean_vec
            })
        
        if (layer + 1) % 5 == 0:
            print(f"  Processed {layer + 1}/{num_layers} layers...")
    
    # 5. Sort by center attention score
    head_scores.sort(key=lambda x: x['center_score'], reverse=True)
    
    # 6. Print top heads
    print(f"\n{'='*80}")
    print(f"TOP {top_k} HEADS WITH HIGHEST CENTER ATTENTION (object: '{object_name}')")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Center Score':<15} {'Description'}")
    print(f"{'-'*80}")
    
    for rank, info in enumerate(head_scores[:top_k], 1):
        layer = info['layer']
        head = info['head']
        score = info['center_score']
        print(f"{rank:<6} L{layer:<7} H{head:<7} {score:.4f}{' '*9} "
              f"{'🎯 HIGH' if score > 0.5 else '✓ Good' if score > 0.3 else 'Low'}")
    
    # 7. Save top K visualizations
    print(f"\nSaving top {top_k} visualizations to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    basename = os.path.splitext(os.path.basename(image_path))[0]
    
    for rank, info in enumerate(head_scores[:top_k], 1):
        layer = info['layer']
        head = info['head']
        score = info['center_score']
        heatmap = info['heatmap']
        
        # Create overlay
        overlay = overlay_on_image(img, heatmap)
        
        # Add text annotation
        overlay_pil = Image.fromarray(overlay)
        
        # Save
        out_filename = f"rank{rank:02d}_L{layer:02d}_H{head:02d}_score{score:.3f}_{basename}_obj-{object_name}.png"
        out_path = os.path.join(output_dir, out_filename)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(overlay)
        plt.title(f"Rank #{rank}: Layer {layer}, Head {head} | Center Score: {score:.4f}", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved top {top_k} visualizations")

    # 9. Create visualization grid for top heads
    print(f"\nCreating comparison grid...")
    create_comparison_grid(head_scores[:top_k], img, output_dir, basename, object_name)
    
    print(f"\n{'='*80}")
    print(f"✓ Analysis complete!")
    print(f"{'='*80}")


def create_comparison_grid(top_heads: List[Dict], img: np.ndarray, 
                          output_dir: str, basename: str, object_name: str):
    """Create a grid comparison of top heads"""
    n = len(top_heads)
    cols = 5
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, info in enumerate(top_heads):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        overlay = overlay_on_image(img, info['heatmap'])
        ax.imshow(overlay)
        ax.set_title(f"#{idx+1}: L{info['layer']} H{info['head']}\nScore: {info['center_score']:.3f}", 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f"Top {n} Heads for '{object_name}' (by center attention)", 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    grid_path = os.path.join(output_dir, f"comparison_grid_{basename}_obj-{object_name}.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison grid to {grid_path}")


if __name__ == "__main__":
    analyze_all_heads(
        json_path="./outputs/attention_cache/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657114112404_results.json",
        image_path="./data/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657114112404.jpg",
        object_name="bus",
        output_dir="./outputs/focused_heads_analysis",
        center_region_ratio=0.4,  # Analyze central 40% of image
        top_k=20  # Save top 20 heads
    )

