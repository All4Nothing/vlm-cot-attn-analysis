import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def save_all_heads(attn: torch.Tensor, meta: Dict, save_dir: str) -> None:
    """Save all layer-head attention maps to individual PNG files.
    
    Args:
        attn: [L, H, 1, V] attention tensor
        meta: metadata dictionary with patch_size and image_file
        save_dir: directory to save attention maps
    """
    L, H, _, V = attn.shape
    P = int(meta.get("patch_size", int(np.sqrt(V))))
    
    # Create directory structure
    all_heads_dir = os.path.join(save_dir, "all_heads")
    os.makedirs(all_heads_dir, exist_ok=True)
    
    print(f"Saving all {L} layers × {H} heads = {L*H} attention maps...")
    
    # Save each layer-head combination
    for layer in range(L):
        layer_dir = os.path.join(all_heads_dir, f"layer_{layer:02d}")
        os.makedirs(layer_dir, exist_ok=True)
        
        for head in range(H):
            # Extract and reshape attention map
            a2d = attn[layer, head, 0].reshape(P, P).detach().cpu().numpy()
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            im = ax.imshow(a2d, cmap="viridis", interpolation='nearest')
            ax.set_title(f"Layer {layer}, Head {head}", fontsize=14, fontweight='bold')
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Save
            filename = f"L{layer:02d}_H{head:02d}.png"
            filepath = os.path.join(layer_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close(fig)
        
        if (layer + 1) % 5 == 0:
            print(f"  Saved {layer + 1}/{L} layers...")
    
    print(f"✓ All attention maps saved to: {all_heads_dir}")


def save_all_heads_grid(attn: torch.Tensor, meta: Dict, save_path: str, heads_per_row: int = 8) -> None:
    """Save all attention maps in a large grid for overview.
    
    Args:
        attn: [L, H, 1, V] attention tensor
        meta: metadata dictionary
        save_path: path to save the grid image
        heads_per_row: number of heads per row in the grid
    """
    L, H, _, V = attn.shape
    P = int(meta.get("patch_size", int(np.sqrt(V))))
    
    total_heads = L * H
    rows = (total_heads + heads_per_row - 1) // heads_per_row
    
    fig, axes = plt.subplots(rows, heads_per_row, figsize=(2 * heads_per_row, 2 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    print(f"Creating overview grid: {rows} rows × {heads_per_row} cols...")
    
    idx = 0
    for layer in range(L):
        for head in range(H):
            row = idx // heads_per_row
            col = idx % heads_per_row
            
            a2d = attn[layer, head, 0].reshape(P, P).detach().cpu().numpy()
            axes[row, col].imshow(a2d, cmap="viridis", interpolation='nearest')
            axes[row, col].set_title(f"L{layer}H{head}", fontsize=8)
            axes[row, col].axis("off")
            
            idx += 1
    
    # Hide unused subplots
    for idx in range(total_heads, rows * heads_per_row):
        row = idx // heads_per_row
        col = idx % heads_per_row
        axes[row, col].axis("off")
    
    plt.suptitle(f"All Attention Heads ({L} layers × {H} heads)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Overview grid saved to: {save_path}")


def plot_heads_grid(attn: torch.Tensor, selected: List[Dict], meta: Dict, save_path: str, show_plot: bool) -> None:
    """Save a figure: original image + top-K attention maps.

    attn: [L, H, 1, V]
    """
    P = int(meta.get("patch_size"))
    W, H_img = meta["image_size"]
    n = len(selected)
    cols = n + 1
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))

    # Original image
    try:
        img = Image.open(meta["image_file"]).convert("RGB")
        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")
    except Exception as e:
        axes[0].text(0.5, 0.5, f"Image load error\n{e}", ha='center', va='center')
        axes[0].axis("off")

    # Attention maps
    for i, hinfo in enumerate(selected):
        l, h = hinfo["layer"], hinfo["head"]
        a2d = attn[l, h, 0].reshape(P, P).detach().cpu().numpy()
        im = axes[i + 1].imshow(a2d, cmap="viridis")
        axes[i + 1].set_title(f"L{l}-H{h}\nSE={hinfo['spatial_entropy']:.2f}")
        axes[i + 1].axis("off")
        plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

