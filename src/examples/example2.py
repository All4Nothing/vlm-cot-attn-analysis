# Add these imports at the top of your handler.py file
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

# This is the full code to add to your LlavaHandler class

def get_attention_for_token(self, image_source: str, prompt_text: str, target_token_str: str, layer: int, head: int):
    """
    Runs inference and extracts the attention scores for a specific token from a specific head.
    """
    if self.model is None:
        raise RuntimeError("Model is not loaded. Call `load_model()` first.")

    # 1. Prepare inputs and run model to get outputs and hidden states
    prompt = f"USER: <image>\n{prompt_text}"
    if image_source.startswith("http"):
        image = Image.open(requests.get(image_source, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_source).convert("RGB")
    
    inputs = self.processor(prompt, images=image, return_tensors="pt").to(self.device)

    # To get attention, we need to pass `output_attentions=True`
    outputs = self.model.generate(**inputs, max_new_tokens=100, output_attentions=True, return_dict_in_generate=True)
    
    # 2. Find the index of the target token in the generated sequence
    generated_ids = outputs.sequences[0]
    target_token_ids = self.processor.tokenizer.encode(target_token_str, add_special_tokens=False)
    
    target_idx = -1
    # Search for the first occurrence of the target token sequence
    for i in range(len(generated_ids) - len(target_token_ids) + 1):
        if torch.equal(generated_ids[i:i+len(target_token_ids)], torch.tensor(target_token_ids, device=self.device)):
            target_idx = i
            break
            
    if target_idx == -1:
        print(f"
        ⚠️ Warning: Target token '{target_token_str}' not found in the generated text.")
        return None

    # 3. Extract the attention scores
    # The image features are always at the beginning of the sequence.
    # For LLaVA-1.5, the image is projected into 576 tokens (24x24 grid).
    num_image_patches = 576 
    
    # attentions shape: (num_layers, batch, num_heads, sequence_length, sequence_length)
    attention_matrix = outputs.attentions[-1][layer][0, head] # We look at the attention from the final generated token
    
    # Get scores from all image patches TO the target text token
    attention_scores = attention_matrix[:num_image_patches, target_idx]
    
    return attention_scores.detach().cpu().numpy()


def visualize_attention(self, image_source: str, prompt_text: str, target_token_str: str, layer: int, head: int, save_path: str = None):
    """
    Visualizes the attention from a specific head to a target token as a heatmap on the image.
    """
    print(f"🔬 Visualizing attention for token '{target_token_str}' from Layer {layer}, Head {head}...")
    
    # Step 1: Extract the relevant attention scores
    attention_scores = self.get_attention_for_token(image_source, prompt_text, target_token_str, layer, head)
    
    if attention_scores is None:
        return

    # Step 2: Reshape scores into a grid (LLaVA uses a 24x24 grid)
    grid_size = int(np.sqrt(attention_scores.shape[0]))
    attention_grid = attention_scores.reshape(grid_size, grid_size)

    # Step 3: Upscale and Overlay on the original image
    if image_source.startswith("http"):
        image = Image.open(requests.get(image_source, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_source).convert("RGB")

    # Resize the low-res attention grid to the original image size
    zoom_factors = (image.height / grid_size, image.width / grid_size)
    resized_attention = zoom(attention_grid, zoom_factors)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    # Overlay the heatmap
    im = ax.imshow(resized_attention, cmap='jet', alpha=0.5) # alpha controls transparency
    ax.axis('off')
    fig.colorbar(im, ax=ax)
    
    title = f"Attention Heatmap for '{target_token_str}'\n(Layer {layer}, Head {head})"
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Visualization saved to {save_path}")
    else:
        plt.show()