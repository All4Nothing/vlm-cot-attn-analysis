import os
import re
import pickle
from typing import Dict, Tuple

import torch
import numpy as np
from PIL import Image

from models.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from models.llava.conversation import conv_templates
from models.llava.model.builder import load_pretrained_model
from models.llava.utils import disable_torch_init
from models.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from lab.stations import MetadataStation


def _sanitize_name(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        import requests
        from io import BytesIO
        resp = requests.get(path_or_url)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"Image not found: {path_or_url}")
    return Image.open(path_or_url).convert("RGB")


def load_model_from_cfg(cfg) -> Tuple[object, object, object, int, str]:
    """Build model/tokenizer/image_processor using Hugging Face path.

    Returns: tokenizer, model, image_processor, context_len, model_name_str
    """
    disable_torch_init()
    # Optional user cache control (force override; must be set before downloads)
    if getattr(cfg.model, "cache_dir", None):
        cache_dir = str(cfg.model.cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_HUB_CACHE"] = cache_dir

    # Choose device string
    device = cfg.device
    if device == "auto":
        device = f"cuda:{cfg.device_id}" if cfg.device_id >= 0 and torch.cuda.is_available() else "cpu"

    model_name_str = get_model_name_from_path(cfg.model.name)
    tok, model, img_proc, context_len = load_pretrained_model(
        model_path=cfg.model.name,
        cache_dir=cfg.model.cache_dir,
        model_base=cfg.model.base,
        model_name=model_name_str,
        device=device,
        use_flash_attn=getattr(cfg.model, "use_flash_attn", False),
    )
    return tok, model, img_proc, context_len, model_name_str


def _forward_collect(model, tokenizer, image_processor, input_ids, image_tensor, image_sizes):
    """Collect attentions via a single forward pass.

    Returns attention focused on image tokens with shape [L, H, 1, V].
    """
    outputs = model(
        input_ids=input_ids,
        images=image_tensor.unsqueeze(0),
        image_sizes=image_sizes,
        output_attentions=True,
        return_dict=True,
    )
    print(f"outputs keys: {outputs.keys()}") # ['logits', 'past_key_values', 'attentions']
    print(f"outputs.logits: {outputs.logits.shape}") # [1, 576 + text_len, 32000(vocab_size)]
    print(f"outputs.past_key_values: {outputs.past_key_values}") # [1, 32, 2, 1, 629, 629]
    print(f"outputs.attentions: {len(outputs.attentions)}") # 32
    print(f"len of outputs.attentions[0]: {len(outputs.attentions[0])}") # 1
    print(f"shape of outputs.attentions[0][0]: { outputs.attentions[0][0].shape}") # [32, 615, 615]

    print(f"len of outputs.attentions[1]: {len( outputs.attentions[1])}") # 1
    print(f"shape of outputs.attentions[1][0]: { outputs.attentions[1][0].shape}") # [32, 615, 615]
    attn_layers = outputs.attentions  # tuple length L of [B,H,Tq,Tk]
    if not attn_layers:
        raise RuntimeError("No attentions returned from forward()")

    layers = []
    for t in attn_layers:
        layers.append(t[0])  # [H,Tq,Tk] for batch=1
    attn = torch.stack(layers, dim=0)  # [L,H,Tq,Tk]
    print(f"shape of attn: {attn.shape}") # [32, 32, 620, 620] 620 = 576 + 44
    begin_pos_vis = MetadataStation.get_begin_pos('vis')
    vis_len = MetadataStation.get_vis_len()
    if begin_pos_vis is None or vis_len is None:
        raise RuntimeError("Missing visual token segmentation info.")
    print(f"begin_pos_vis: {begin_pos_vis}")
    print(f"vis_len: {vis_len}")
    attn_last_to_vis = attn[:, :, -1:, begin_pos_vis:begin_pos_vis + vis_len]
    
    masked_attn_vis = attn[-1, -1, :, :]
    # Print number of non-zero values for each row in masked_attn_vis
    """ for row_idx in range(10):
        print(masked_attn_vis[row_idx, :10]) # lower triangular matrix
        non_zero_count = torch.count_nonzero(masked_attn_vis[row_idx]).item()
        print(f"Row {row_idx}: {non_zero_count} non-zero values")
    print(f"shape of masked_attn_vis: {masked_attn_vis.shape}") # [sys + image + input, sys + image + input]
    print(f"masked_attn_vis: {masked_attn_vis}")"""
    print(f"shape of attn_last_to_vis: {attn_last_to_vis.shape}") # [32, 32, 1, 576]
    return attn_last_to_vis


def _generate_collect(model, tokenizer, image_processor, input_ids, image_tensor, image_sizes, max_new_tokens=10, do_sample=False, num_beams=1):
    """Run generate to obtain output tokens, and try to collect attentions from the first generation step.

    Returns: (attn [L,H,1,V] or None, generated_text str)
    """
    gen = model.generate(
        inputs=input_ids,
        images=image_tensor.unsqueeze(0),
        image_sizes=image_sizes,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        return_dict_in_generate=True,
        output_attentions=True,
    )
    sequences = gen.sequences
    input_len = input_ids.shape[1]
    
    # Decode input text (excluding IMAGE_TOKEN_INDEX = -200)
    print(f"\n=== Input Information ===")
    print(f"Input token IDs: {input_ids[0]}")
    # Filter out IMAGE_TOKEN_INDEX (-200) for decoding
    text_only_ids = input_ids[0][input_ids[0] != IMAGE_TOKEN_INDEX]
    input_text = tokenizer.decode(text_only_ids, skip_special_tokens=True) # system prompt + IMAGE_TOKEN_INDEX + query text
    print(f"Input text (decoded): '{input_text}'") # 'You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language. car'
    print(f"Input length (with image placeholder): {input_len} tokens") # 40
    print(f"========================\n")
    
    gen_ids = sequences
    generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    attn_last_to_vis = None
    # Print input token length

    begin_pos_vis = MetadataStation.get_begin_pos('vis')
    vis_len = MetadataStation.get_vis_len()
    total_input_len = input_len + vis_len
    print(f"Total input length (text + image tokens): {total_input_len}") # 616
    print(f"Text token length: {input_len}") # 40
    print(f"Image token length: {vis_len}") # 576

    print(f"gen.keys(): {gen.keys()}") # ['sequences', 'attentions', 'past_key_values']
    print(f"gen.sequences: {gen.sequences}") # [1, 10]

    idx = None
    for i in range(len(gen.sequences[0])):
        print(f"gen.sequences[{i}]: {gen.sequences[0][i]}, decoded: {tokenizer.decode(gen.sequences[0][i])}")
        if gen.sequences[0][i] == 3593: # bus
            idx = i
            
    if idx is None:
        raise RuntimeError("Generated text not found.")

    print(f"generated text: {generated_text}")
    print(f"len gen.attentions: {len(gen.attentions)}") # 10
    print(f"len of gen.attentions[0]: {len(gen.attentions[0])}") # 32
    print(f"shape of gen.attentions[0][0]: {gen.attentions[0][0].shape}") # [1, 32, 615, 615]
    # print(f"gen.sequences[0, input_len:]: {gen.sequences[0, input_len:]}") #

    print(f"len of gen.attentions[1]: {len(gen.attentions[1])}") # 32
    print(f"shape of gen.attentions[1][0]: {gen.attentions[1][0].shape}") # [1, 32, 1, 616]
    # print(f"gen.sequences[0, input_len + 1]: {gen.sequences[0, input_len + 1]}") # 616

    if hasattr(gen, 'attentions') and gen.attentions:   
        step_idx = gen.attentions[idx+1]  # [B,H,1,src]
        print(f"shape of step_idx[0]: {step_idx[0].shape}") # [1, 32, 1, 616]
        print(f"shape of step_idx[0][0]: {step_idx[0][0].shape}") # [32, 1, 616]
        layers = [t[0] for t in step_idx]  # list of [H,1,src]

        attn = torch.stack(layers, dim=0)  # [L,H,1,src]
        print(f"shape of attn: {attn.shape}") # [32, 32, 1, 629] 629 = 576 + 53
        # print(f"attn: {attn}")
        begin_pos_vis = MetadataStation.get_begin_pos('vis')
        print(f"begin_pos_vis: {begin_pos_vis}") 
        print(f"vis_len: {vis_len}") # 576
        vis_len = MetadataStation.get_vis_len()
        if begin_pos_vis is None or vis_len is None:
            raise RuntimeError("Missing visual token segmentation info.")
        attn_last_to_vis = attn[:, :, -1:, begin_pos_vis:begin_pos_vis + vis_len]
        
        masked_attn_vis = attn[-1:, -1:, :, :]
        print(f"shape of masked_attn_vis: {masked_attn_vis.shape}") # [1, 1, 1, input_len+vis_len+idx]
        print(f"masked_attn_vis: {masked_attn_vis[:,:,:,-10:]}")
        
        print(f"shape of attn_last_to_vis: {attn_last_to_vis.shape}") # [32, 32, 1, 576]
    return attn_last_to_vis, generated_text


def collect_attention(cfg, image_file: str, query: str, save_dir: str, save_id: str) -> str:
    """Run one forward pass and save attention focused on image tokens.

    Saves a pickle with dict: {
      'attn': Tensor[L, H, 1, V],
      'meta': {image_file, query, image_size, model_name, vis_len, patch_size, num_layers, num_heads}
    }
    Returns the saved file path.
    """
    tokenizer, model, image_processor, _, model_name_str = load_model_from_cfg(cfg)

    # Prepare image
    image = load_image(image_file)
    image_size = image.size  # (W, H)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    image_sizes = [image.size]

    # Prepare prompt
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in query:
        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, query) if model.config.mm_use_im_start_end else re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
    else:
        qs = (image_token_se + "\n" + query) if model.config.mm_use_im_start_end else (DEFAULT_IMAGE_TOKEN + "\n" + query)

    conv = conv_templates[cfg.model.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
        conv=conv,
    ).unsqueeze(0).to(model.device)

    # Collect attentions (and optional generated text)
    with torch.inference_mode():
        if getattr(cfg.model, 'use_generate', False):
            attn_last_to_vis, gen_text = _generate_collect(
                model, tokenizer, image_processor, input_ids, image_tensor, image_sizes,
                max_new_tokens=getattr(cfg.model, 'max_new_tokens', 10),
                do_sample=getattr(cfg.model, 'do_sample', False),
                num_beams=getattr(cfg.model, 'num_beams', 1),
            )
            if attn_last_to_vis is None:
                attn_last_to_vis = _forward_collect(
                    model, tokenizer, image_processor, input_ids, image_tensor, image_sizes
                )
        else:
            gen_text = None
            attn_last_to_vis = _forward_collect(
                model, tokenizer, image_processor, input_ids, image_tensor, image_sizes
            )

    P = int(np.sqrt(attn_last_to_vis.shape[-1]))
    meta = {
        "image_file": image_file,
        "query": query,
        "image_size": image_size,
        "model_name": model_name_str,
        "vis_len": int(attn_last_to_vis.shape[-1]),
        "patch_size": int(P),
        "num_layers": int(attn_last_to_vis.shape[0]),
        "num_heads": int(attn_last_to_vis.shape[1]),
    }
    if getattr(cfg.model, 'use_generate', False) and gen_text is not None:
        meta["generated_text"] = gen_text

    model_dir = _sanitize_name(cfg.model.name)
    out_dir = os.path.join(save_dir, model_dir)
    _ensure_dir(out_dir)

    save_path = os.path.join(out_dir, f"{save_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"attn": attn_last_to_vis.detach().cpu(), "meta": meta}, f)

    # Save a small side metadata for convenience
    with open(os.path.join(out_dir, f"{save_id}_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    return save_path
