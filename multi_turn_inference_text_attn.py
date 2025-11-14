import os
import re
import pickle
from typing import Dict, Tuple

import torch
import numpy as np
from PIL import Image
from omegaconf import DictConfig, OmegaConf

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
from lab.prompts import PromptTemplates, PromptConfig

from analyze import load_attention_file, analyze_heads
from viz import plot_heads_grid, save_all_heads, save_all_heads_grid

from typing import Optional
import json

def _sanitize(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_")


def _model_dir(cfg: DictConfig) -> str:
    return _sanitize(cfg.model.name)


def _out_root(cfg) -> str:
    return cfg.data.output_dir

def _replace_image_placeholder(text: str, model_config) -> str:
    """Replace IMAGE_PLACEHOLDER with actual image tokens based on model config"""
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if IMAGE_PLACEHOLDER in text:
        # Replace placeholder with appropriate image token
        if model_config.mm_use_im_start_end:
            return re.sub(IMAGE_PLACEHOLDER, image_token_se, text)
        else:
            return re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, text)
    return text


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

    print(f"Model embedding size: {model.get_input_embeddings().weight.shape[0]}")
    print(f"Tokenizer vocab size: {len(tok)}")
    return tok, model, img_proc, context_len, model_name_str

def _analyze_attn(cfg: DictConfig, attn_file, stage_name: str, save_id: str):
    """Analyze attention and save results to image-specific directory"""
    os.makedirs(_out_root(cfg), exist_ok=True)
    model_dir = _model_dir(cfg)
    # Create image-specific subdirectory
    attn_root = os.path.join(_out_root(cfg), model_dir, save_id)
    os.makedirs(attn_root, exist_ok=True)

    # Stage: analyze
    attn, meta = load_attention_file(attn_file)
    selected = analyze_heads(cfg, attn, meta)

    # Save analysis (simplified filename)
    with open(attn_file.replace('.pkl', '_analysis.pkl'), 'wb') as f:
        pickle.dump(selected, f)

    # Visualization (simplified filename)
    if cfg.save_fig:
        fig_path = os.path.join(attn_root, f"{stage_name}_top{cfg.logic.top_k}.png")
        plot_heads_grid(attn, selected[: cfg.logic.top_k], meta, fig_path, show_plot=cfg.show_plot)

    # Save all heads if enabled
    if cfg.get("save_all_heads", False):
        save_all_heads(attn, meta, attn_root, stage_name)
        grid_path = os.path.join(attn_root, f"{stage_name}_all_heads_overview.png")
        save_all_heads_grid(attn, meta, grid_path, heads_per_row=8)


def _prepare_prompt_and_tokenize_from_messages(messages: list, tokenizer, cfg, model) -> torch.Tensor:
    """Prepare prompt from messages list and tokenize

    Args:
        messages: List of {"role": "user"/"assistant", "content": str}
        tokenizer: Model tokenizer
        cfg: Config object
        model: Model instance

    Returns:
        input_ids tensor
    """
    # Use conversation template to format messages
    conv = conv_templates[cfg.model.conv_mode].copy()

    # Add all messages to conversation
    # Only the first user message should have image token
    for i, msg in enumerate(messages):
        role = conv.roles[0] if msg["role"] == "user" else conv.roles[1]
        content = msg["content"]

        # Replace IMAGE_PLACEHOLDER with actual image tokens (ONLY for first user message)
        if i == 0 and msg["role"] == "user":
            content = _replace_image_placeholder(content, model.config)

        conv.append_message(role, content)

    # Add None for assistant's turn (generation target)
    if messages[-1]["role"] == "user":
        conv.append_message(conv.roles[1], None)

    # Get formatted prompt
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX,
        return_tensors="pt", conv=conv
    ).unsqueeze(0).to(model.device)

    return input_ids


def _generate_with_attention(stage, model, tokenizer, input_ids: torch.Tensor,
                            image_tensor: Optional[torch.Tensor] = None,
                            image_sizes: Optional[list] = None,
                            max_new_tokens: int = 150,
                            do_sample: bool = False,
                            num_beams: int = 1,
                            vis_token_pos: Optional[Tuple[int, int]] = None,
                            **generation_kwargs) -> Tuple[str, Optional[torch.Tensor], Optional[Tuple[int, int]]]:
    """Generate text and collect attention weights

    Args:
        image_tensor: Image tensor. Should be provided ONLY in the first turn.
                      For subsequent turns, pass None to reuse K-V cached visual features.
        vis_token_pos: Tuple of (begin_pos, vis_len) for visual tokens.
                       If None, will try to get from MetadataStation.
                       If provided, use these positions to extract visual attention.
        **generation_kwargs: Additional arguments for model.generate() (e.g., temperature, top_p, etc.)

    Returns:
        (generated_text, attn_last_to_vis, vis_token_pos)
        vis_token_pos is returned so it can be reused in subsequent turns
    """
    # Prepare images argument
    # First turn: pass image tensor to encode visual features
    # Subsequent turns: pass None to reuse K-V cached features
    if image_tensor is not None:
        images_arg = image_tensor.unsqueeze(0)
    else:
        images_arg = None

    gen = model.generate(
        inputs=input_ids,
        images=images_arg,  # None for turns 2+, reuses K-V cache
        image_sizes=image_sizes,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        return_dict_in_generate=True,
        output_attentions=True,
        **generation_kwargs,  # Pass additional generation arguments
    )


    sequences = gen.sequences
    generated_text = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]

    # Find the index of the generated text
    ## len(sequences[0] == len(gen.attentions)
    if stage == "stage1" or stage == "stage2":
        idx = None
        len_sequences = len(sequences[0])
        # print(f"sequences[0][0]: {sequences[0][0]}")
        # print(f"sequences[0][1]: {sequences[0][1]}")
        # print(f"sequences[0][-2]: {sequences[0][-2]}")
        # print(f"sequences[0][-1]: {sequences[0][-1]}")

        # print(f"decoded sequences[0][0]: {tokenizer.decode(sequences[0][0])}")
        # print(f"decoded sequences[0][1]: {tokenizer.decode(sequences[0][1])}")
        # print(f"decoded sequences[0][{len_sequences-2}]: {tokenizer.decode(sequences[0][len_sequences-2])}")
        # print(f"decoded sequences[0][{len_sequences-1}]: {tokenizer.decode(sequences[0][len_sequences-1])}")
        for i in range(len(sequences[0])):
            # print(f"gen.sequences[{i}]: {gen.sequences[0][i]}, decoded: {tokenizer.decode(gen.sequences[0][i])}")
            if sequences[0][i] == 7933 and idx is None:
                idx = i
                print(f"idx: {idx}")
                print(f"decoded idx-1: {tokenizer.decode(sequences[0][idx-1])}")
                print(f"decoded idx: {tokenizer.decode(sequences[0][idx])}")
                print(f"decoded idx+1: {tokenizer.decode(sequences[0][idx+1])}")
        if idx is None:
            print("⚠️ Generated text not found.")
            idx = 0
    else:
        idx = 0
    # idx = -1

    attn_last_to_vis = None

    # Get visual token position
    if vis_token_pos is None:
        # First turn: get from MetadataStation
        begin_pos_vis = MetadataStation.get_begin_pos('vis')
        vis_len = MetadataStation.get_vis_len()
        if begin_pos_vis is not None and vis_len is not None:
            vis_token_pos = (begin_pos_vis, vis_len)
    else:
        # Subsequent turns: use cached position
        begin_pos_vis, vis_len = vis_token_pos

    # Collect visual attention using the position
    if vis_token_pos is not None and hasattr(gen, 'attentions') and gen.attentions:
        begin_pos_vis, vis_len = vis_token_pos
        print(f"len of gen.attentions: {len(gen.attentions)}")
        print(f"len input_ids: {input_ids.shape}") # 1 x 1024
        print(f"begin_pos_vis: {begin_pos_vis}")
        print(f"vis_len: {vis_len}")
        step_idx = gen.attentions[idx]
        print(f"len of step_idx: {len(step_idx)}") # 32
        print(f"shape of step_idx[0]: {step_idx[0].shape}") # 1 x 32 x 1252 x 1252
        print(f"shape of step_idx[0][0]: {step_idx[0][0].shape}") # 32 x 1252 x 1252
        layers = [t[0] for t in step_idx]
        attn = torch.stack(layers, dim=0)
        
        print(f"shape of attn: {attn.shape}") # 32 x 32 x 1 x 1252
        # 각 헤드별로 값이 가장 높은 key의 위치를 값으로 갖는 32 x 32 행렬로
        # attn: [num_layers, num_heads, 1, num_keys]
        # attn_max_idx_by_head = attn.argmax(dim=-1)  # shape: [num_layers, num_heads, 1]
        # attn_max_idx_by_head_matrix = attn_max_idx_by_head.squeeze(-1)  # shape: [32, 32]
        # print(f"attn_max_idx_by_head_matrix.shape: {attn_max_idx_by_head_matrix.shape}")
        # print(f"attn_max_idx_by_head_matrix: {attn_max_idx_by_head_matrix}")  # shape: [32, 32]

        # Extract attention to visual tokens
        # NOTE: In subsequent turns, K-V cache contains visual tokens from first turn
        # system prompt + visual tokens + user prompt = 34 + 576 + 107
        attn_last_to_vis = attn[:, :, -1:, begin_pos_vis:begin_pos_vis + vis_len] # 32 x 32 x 1 x 722(34+576)
        attn_last_to_text = attn[:, :, -1:, begin_pos_vis + vis_len:] # 32 x 32 x 1 x 107
        attn_last_to_sys = attn[:, :, -1:, :begin_pos_vis] # 32 x 32 x 1 x 34

    return generated_text, attn_last_to_vis, vis_token_pos,  attn_last_to_sys, attn_last_to_text, begin_pos_vis, vis_len


def _save_stage_result(
    save_dir: str,
    model_dir: str,
    save_id: str,
    stage_name: str,
    attn: Optional[torch.Tensor],
    meta: dict
) -> str:
    """Save results for each stage consistently"""
    # Create image-specific subdirectory: outputs/results/model_name/image_name/
    out_dir = os.path.join(save_dir, model_dir, save_id)
    _ensure_dir(out_dir)

    # Simplified filename (no need to repeat save_id in filename)
    stage_filename = f"{stage_name}.pkl"
    save_path = os.path.join(out_dir, stage_filename)

    result = {
        "attn": attn.detach().cpu() if attn is not None else None,
        "meta": meta
    }

    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    return save_path


# ═══════════════════════════════════════════════════════
# Multi-turn Inference Main Function
# ═══════════════════════════════════════════════════════

def multi_turn_inference_text_attn(
    cfg,
    image_file: str,
    save_dir: str,
    save_id: str
) -> Dict:
    """
    Sequentially perform 3-step multi-turn inference

    Stage 1: Scene Description - Generate image description
    Stage 2: Scene Analysis - Analyze based on description
    Stage 3: Planning - Plan based on analysis

    Args:
        cfg: Hydra config
        image_file: image file path
        save_dir: result output directory
        save_id: save file ID

    Returns:
        Dict with all stage results and file paths
    """

    # ═══════════════════════════════════════════════════════
    # 1️⃣ Initialization Stage (Setup Phase)
    # ═══════════════════════════════════════════════════════
    print(f"[Multi-turn] Loading model...")
    tokenizer, model, image_processor, _, model_name_str = load_model_from_cfg(cfg)

    # Load and preprocess image
    print(f"[Multi-turn] Loading image: {image_file}")
    image = load_image(image_file)
    image_size = image.size  # (W, H)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    image_sizes = [image.size]

    # Set output directory - Create image-specific subdirectory
    model_dir = _sanitize_name(cfg.model.name)
    out_dir = os.path.join(save_dir, model_dir, save_id)
    _ensure_dir(out_dir)

    # Result storage dict
    results = {
        "model_name": model_name_str,
        "image_file": image_file,
        "image_size": image_size,
        "save_id": save_id
    }

    # ═══════════════════════════════════════════════════════
    # Multi-turn conversation history
    # ═══════════════════════════════════════════════════════
    messages = []
    vis_token_pos = None  # Visual token position cache (reuse from K-V cache)

    with torch.inference_mode():
        try:
            # ═══════════════════════════════════════════════════════
            # 2️⃣ Stage 1: Scene Description
            # ═══════════════════════════════════════════════════════
            print(f"[Multi-turn] Stage 1: Scene Description")

            # Prepare prompt - add first user message to messages
            query_stage1 = PromptTemplates.format_scene_description(
                image_placeholder=IMAGE_PLACEHOLDER
            )
            messages.append({"role": "user", "content": query_stage1})

            # Tokenize from messages
            input_ids_stage1 = _prepare_prompt_and_tokenize_from_messages(
                messages, tokenizer, cfg, model
            )

            print(f"Stage 1 input_ids max value: {input_ids_stage1.max().item()}")

            # Run inference (first turn: provide image_tensor to generate visual features)
            scene_desc_text, attn_stage1, vis_token_pos, attn_last_to_sys, attn_last_to_text, begin_pos_vis, vis_len = _generate_with_attention(
                "stage1",
                model, tokenizer, input_ids_stage1,
                image_tensor=image_tensor,  # Only provide first turn
                image_sizes=image_sizes,
                vis_token_pos=None,  # None in the first turn
                **PromptConfig.SCENE_DESCRIPTION
            )
            
            # Add assistant response to messages
            messages.append({"role": "assistant", "content": scene_desc_text})

            print(f"Scene Description: {scene_desc_text[:100]}...")

            # Compose metadata
            P = int(np.sqrt(attn_stage1.shape[-1])) if attn_stage1 is not None else 0
            meta_stage1 = {
                "stage": "scene_description",
                "generated_text": scene_desc_text,
                "image_file": image_file,
                "image_size": image_size,
                "model_name": model_name_str,
                "vis_len": int(attn_stage1.shape[-1]) if attn_stage1 is not None else 0,
                "patch_size": P,
                "num_layers": int(attn_stage1.shape[0]) if attn_stage1 is not None else 0,
                "num_heads": int(attn_stage1.shape[1]) if attn_stage1 is not None else 0,
                "prompt": query_stage1,
                "conversation_history": messages.copy(),
            }

            # Save result
            stage1_path = _save_stage_result(
                save_dir, model_dir, save_id, "stage1_description",
                attn_stage1, meta_stage1
            )

            print(f"attn_last_to_sys: {attn_last_to_sys.shape}")
            print(f"attn_last_to_text: {attn_last_to_text.shape}")
            print(f"begin_pos_vis: {begin_pos_vis}")
            print(f"vis_len: {vis_len}")
            
            # sys
            avg_sys = attn_last_to_sys.mean(dim=(0, 1)).squeeze()
            top_scores, top_indices = torch.topk(avg_sys, k=10)
            top_scores = top_scores.cpu().tolist()
            top_indices = top_indices.cpu().tolist()
            sys_results = []
            print(f"shape of input_ids_stage1: {input_ids_stage1.shape}")
            for score, idx in zip(top_scores, top_indices):
                token_id = input_ids_stage1[0, idx].item()
                # convert_ids_to_tokens takes a list
                token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
                
                # Replace the '▁' (U+2581) symbol from SentencePiece tokenizer with a whitespace character
                token_str = token_str.replace(' ', ' ') 
                
                sys_results.append((idx, token_str, score))
            print("Top Attended SYS Tokens:")
            print("\n".join([f" Index: {i:<15} | Token: {t:<15} | Score: {s:.4f}" for i, t, s in sys_results]))

            # text
            avg_text = attn_last_to_text.mean(dim=(0, 1)).squeeze()
            top_scores, top_indices = torch.topk(avg_text, k=10)
            top_scores = top_scores.cpu().tolist()
            top_indices = top_indices.cpu().tolist()
            text_results = []
            for score, idx in zip(top_scores, top_indices):
                print(f"top_scores: {score}")
                print(f"top_indices: {idx}")
                token_id = input_ids_stage1[0, idx+begin_pos_vis].item()
                token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
                token_str = token_str.replace(' ', ' ') 
                text_results.append((idx, token_str, score))
            print("Top Attended TEXT Tokens:")
            print("\n".join([f" Index: {i:<15} | Token: {t:<15} | Score: {s:.4f}" for i, t, s in text_results]))

            _analyze_attn(cfg, stage1_path, "stage1_description", save_id)

            results["stage1"] = {
                "text": scene_desc_text,
                "attn_shape": list(attn_stage1.shape) if attn_stage1 is not None else None,
                "save_path": stage1_path,
                "meta": meta_stage1
            }

        except Exception as e:
            print(f"[Multi-turn] Error in Stage 1: {e}")
            results["stage1"] = {"error": str(e)}
            raise

        try:
            # ═══════════════════════════════════════════════════════
            # 3️⃣ Stage 2: Scene Analysis
            # ═══════════════════════════════════════════════════════
            print(f"[Multi-turn] Stage 2: Scene Analysis")

            # Prepare prompt - add next user message to messages
            # (previous conversation history is already in messages)
            query_stage2 = PromptTemplates.SCENE_ANALYSIS
            messages.append({"role": "user", "content": query_stage2})

            # Tokenize from messages (with conversation history)
            input_ids_stage2 = _prepare_prompt_and_tokenize_from_messages(
                messages, tokenizer, cfg, model
            )

            # Run inference (second turn: image_tensor=None, reuse K-V cache)
            analysis_text, attn_stage2, _ = _generate_with_attention(
                "stage2",
                model, tokenizer, input_ids_stage2,
                image_tensor=image_tensor, # None,  # Reuse K-V cache's visual features
                image_sizes=image_sizes, # None,
                vis_token_pos=vis_token_pos,  # Reuse visual token position
                **PromptConfig.SCENE_ANALYSIS
            )

            print(f"Stage 2 attn_stage2: {attn_stage2.shape}")
            # Add assistant response to messages
            messages.append({"role": "assistant", "content": analysis_text})

            print(f"[Multi-turn] Stage 2 output: {analysis_text[:100]}...")

            # Compose metadata
            P = int(np.sqrt(attn_stage2.shape[-1])) if attn_stage2 is not None else 0
            meta_stage2 = {
                "stage": "scene_analysis",
                "generated_text": analysis_text,
                "image_file": image_file,
                "image_size": image_size,
                "model_name": model_name_str,
                "vis_len": int(attn_stage2.shape[-1]) if attn_stage2 is not None else 0,
                "patch_size": P,
                "num_layers": int(attn_stage2.shape[0]) if attn_stage2 is not None else 0,
                "num_heads": int(attn_stage2.shape[1]) if attn_stage2 is not None else 0,
                "prompt": query_stage2,
                "conversation_history": messages.copy(),
            }

            # Save result
            stage2_path = _save_stage_result(
                save_dir, model_dir, save_id, "stage2_analysis",
                attn_stage2, meta_stage2
            )

            print(f"attn_last_to_sys: {attn_last_to_sys.shape}")
            print(f"attn_last_to_text: {attn_last_to_text.shape}")
            print(f"begin_pos_vis: {begin_pos_vis}")
            print(f"vis_len: {vis_len}")
            
            # sys
            avg_sys = attn_last_to_sys.mean(dim=(0, 1)).squeeze()
            top_scores, top_indices = torch.topk(avg_sys, k=10)
            top_scores = top_scores.cpu().tolist()
            top_indices = top_indices.cpu().tolist()
            sys_results = []
            print(f"shape of input_ids_stage2: {input_ids_stage2.shape}")

            for score, idx in zip(top_scores, top_indices):
                token_id = input_ids_stage2[0, idx].item()
                # convert_ids_to_tokens는 리스트를 받습니다
                token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
                
                # SentencePiece 토크나이저의 ' ' (U+2581) 기호를 공백으로 변환
                token_str = token_str.replace(' ', ' ') 
                
                sys_results.append((idx, token_str, score))
            print("Top Attended SYS Tokens:")
            print("\n".join([f" Index: {i:<15} | Token: {t:<15} | Score: {s:.4f}" for i, t, s in sys_results]))

            # text
            avg_text = attn_last_to_text.mean(dim=(0, 1)).squeeze()
            top_scores, top_indices = torch.topk(avg_text, k=10)
            top_scores = top_scores.cpu().tolist()
            top_indices = top_indices.cpu().tolist()
            text_results = []
            for score, idx in zip(top_scores, top_indices):
                token_id = input_ids_stage2[0, idx-begin_pos_vis].item()
                token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
                token_str = token_str.replace(' ', ' ') 
                text_results.append((idx, token_str, score))
            print("Top Attended TEXT Tokens:")
            print("\n".join([f" Index: {i:<15} | Token: {t:<15} | Score: {s:.4f}" for i, t, s in text_results]))

            results["stage2"] = {
                "text": analysis_text,
                "attn_shape": list(attn_stage2.shape) if attn_stage2 is not None else None,
                "save_path": stage2_path,
                "meta": meta_stage2
            }

            _analyze_attn(cfg, stage2_path, "stage2_analysis", save_id)

        except Exception as e:
            print(f"[Multi-turn] Error in Stage 2: {e}")
            results["stage2"] = {"error": str(e)}
            # Even if Stage 2 fails, Stage 1 results are preserved

        try:
            # ═══════════════════════════════════════════════════════
            # 4️⃣ Stage 3: Planning
            # ═══════════════════════════════════════════════════════
            print(f"[Multi-turn] Stage 3: Planning")

            # Prepare prompt - add next user message to messages
            # (conversation history from Stage 1, 2 already in messages)
            query_stage3 = PromptTemplates.PLANNING
            messages.append({"role": "user", "content": query_stage3})

            # Tokenize from messages (with full conversation history)
            input_ids_stage3 = _prepare_prompt_and_tokenize_from_messages(
                messages, tokenizer, cfg, model
            )

            # Run inference (third turn: image_tensor=None, reuse K-V cache)
            planning_text, attn_stage3, _ = _generate_with_attention(
                "stage3",
                model, tokenizer, input_ids_stage3,
                image_tensor=image_tensor, # None,  # Reuse K-V cache's visual features
                image_sizes=image_sizes, # None,
                vis_token_pos=vis_token_pos,  # Reuse visual token position
                **PromptConfig.PLANNING
            )
            # Add assistant response to messages
            messages.append({"role": "assistant", "content": planning_text})

            print(f"[Multi-turn] Stage 3 output: {planning_text[:100]}...")

            # Compose metadata
            P = int(np.sqrt(attn_stage3.shape[-1])) if attn_stage3 is not None else 0
            meta_stage3 = {
                "stage": "planning",
                "generated_text": planning_text,
                "image_file": image_file,
                "image_size": image_size,
                "model_name": model_name_str,
                "vis_len": int(attn_stage3.shape[-1]) if attn_stage3 is not None else 0,
                "patch_size": P,
                "num_layers": int(attn_stage3.shape[0]) if attn_stage3 is not None else 0,
                "num_heads": int(attn_stage3.shape[1]) if attn_stage3 is not None else 0,
                "prompt": query_stage3,
                "conversation_history": messages.copy(),
            }

            # Save result
            stage3_path = _save_stage_result(
                save_dir, model_dir, save_id, "stage3_planning",
                attn_stage3, meta_stage3
            )

            results["stage3"] = {
                "text": planning_text,
                "attn_shape": list(attn_stage3.shape) if attn_stage3 is not None else None,
                "save_path": stage3_path,
                "meta": meta_stage3
            }

            _analyze_attn(cfg, stage3_path, "stage3_planning", save_id)

        except Exception as e:
            print(f"[Multi-turn] Error in Stage 3: {e}")
            results["stage3"] = {"error": str(e)}

    # ═══════════════════════════════════════════════════════
    # 5️⃣ Save aggregated results
    # ═══════════════════════════════════════════════════════

    # Save the full results as a single file (simplified filename)
    multi_turn_path = os.path.join(out_dir, "multi_turn.pkl")
    with open(multi_turn_path, "wb") as f:
        pickle.dump(results, f)

    # Save human-readable text summary JSON (simplified filename)
    summary = {
        "model": model_name_str,
        "image": image_file,
        "conversation_history": messages,  # Full conversation history
        "scene_description": results.get("stage1", {}).get("text", "N/A"),
        "scene_analysis": results.get("stage2", {}).get("text", "N/A"),
        "planning": results.get("stage3", {}).get("text", "N/A"),
    }

    summary_path = os.path.join(out_dir, "multi_turn_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    results["multi_turn_file"] = multi_turn_path
    results["summary_file"] = summary_path

    print(f"[Multi-turn] Complete! Saved to:")
    print(f"  - Multi-turn: {multi_turn_path}")
    print(f"  - Summary: {summary_path}")

    return results