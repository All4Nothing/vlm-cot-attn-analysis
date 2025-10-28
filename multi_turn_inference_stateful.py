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
from lab.prompts import PromptTemplates, PromptConfig

from typing import Optional
import json


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


def _generate_with_attention(model, tokenizer, input_ids: torch.Tensor, 
                            image_tensor: Optional[torch.Tensor] = None, 
                            image_sizes: Optional[list] = None, 
                            max_new_tokens: int = 150, 
                            do_sample: bool = False, 
                            num_beams: int = 1, 
                            vis_token_pos: Optional[Tuple[int, int]] = None,
                            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, # [수정] K-V 캐시 인자 추가
                            **generation_kwargs) -> Tuple[str, Optional[torch.Tensor], Optional[Tuple[int, int]], Optional[Tuple[Tuple[torch.Tensor]]]]:
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
        past_key_values=past_key_values,    # [수정] K-V 캐시 전달
        use_cache=True,     # [수정] K-V 캐시 전달
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        return_dict_in_generate=True,
        output_attentions=True,
        **generation_kwargs,  # Pass additional generation arguments
    )
    
    # sequences = gen.sequences
    output_ids = gen.sequences[:, input_ids.shape[1]:]
    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    
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
        step_idx = gen.attentions[-1]
        layers = [t[0] for t in step_idx]  # list of [H,1,src]
        attn = torch.stack(layers, dim=0)  # [L,H,1,src]
        
        # Extract attention to visual tokens
        # NOTE: In subsequent turns, K-V cache contains visual tokens from first turn
        attn_last_to_vis = attn[:, :, -1:, begin_pos_vis:begin_pos_vis + vis_len]
    
    return generated_text, attn_last_to_vis, vis_token_pos, gen.past_key_values # [수정] K-V 캐시 반환


def _save_stage_result(
    save_dir: str,
    model_dir: str,
    save_id: str,
    stage_name: str,
    attn: Optional[torch.Tensor],
    meta: dict
) -> str:
    """각 stage 결과를 일관되게 저장"""
    out_dir = os.path.join(save_dir, model_dir)
    _ensure_dir(out_dir)
    
    stage_filename = f"{save_id}_{stage_name}.pkl"
    save_path = os.path.join(out_dir, stage_filename)
    
    result = {
        "attn": attn.detach().cpu() if attn is not None else None,
        "meta": meta
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(result, f)
    
    return save_path


# ═══════════════════════════════════════════════════════
# Multi-turn Inference 메인 함수
# ═══════════════════════════════════════════════════════

def multi_turn_inference(
    cfg,
    image_file: str,
    save_dir: str,
    save_id: str
) -> Dict:
    """
    3단계 multi-turn inference를 순차적으로 수행
    
    Stage 1: Scene Description - 이미지 설명 생성
    Stage 2: Scene Analysis - 설명을 바탕으로 분석
    Stage 3: Planning - 분석을 바탕으로 계획 수립
    
    Args:
        cfg: Hydra config
        image_file: 이미지 파일 경로
        save_dir: 결과 저장 디렉토리
        save_id: 저장 파일 ID
    
    Returns:
        Dict with all stage results and file paths
    """
    
    # ═══════════════════════════════════════════════════════
    # 1️⃣ 초기화 단계 (Setup Phase)
    # ═══════════════════════════════════════════════════════
    print(f"[Multi-turn] Loading model...")
    tokenizer, model, image_processor, _, model_name_str = load_model_from_cfg(cfg)
    
    # 이미지 로드 및 전처리
    print(f"[Multi-turn] Loading image: {image_file}")
    image = load_image(image_file)
    image_size = image.size  # (W, H)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    image_sizes = [image.size]
    
    # 출력 디렉토리 설정
    model_dir = _sanitize_name(cfg.model.name)
    out_dir = os.path.join(save_dir, model_dir)
    _ensure_dir(out_dir)
    
    # 결과 저장용 딕셔너리
    results = {
        "model_name": model_name_str,
        "image_file": image_file,
        "image_size": image_size,
        "save_id": save_id
    }
    
    # ═══════════════════════════════════════════════════════
    # 대화 이력 관리 (Multi-turn conversation history)
    # ═══════════════════════════════════════════════════════
    messages = []
    vis_token_pos = None  # Visual token 위치 캐시 (K-V cache에서 재사용)
    past_key_values = None # [추가] K-V 캐시 상태 변수
    
    # [추가] Conversation 객체를 루프 밖에서 생성
    conv = conv_templates[cfg.model.conv_mode].copy()
    
    with torch.inference_mode():
        try:
            # ═══════════════════════════════════════════════════════
            # 2️⃣ Stage 1: Scene Description
            # ═══════════════════════════════════════════════════════
            print(f"[Multi-turn] Stage 1: Scene Description")
            
            # 프롬프트 준비 - messages에 첫 user 메시지 추가
            query_stage1 = PromptTemplates.format_scene_description(
                image_placeholder=IMAGE_PLACEHOLDER
            )
            query_stage1 = _replace_image_placeholder(query_stage1, model.config)
            conv.append_message(conv.roles[0], query_stage1)
            conv.append_message(conv.roles[1], None)
            
            # [수정] _prepare_... 함수 대신 직접 토크나이징
            prompt_stage1 = conv.get_prompt()
            input_ids_stage1 = tokenizer_image_token(
                prompt_stage1, tokenizer, IMAGE_TOKEN_INDEX, 
                return_tensors="pt", conv=conv
            ).unsqueeze(0).to(model.device)

            print(f"Stage 1 input_ids max value: {input_ids_stage1.max().item()}")
            
            attention_mask = None
            position_ids = None

            # Inference 실행 (첫 턴: image_tensor 전달하여 visual features 생성)
            # [수정] _generate_with_attention 호출 (K-V 캐시 반환)
            scene_desc_text, attn_stage1, vis_token_pos, past_key_values = _generate_with_attention(
                model, tokenizer, input_ids_stage1, 
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_tensor=image_tensor,  # 첫 턴에만 전달
                image_sizes=image_sizes,
                vis_token_pos=None,
                past_key_values=None, # 첫 턴이므로 None
                **PromptConfig.SCENE_DESCRIPTION
            )
            
            # [★추가★] K-V 캐시 구조 및 시퀀스 길이 확인
            if past_key_values is not None:
                print(f"KV Cache structure: (num_layers={len(past_key_values)}, 2, batch, heads, seq_len, head_dim)")
                # 첫 번째 레이어의 key 텐서 모양 확인 (보통 key와 value 모양 동일)
                first_layer_key_shape = past_key_values[0][0].shape 
                print(f"First layer Key tensor shape: {first_layer_key_shape}")
                cached_sequence_length = first_layer_key_shape[3] # LLaVA 구현에 따라 인덱스가 다를 수 있음 (보통 2 또는 3)
                print(f"Cached sequence length from Stage 1: {cached_sequence_length}") 
            else:
                print("Stage 1 did not return K-V cache.")

            # [수정] conv 객체와 messages 로그 모두 업데이트
            conv.messages[-1][1] = scene_desc_text # conv의 'None'을 실제 응답으로 교체
            messages.append({"role": "user", "content": query_stage1})
            messages.append({"role": "assistant", "content": scene_desc_text})
            
            print(f"Scene Description: {scene_desc_text}")
            
            # 메타데이터 구성
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
                "conversation_history": messages.copy(),  # 대화 이력 포함
            }
            
            # 결과 저장
            stage1_path = _save_stage_result(
                save_dir, model_dir, save_id, "stage1_description",
                attn_stage1, meta_stage1
            )
            
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
            
            # 프롬프트 준비 - messages에 다음 user 메시지 추가
            # (이전 대화 이력은 messages에 이미 포함되어 있음)
            query_stage2 = PromptTemplates.SCENE_ANALYSIS

            # [수정] conv 객체에 새 턴 추가
            # conv.append_message(conv.roles[0], query_stage2)
            # conv.append_message(conv.roles[1], None)

            full_prompt_stage2 = conv.get_prompt()

            # [★수정★] Stage 2 이후에도 tokenizer_image_token 사용
            # 이 함수가 이미지 토큰이 없더라도 텍스트 부분을 올바르게 처리해야 함
            full_input_ids_stage2 = tokenizer_image_token(
                full_prompt_stage2, 
                tokenizer, 
                IMAGE_TOKEN_INDEX, # 이 인자가 여전히 필요할 수 있음
                return_tensors="pt", 
                conv=conv # conv 객체 전달 필요할 수 있음
            ).unsqueeze(0).to(model.device)

            # K-V 캐시 길이 가져오기
            if past_key_values is not None:
                cached_sequence_length = past_key_values[0][0].shape[2] 
            else: 
                cached_sequence_length = 0 

            # 슬라이싱 (이제 full_input_ids 길이가 cache 길이보다 커야 함)
            input_ids_stage2_new_tokens = full_input_ids_stage2[:, cached_sequence_length:]
            
            # (Optional) 슬라이싱된 attention mask 준비 - 필요하면 추가
            # attention_mask_stage2 = torch.ones_like(input_ids_stage2_new_tokens)

            print(f"Stage 2 FULL input_ids shape: {full_input_ids_stage2.shape}") 
            print(f"Stage 2 NEW TOKENS input_ids shape: {input_ids_stage2_new_tokens.shape}") 
            print(f"Stage 2 NEW TOKENS input_ids max value: {input_ids_stage2_new_tokens.max().item()}")

            # Inference 실행 시 '새로운 토큰'과 그에 맞는 attention_mask 전달
            analysis_text, attn_stage2, _, past_key_values = _generate_with_attention(
                model, tokenizer, 
                input_ids=input_ids_stage2_new_tokens, 
                # attention_mask=attention_mask_new_tokens, # <--- attention_mask도 함께 전달 시도
                image_tensor=None,
                image_sizes=None,
                vis_token_pos=vis_token_pos,
                past_key_values=past_key_values, 
                **PromptConfig.SCENE_ANALYSIS
            )
            
            # [수정] conv 객체와 messages 로그 모두 업데이트
            conv.messages[-1][1] = analysis_text # conv의 'None'을 실제 응답으로 교체
            messages.append({"role": "user", "content": query_stage2})
            messages.append({"role": "assistant", "content": analysis_text})
            
            print(f"[Multi-turn] Stage 2 output: {analysis_text[:100]}...")
            
            # 메타데이터 구성
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
                "conversation_history": messages.copy(),  # 대화 이력 포함
            }
            
            # 결과 저장
            stage2_path = _save_stage_result(
                save_dir, model_dir, save_id, "stage2_analysis",
                attn_stage2, meta_stage2
            )
            
            results["stage2"] = {
                "text": analysis_text,
                "attn_shape": list(attn_stage2.shape) if attn_stage2 is not None else None,
                "save_path": stage2_path,
                "meta": meta_stage2
            }
            
        except Exception as e:
            print(f"[Multi-turn] Error in Stage 2: {e}")
            results["stage2"] = {"error": str(e)}
            # Stage 2 실패해도 Stage 1 결과는 있으므로 부분 저장
            
        try:
            # ═══════════════════════════════════════════════════════
            # 4️⃣ Stage 3: Planning
            # ═══════════════════════════════════════════════════════
            print(f"[Multi-turn] Stage 3: Planning")
            
            # [수정] conv 객체에 새 턴 추가
            # conv.append_message(conv.roles[0], query_stage3)
            # conv.append_message(conv.roles[1], None)

            # [★수정★] 전체 대화 프롬프트를 가져와 토크나이징
            full_prompt_stage3 = conv.get_prompt()

            inputs_stage3 = tokenizer(full_prompt_stage3, return_tensors="pt")
            full_input_ids_stage3 = inputs_stage3.input_ids.to(model.device)

            # K-V 캐시에 저장된 시퀀스 길이 가져오기
            if past_key_values is not None:
                cached_sequence_length = past_key_values[0][0].shape[2] # 인덱스 [2] 확인 필요
            else: # Stage 1에서는 캐시 없음 (이 코드는 Stage 2 이후 실행됨)
                cached_sequence_length = 0

            input_ids_stage3_new_tokens = full_input_ids_stage3[:, cached_sequence_length:]
            
            print(f"Stage 3 FULL input_ids shape: {full_input_ids_stage3.shape}")
            print(f"Stage 3 NEW TOKENS input_ids shape: {input_ids_stage3_new_tokens.shape}") # 이 길이가 이전의 input_ids_stage2 길이와 같은지 확인
            print(f"Stage 3 NEW TOKENS input_ids max value: {input_ids_stage3_new_tokens.max().item()}")


            # [수정] _generate_with_attention 호출 (K-V 캐시 전달)
            planning_text, attn_stage3, _, past_key_values = _generate_with_attention(
                model, tokenizer, 
                input_ids=input_ids_stage3_new_tokens, 
                image_tensor=None,  # K-V cache 재사용
                image_sizes=None,
                vis_token_pos=vis_token_pos,
                past_key_values=past_key_values, # [수정] Stage 2의 캐시 전달
                **PromptConfig.PLANNING
            )
            
            # [수정] conv 객체와 messages 로그 모두 업데이트
            conv.messages[-1][1] = planning_text
            messages.append({"role": "user", "content": query_stage3})
            messages.append({"role": "assistant", "content": planning_text})
            
            print(f"[Multi-turn] Stage 3 output: {planning_text[:100]}...")
            
            # 메타데이터 구성
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
                "conversation_history": messages.copy(),  # 대화 이력 포함
            }
            
            # 결과 저장
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
            
        except Exception as e:
            print(f"[Multi-turn] Error in Stage 3: {e}")
            results["stage3"] = {"error": str(e)}
    
    # ═══════════════════════════════════════════════════════
    # 5️⃣ 통합 결과 저장
    # ═══════════════════════════════════════════════════════
    
    # 전체 결과를 하나의 파일로 저장
    multi_turn_path = os.path.join(out_dir, f"{save_id}_multi_turn.pkl")
    with open(multi_turn_path, "wb") as f:
        pickle.dump(results, f)
    
    # 사람이 읽기 쉬운 텍스트 요약 JSON 저장
    summary = {
        "model": model_name_str,
        "image": image_file,
        "conversation_history": messages,  # 전체 대화 이력
        "scene_description": results.get("stage1", {}).get("text", "N/A"),
        "scene_analysis": results.get("stage2", {}).get("text", "N/A"),
        "planning": results.get("stage3", {}).get("text", "N/A"),
    }
    
    summary_path = os.path.join(out_dir, f"{save_id}_multi_turn_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    results["multi_turn_file"] = multi_turn_path
    results["summary_file"] = summary_path
    
    print(f"[Multi-turn] Complete! Saved to:")
    print(f"  - Multi-turn: {multi_turn_path}")
    print(f"  - Summary: {summary_path}")
    
    return results