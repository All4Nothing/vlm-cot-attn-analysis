import os
import sys
import re
import pickle
from typing import Dict, Tuple

import torch
import numpy as np
from PIL import Image

# Add models/llava to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from llava import LLaVAInferenceEngine, Config
from lab.stations import MetadataStation

# Constants for LLaVA 1.6
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


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
    """Build model/tokenizer/image_processor using LLaVA 1.6 Vicuna.

    Returns: tokenizer, model, processor, context_len, model_name_str
    """
    # Create custom config from hydra cfg
    class CustomConfig(Config):
        pass
    
    # Set model configuration
    CustomConfig.MODEL_NAME = cfg.model.name  # e.g., "llava-hf/llava-v1.6-vicuna-7b-hf"
    CustomConfig.MODEL_PATH = getattr(cfg.model, "base", None) or "./models/llava/model_weights/"
    
    # Optional cache control
    if getattr(cfg.model, "cache_dir", None):
        cache_dir = str(cfg.model.cache_dir)
        CustomConfig.CACHE_DIR = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_HUB_CACHE"] = cache_dir
    
    # Choose device
    device = cfg.device
    if device == "auto":
        device = f"cuda:{cfg.device_id}" if cfg.device_id >= 0 and torch.cuda.is_available() else "cpu"
    
    CustomConfig.DEVICE = device
    if "cuda" in device and cfg.device_id >= 0:
        CustomConfig.set_cuda_device(str(cfg.device_id))
    
    # Load using LLaVAInferenceEngine
    engine = LLaVAInferenceEngine(CustomConfig)
    engine.load_model()
    
    tokenizer = engine.processor.tokenizer
    model = engine.model
    processor = engine.processor  # LlavaNextProcessor
    
    context_len = getattr(model.config, 'max_position_embeddings', 2048)
    model_name_str = cfg.model.name.split('/')[-1]
    
    return tokenizer, model, processor, context_len, model_name_str


def _forward_collect(model, processor, input_ids, pixel_values):
    """Collect attentions via a single forward pass.

    Returns attention focused on image tokens with shape [L, H, 1, V].
    """
    # LLaVA 1.6 forward
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        output_attentions=True,
        return_dict=True,
    )
    
    print(f"outputs keys: {outputs.keys()}")
    print(f"outputs.logits: {outputs.logits.shape}")
    print(f"outputs.attentions: {len(outputs.attentions)}")
    
    attn_layers = outputs.attentions  # tuple of [B, H, Tq, Tk]
    if not attn_layers:
        raise RuntimeError("No attentions returned from forward()")

    # Stack layers: [L, B, H, Tq, Tk]
    layers = []
    for t in attn_layers:
        layers.append(t[0])  # [H, Tq, Tk] for batch=1
    attn = torch.stack(layers, dim=0)  # [L, H, Tq, Tk]
    
    print(f"shape of attn: {attn.shape}")
    
    begin_pos_vis = MetadataStation.get_begin_pos('vis')
    vis_len = MetadataStation.get_vis_len()
    if begin_pos_vis is None or vis_len is None:
        raise RuntimeError("Missing visual token segmentation info.")
    
    print(f"begin_pos_vis: {begin_pos_vis}, vis_len: {vis_len}")
    attn_last_to_vis = attn[:, :, -1:, begin_pos_vis:begin_pos_vis + vis_len]
    print(f"shape of attn_last_to_vis: {attn_last_to_vis.shape}")
    
    return attn_last_to_vis


def _generate_collect(model, processor, tokenizer, input_ids, pixel_values, max_new_tokens=10):
    """Run generate to obtain output tokens and collect attentions.

    Returns: (attn [L,H,1,V] or None, generated_text str)
    """
    gen = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_attentions=True,
    )
    
    sequences = gen.sequences
    generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
    
    print(f"Generated text: {generated_text}")
    print(f"gen.keys(): {gen.keys()}")
    print(f"len gen.attentions: {len(gen.attentions)}")
    
    if hasattr(gen, 'attentions') and gen.attentions:
        # Get attention from last generation step
        last_step_attn = gen.attentions[-1]  # tuple of layers
        print(f"len of last_step_attn: {len(last_step_attn)}")
        print(f"shape of last_step_attn[0]: {last_step_attn[0].shape}")
        
        layers = [t[0] for t in last_step_attn]  # [H, 1, src] for each layer
        attn = torch.stack(layers, dim=0)  # [L, H, 1, src]
        
        begin_pos_vis = MetadataStation.get_begin_pos('vis')
        vis_len = MetadataStation.get_vis_len()
        if begin_pos_vis is None or vis_len is None:
            raise RuntimeError("Missing visual token segmentation info.")
        
        attn_last_to_vis = attn[:, :, -1:, begin_pos_vis:begin_pos_vis + vis_len]
        print(f"shape of attn_last_to_vis: {attn_last_to_vis.shape}")
        
        return attn_last_to_vis, generated_text
    
    return None, generated_text


def collect_attention(cfg, image_file: str, query: str, save_dir: str, save_id: str) -> str:
    """Run forward/generate and save attention focused on image tokens.

    Saves a pickle with dict: {
      'attn': Tensor[L, H, 1, V],
      'meta': {image_file, query, image_size, model_name, vis_len, patch_size, num_layers, num_heads}
    }
    Returns the saved file path.
    """
    tokenizer, model, processor, _, model_name_str = load_model_from_cfg(cfg)

    # Prepare image and prompt
    image = load_image(image_file)
    image_size = image.size  # (W, H)
    
    # LLaVA 1.6 Vicuna prompt format
    prompt = f"<|im_start|>user\n<image>\n{query}<|im_end|>\n<|im_start|>assistant\n"
    
    # Process inputs using LlavaNextProcessor
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )
    
    # Move to device
    device = model.device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    
    # Collect attentions
    with torch.inference_mode():
        if getattr(cfg.model, 'use_generate', False):
            attn_last_to_vis, gen_text = _generate_collect(
                model, processor, tokenizer, input_ids, pixel_values,
                max_new_tokens=getattr(cfg.model, 'max_new_tokens', 10)
            )
            if attn_last_to_vis is None:
                attn_last_to_vis = _forward_collect(model, processor, input_ids, pixel_values)
        else:
            gen_text = None
            attn_last_to_vis = _forward_collect(model, processor, input_ids, pixel_values)

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
    if gen_text is not None:
        meta["generated_text"] = gen_text

    model_dir = _sanitize_name(cfg.model.name)
    out_dir = os.path.join(save_dir, model_dir)
    _ensure_dir(out_dir)

    save_path = os.path.join(out_dir, f"{save_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"attn": attn_last_to_vis.detach().cpu(), "meta": meta}, f)

    with open(os.path.join(out_dir, f"{save_id}_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    return save_path


"""prompt
# 1단계: 직접 프롬프트 작성 (간단!)
prompt = f"<|im_start|>user\n<image>\n{query}<|im_end|>\n<|im_start|>assistant\n"
# "<|im_start|>user\n<image>\ncar<|im_end|>\n<|im_start|>assistant\n"

# 2단계: Processor가 자동 처리
inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
)
# processor가 자동으로:
# 1. 텍스트 토큰화
# 2. 이미지 전처리
# 3. 이미지 토큰 위치 자동 계산
# 4. 모든 것을 올바르게 정렬

input_ids = inputs["input_ids"]  # 바로 사용 가능!
"""

"""이미지 처리
from llava.mm_utils import process_images

# 1단계: PIL 이미지 로드
image = load_image(image_file)  # PIL.Image
image_size = image.size  # (W, H)

# 2단계: 수동 전처리
image_tensor = process_images(
    [image],              # 리스트로 감싸기
    image_processor,      # 별도의 image_processor
    model.config          # 모델 설정 필요
)
# shape: [1, 3, 336, 336] (LLaVA 1.5)

# 3단계: 디바이스 이동 및 타입 변환
image_tensor = image_tensor.to(model.device, dtype=torch.float16)
image_sizes = [image.size]  # 원본 크기 별도 저장

# 4단계: forward/generate 시 전달
outputs = model(
    input_ids=input_ids,
    images=image_tensor.unsqueeze(0),  # [1, 1, 3, 336, 336]
    image_sizes=image_sizes,           # [(W, H)]
    ...
)

"""

"""LlavaNextProcessor 내부
# process_images 내부
def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, ...)  # 정사각형으로 패딩
            image = image_processor.preprocess(image, ...)
            new_images.append(image)
    else:
        # 직접 resize
        return image_processor(images, ...)
    
    return torch.stack(new_images, dim=0)
"""

"""inference
# Forward
outputs = model(
    input_ids=input_ids,              # [1, seq_len]
    images=image_tensor.unsqueeze(0), # [1, 1, 3, 336, 336] ← unsqueeze 필요!
    image_sizes=image_sizes,          # [(W, H)] ← 원본 크기
    output_attentions=True,
    return_dict=True
)

# Generate
gen = model.generate(
    inputs=input_ids,                 # ← "inputs" 이름 주의!
    images=image_tensor.unsqueeze(0),
    image_sizes=image_sizes,
    max_new_tokens=100,
    output_attentions=True,
    return_dict_in_generate=True
)
매개변수:
pixel_values: [batch, C, H, W] - 4D 텐서 (더 직관적)
image_sizes 불필요: processor가 자동 처리
input_ids: forward와 generate에서 동일한 이름

# 모델 내부에서
def forward(self, input_ids, images, image_sizes, ...):
    # 1. 이미지를 vision encoder에 통과
    image_features = self.vision_tower(images)  # [1, 1, 576, 1024]
    
    # 2. image_sizes 사용하여 패치 재구성
    image_features = self.resampler(
        image_features, 
        image_sizes=image_sizes
    )
    
    # 3. input_ids에서 IMAGE_TOKEN_INDEX (-200) 찾기
    image_positions = (input_ids == IMAGE_TOKEN_INDEX)
    
    # 4. 텍스트 임베딩과 이미지 임베딩 병합
    inputs_embeds = self.merge_embeddings(
        input_ids, 
        image_features, 
        image_positions
    )
    
    return super().forward(inputs_embeds=inputs_embeds, ...)
"""