
## LLaVA-v1.5.-7B
- Vision Encoder: CLIP ViT-L/14 (336px)
    - image -> 336 x 336으로 변환
    - 14x14 patch로 이미지 나눔
    - 336/14 * 336/14 = 24 * 24 = 576: image tokens

- LLM
    - token lenght: 4096
    - 32 Layers 32 Heads = 1024 heads
    - vocab size: 32000
