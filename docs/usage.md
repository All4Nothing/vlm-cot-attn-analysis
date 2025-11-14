# multi-turn
CUDA_VISIBLE_DEVICES=0 python pipeline.py \
  stage=multi_turn \
  data.image_file={image_path} save_all_heads=true

# single
CUDA_VISIBLE_DEVICES=0 python pipeline.py \
  stage=pipeline \
  data.image_file={image_path} data.query={input_text} model.use_generate=true save_all_heads=true

# Debugging
For GPU debugging, use 'CUDA_LAUNCH_BLOCKING=1'. Note that this may significantly reduce performance.