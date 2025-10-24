```python
CUDA_VISIBLE_DEVICES=0 python pipeline.py stage=pipeline data.image_file=data/nuscenes/n008.jpg data.query="Describe the image" model.use_generate=true save_all_heads=true
```