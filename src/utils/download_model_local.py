#!/usr/bin/env python3
"""
Download the LLaVA model to the local folder
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_llava_model():
    """Download the LLaVA model to the local folder"""
    
    # Model information
    model_name = "llava-hf/llava-v1.6-vicuna-7b-hf"
    local_dir = "./models/llava/model_weights/"
    
    # Create the local directory
    os.makedirs(local_dir, exist_ok=True)
    logger.info(f"Created directory: {os.path.abspath(local_dir)}")
    
    try:
        logger.info(f"Starting download of {model_name}...")
        logger.info(f"Download location: {os.path.abspath(local_dir)}")
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Use actual files instead of symbolic links
            resume_download=True,  # Resume the download if it is interrupted
            ignore_patterns=["*.msgpack", "*.h5"]  # 불필요한 파일 제외
        )
        
        logger.info("✅ Model downloaded successfully!")
        
        # Check the downloaded files
        logger.info("\n📁 Downloaded files:")
        for root, dirs, files in os.walk(local_dir):
            level = root.replace(local_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024**3)  # GB
                logger.info(f"{subindent}{file} ({file_size:.2f} GB)")
        
        # config.py 업데이트 안내
        logger.info("\n🔧 Next steps:")
        logger.info("1. Update config.py:")
        logger.info('   MODEL_PATH = "./models/llava/model_weights/"')
        logger.info("2. Run your inference code!")
        
    except Exception as e:
        logger.error(f"❌ Error downloading model: {str(e)}")
        return False
    
    return True

def check_disk_space():
    """Check the disk space"""
    import shutil
    
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    
    logger.info(f"💾 Available disk space: {free_gb:.2f} GB")
    
    if free_gb < 15:  # LLaVA-7B는 약 13GB
        logger.warning("⚠️  Warning: Low disk space! LLaVA-7B requires ~13GB")
        return False
    return True

def main():
    logger.info("🚀 LLaVA Model Local Download Script")
    logger.info("=" * 50)
    
    # Check the disk space
    if not check_disk_space():
        logger.error("❌ Insufficient disk space!")
        return
    
    # Download the model
    if download_llava_model():
        logger.info("\n🎉 Download completed successfully!")
        logger.info("You can now set MODEL_PATH in config.py to use the local model.")
    else:
        logger.error("\nDownload failed!")

if __name__ == "__main__":
    main()
