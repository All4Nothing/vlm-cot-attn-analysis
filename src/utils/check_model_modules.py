#!/usr/bin/env python3
"""
Check the module structure of the model
"""

# Import path setup is handled by __init__.py
from model_loader import LLaVAModelLoader
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_modules():
    """Check the module structure of the model"""
    
    config = Config()
    loader = LLaVAModelLoader(config)
    
    try:
        logger.info("Loading model to check module structure...")
        model, processor = loader.load_model()
        
        logger.info("\n" + "="*60)
        logger.info("MODEL MODULE STRUCTURE")
        logger.info("="*60)
        
        # Check all named modules
        logger.info("\nAll named modules:")
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if name == "":
                logger.warning(f"EMPTY NAME: '' -> {module_type}")
            elif not name.replace(".", "_").replace("-", "_").isidentifier():
                logger.warning(f"INVALID NAME: '{name}' -> {module_type}")
            else:
                logger.info(f"'{name}' -> {module_type}")
        
        logger.info("\n" + "="*60)
        logger.info("NAMED PARAMETERS")
        logger.info("="*60)
        
        # Check the parameter names
        param_count = 0
        invalid_names = []
        
        for name, param in model.named_parameters():
            param_count += 1
            if name == "":
                invalid_names.append(f"EMPTY: '' (shape: {param.shape})")
            elif not name.replace(".", "_").replace("-", "_").isidentifier():
                invalid_names.append(f"INVALID: '{name}' (shape: {param.shape})")
        
        logger.info(f"Total parameters: {param_count}")
        logger.info(f"Invalid parameter names: {len(invalid_names)}")
        
        if invalid_names:
            logger.warning("\nInvalid parameter names found:")
            for invalid in invalid_names[:10]:  # Show the first 10 invalid names
                logger.warning(f"  {invalid}")
            if len(invalid_names) > 10:
                logger.warning(f"  ... and {len(invalid_names) - 10} more")
        
        logger.info("\n" + "="*60)
        logger.info("MODEL STRUCTURE SUMMARY")
        logger.info("="*60)
        
        # The main components of the model
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Model config: {model.config}")
        
        # The main submodules
        main_modules = []
        for name, module in model.named_children():
            main_modules.append(f"{name}: {type(module).__name__}")
        
        logger.info("Main submodules:")
        for module_info in main_modules:
            logger.info(f"  {module_info}")
            
    except Exception as e:
        logger.error(f"Error checking model: {str(e)}")
    finally:
        if 'loader' in locals():
            loader.unload_model()

if __name__ == "__main__":
    check_model_modules()
