"""
Usage examples for VLM attention analysis

This module contains various examples demonstrating how to use
the VLM attention analysis tools for research and experimentation.
"""

# Examples can be imported and run programmatically
from . import basic_usage
from . import attention_analysis

def run_basic_example():
    """Run basic inference example"""
    basic_usage.example_single_inference()

def run_attention_example():
    """Run comprehensive attention analysis example"""
    attention_analysis.example_comprehensive_analysis()

def run_all_examples():
    """Run all available examples"""
    print("Running Basic Usage Examples...")
    basic_usage.main()
    
    print("\n" + "="*80)
    print("Running Attention Analysis Examples...")
    attention_analysis.main()

__all__ = [
    'basic_usage',
    'attention_analysis',
    'run_basic_example',
    'run_attention_example',
    'run_all_examples'
]
