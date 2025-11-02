#!/usr/bin/env python3

import sys
from models.llava.model.builder import load_pretrained_model
from models.llava.mm_utils import get_model_name_from_path
from models.llava.utils import disable_torch_init

def load_tokenizer(model_name="liuhaotian/llava-v1.5-7b"):
    disable_torch_init()
    model_name_str = get_model_name_from_path(model_name)
    tok, _, _, _ = load_pretrained_model(
        model_path=model_name,
        cache_dir=None,
        model_base=None,
        model_name=model_name_str,
        device="cpu",
        load_8bit=False,
        load_4bit=False,
    )
    return tok

def main():
    tokenizer = load_tokenizer()
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print()
    
    example_texts = [
            "multiple cars and trucks",
            "traffic signs and traffic lights",
            "green light",
            "pedestrians and cyclists"
    ]
    
    for example in example_texts:
        print(f"Text: '{example}'")
        
        # add_special_tokens=True (기본값, BOS 토큰 포함)
        token_ids = tokenizer.encode(example, add_special_tokens=True)
        print(f"  With special tokens ({len(token_ids)} tokens):")
        for i, token_id in enumerate(token_ids):
            decoded = tokenizer.decode(token_id)
            print(f"    [{i}] Token ID: {token_id} | Decoded: '{decoded}'")
        
        # add_special_tokens=False (순수 텍스트만)
        token_ids_no_special = tokenizer.encode(example, add_special_tokens=False)
        print(f"  Without special tokens ({len(token_ids_no_special)} tokens):")
        for i, token_id in enumerate(token_ids_no_special):
            decoded = tokenizer.decode(token_id)
            print(f"    [{i}] Token ID: {token_id} | Decoded: '{decoded}'")
        print()

if __name__ == "__main__":
    main()
