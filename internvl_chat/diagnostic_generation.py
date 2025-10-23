#!/usr/bin/env python3
"""
Diagnostic script to test if the merged model generates properly
"""
import torch
from internvl.model.internvl_chat import InternVLChatModel
from transformers import AutoTokenizer
from mpo_intern_evaluate_parallel_prod_auto import load_image

import sys

def test_model_generation(model_path, test_image_path=None):
    print(f"Loading model from: {model_path}")
    model = InternVLChatModel.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).eval().cuda()
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"\nModel config:")
    print(f"  use_llm_lora: {model.config.use_llm_lora}")
    print(f"  use_backbone_lora: {model.config.use_backbone_lora}")
    
    # Test 1: Simple text generation (no image)
    print("\n" + "="*60)
    print("TEST 1: Simple text generation (no image)")
    print("="*60)
    
    query = "What is 2+2?"
    generation_config = dict(
        max_new_tokens=100,
        do_sample=False,
        num_beams=1,  # Start with greedy
        repetition_penalty=1.0,
    )
    
    try:
        response = model.chat(
            tokenizer,
            pixel_values=None,
            question=query,
            generation_config=generation_config,
        )
        print(f"Query: {query}")
        print(f"Response: {response}")
        print(f"Response length: {len(response)}")
        
        # Check if it's all repetition
        if response.count('(') > len(response) * 0.8:
            print("❌ ISSUE: Model is generating mostly '(' characters!")
        else:
            print("✓ Normal generation")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: With image (if provided)
    if test_image_path:
        print("\n" + "="*60)
        print("TEST 2: Generation with image")
        print("="*60)
        
        try:
            pixel_values = load_image(test_image_path, max_num=12).to(torch.bfloat16).cuda()
            
            query = "Describe this image in detail."
            response = model.chat(
                tokenizer,
                pixel_values=pixel_values,
                question=query,
                generation_config=generation_config,
            )
            print(f"Query: {query}")
            print(f"Response: {response}")
            print(f"Response length: {len(response)}")
            
            if response.count('(') > len(response) * 0.8:
                print("❌ ISSUE: Model is generating mostly '(' characters!")
            else:
                print("✓ Normal generation")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 3: Different generation configs
    print("\n" + "="*60)
    print("TEST 3: Different generation configs")
    print("="*60)
    
    configs_to_test = [
        ("Greedy (no beams)", dict(max_new_tokens=100, do_sample=False, num_beams=1, repetition_penalty=1.0)),
        ("With repetition penalty 1.2", dict(max_new_tokens=100, do_sample=False, num_beams=1, repetition_penalty=1.2)),
        ("Beam search (num_beams=2)", dict(max_new_tokens=100, do_sample=False, num_beams=2, repetition_penalty=1.0)),
        ("With sampling", dict(max_new_tokens=100, do_sample=True, num_beams=1, top_p=0.9, temperature=0.7, repetition_penalty=1.0)),
    ]
    
    query = "What is artificial intelligence?"
    
    for config_name, config in configs_to_test:
        try:
            response = model.chat(
                tokenizer,
                pixel_values=None,
                question=query,
                generation_config=config,
            )
            paren_ratio = response.count('(') / max(len(response), 1)
            status = "❌ BAD" if paren_ratio > 0.8 else "✓ GOOD"
            print(f"\n{config_name}: {status}")
            print(f"  Length: {len(response)}, '(' ratio: {paren_ratio:.2%}")
            print(f"  Preview: {response[:150]}...")
        except Exception as e:
            print(f"\n{config_name}: ❌ ERROR - {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnostic.py <merged_model_path> [test_image_path]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_model_generation(model_path, test_image_path)
