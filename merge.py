"""
Merge LoRA adapter with base model (FULL PRECISION).
"""

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from huggingface_hub import login
import gc
import os


def get_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--adapter_repo",
        type=str,
        required=True,
        help="HuggingFace repo containing the LoRA adapter"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="bigcode/starcoder2-3b",
        help="Base model to merge with (full precision)"
    )
    parser.add_argument(
        "--output_repo",
        type=str,
        default=None,
        help="HuggingFace repo to upload merged model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./merged_model",
        help="Local directory to save merged model"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the uploaded model private"
    )
    
    return parser.parse_args()


def merge_and_save(args):
    """Merge LoRA adapter with base model in FULL PRECISION"""
    
    if args.output_repo:
        print("Logging in to HuggingFace...")
        login()
    
    print(f"Loading BASE MODEL in FULL PRECISION from {args.base_model}...")
    # Load base model in FULL PRECISION (bf16, not quantized)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,  # Full precision bf16
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    print(f"Loading LoRA adapter from {args.adapter_repo}...")
    # Load adapter on top of full precision model
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_repo,
        torch_dtype=torch.bfloat16,
    )
    
    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    
    # Free memory
    del model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_repo)
    
    # Save locally
    print(f"Saving merged FULL PRECISION model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Upload if requested
    if args.output_repo:
        print(f"Uploading merged model to {args.output_repo}...")
        
        # Move to CPU if on GPU
        if next(merged_model.parameters()).is_cuda:
            print("Moving model to CPU for upload...")
            merged_model = merged_model.cpu()
            torch.cuda.empty_cache()
            gc.collect()
        
        merged_model.push_to_hub(
            args.output_repo,
            commit_message="Upload FULL PRECISION merged model",
            private=args.private,
        )
        
        tokenizer.push_to_hub(args.output_repo, private=args.private)
        
        print(f"✓ Uploaded to: https://huggingface.co/{args.output_repo}")
    
    print("\nMerge completed successfully!")


def main():
    args = get_args()
    merge_and_save(args)


if __name__ == "__main__":
    main()