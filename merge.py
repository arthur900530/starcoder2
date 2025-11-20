"""
Merge LoRA adapter with base model and optionally upload to HuggingFace Hub.

Usage:
    python merge.py --adapter_repo your-username/model-adapter --output_repo your-username/model-merged
    python merge.py --adapter_repo your-username/model-adapter --output_dir ./local_merged_model
"""

import argparse
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import login
import gc
import os


def get_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--adapter_repo",
        type=str,
        required=True,
        help="HuggingFace repo containing the LoRA adapter (e.g., 'username/model-adapter')"
    )
    parser.add_argument(
        "--output_repo",
        type=str,
        default=None,
        help="HuggingFace repo to upload merged model (e.g., 'username/model-merged'). If not provided, only saves locally."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./merged_model",
        help="Local directory to save merged model (default: './merged_model')"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit to save memory (useful for large models)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the uploaded model private on HuggingFace Hub"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, will prompt if not provided and uploading)"
    )
    
    return parser.parse_args()


def merge_and_save(args):
    """Merge LoRA adapter with base model and save/upload"""
    
    # Login to HuggingFace if uploading
    if args.output_repo:
        if args.token:
            login(token=args.token)
        else:
            print("Please login to HuggingFace:")
            login()
    
    print(f"Loading adapter from {args.adapter_repo}...")
    
    # Load model with adapter
    load_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    
    if args.load_in_8bit:
        print("Loading in 8-bit mode to save memory...")
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_repo,
        **load_kwargs
    )
    
    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    
    # Free up memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_repo)
    
    # Save locally
    print(f"Saving merged model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✓ Saved locally to {args.output_dir}")
    
    # Upload to HuggingFace Hub if requested
    if args.output_repo:
        print(f"Uploading merged model to {args.output_repo}...")
        
        # Move to CPU if on GPU to avoid memory issues during upload
        if next(merged_model.parameters()).is_cuda:
            print("Moving model to CPU for upload...")
            merged_model = merged_model.cpu()
            torch.cuda.empty_cache()
            gc.collect()
        
        merged_model.push_to_hub(
            args.output_repo,
            commit_message="Upload merged model",
            private=args.private,
        )
        
        tokenizer.push_to_hub(
            args.output_repo,
            commit_message="Upload tokenizer",
            private=args.private,
        )
        
        print(f"✓ Uploaded to: https://huggingface.co/{args.output_repo}")
    
    print("\nMerge completed successfully!")


def main():
    args = get_args()
    
    # Validate arguments
    if not args.output_repo and not args.output_dir:
        raise ValueError("Must provide either --output_repo or --output_dir")
    
    merge_and_save(args)


if __name__ == "__main__":
    main()