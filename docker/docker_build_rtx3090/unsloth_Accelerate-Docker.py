#!/usr/bin/env python
import os
import sys
import gc

# --- Patch Huggingface's HfFileSystem.glob to handle local paths for docker ---
import glob
from huggingface_hub.hf_file_system import HfFileSystem

_orig_glob = HfFileSystem.glob
def _glob_override(self, pattern, *args, **kwargs):
    if pattern.startswith("file://"):
        return glob.glob(pattern[len("file://"):])
    if os.path.isabs(pattern) or pattern.startswith('./') or pattern.startswith('../') or '/' in pattern:
         if os.path.exists(pattern) or '*' in pattern or '?' in pattern or '[' in pattern:
             return glob.glob(pattern)
    return _orig_glob(self, pattern, *args, **kwargs)

HfFileSystem.glob = _glob_override
print("[PID {}] Patched HfFileSystem.glob for local paths.".format(os.getpid()), flush=True)

# --- Critical Environment Variables ---
os.environ["TORCH_DISTRIBUTED_USE_DTENSOR"] = "0"
os.environ["TORCH_DIST_DDP_SHARDING"] = "0"
os.environ["ACCELERATE_USE_TP"] = "false"
os.environ["PYTORCH_ENABLE_DISTRIBUTED"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# RTX 3090 specific optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

OUTPUT_ROOT = "/app/output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

print(f"[PID {os.getpid()}] Script start. Python version: {sys.version}", flush=True)
print(f"[PID {os.getpid()}] Current PWD: {os.getcwd()}", flush=True)

import torch
print(f"[PID {os.getpid()}] Imported torch. Version: {torch.__version__}. CUDA available: {torch.cuda.is_available()}", flush=True)

# DTensor patch
try:
    from torch.distributed.tensor import DTensor
    if hasattr(DTensor, "_op_dispatcher") and \
       hasattr(DTensor._op_dispatcher.sharding_propagator, "propagate"):
        def _no_op_propagate(self_sharding_prop, op_info, *args, **kwargs):
            return op_info.output_sharding
        DTensor._op_dispatcher.sharding_propagator.propagate = _no_op_propagate
        print(f"✅ Successfully patched DTensor.", flush=True)
except Exception as e:
    print(f"⚠️ Error during DTensor patching: {e}", flush=True)

from accelerate import Accelerator
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# Configuration - Adjusted for RTX 3090
MODEL_PATH = "/app/models/Hermes-3-Llama-3.1-8B"
MAX_SEQ_LENGTH = int(os.getenv("UNSLOTH_MAX_SEQ", 4096))  # Reduced from 8192
LORA_R = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LOCAL_JSONL = "/app/data/unsloth_datasets/qwen_combined_valid_prompts.jsonl"
TEST_SPLIT_RATIO = 0.1238
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2  # Increased to compensate for smaller batch size
MAX_STEPS = 30
LEARNING_RATE = 2e-4

def load_model(current_accelerator):
    rank_idx = current_accelerator.process_index
    pid = os.getpid()
    print(f"[PID {pid}, Rank {rank_idx}] Loading model...", flush=True)
    
    device_map_config = {"": current_accelerator.device}
    
    model_kwargs = {
        "model_name": MODEL_PATH,
        "max_seq_length": MAX_SEQ_LENGTH,
        "load_in_4bit": True,
        "attn_implementation": "flash_attention_2",
        "device_map": device_map_config,
        "dtype": torch.bfloat16,
        # RTX 3090 specific: Enable more aggressive memory saving
        "use_cache": False,
        "rope_scaling": None,  # Disable if causing issues
    }

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
        print(f"[PID {pid}, Rank {rank_idx}] Model loaded successfully.", flush=True)
        
        # Enable gradient checkpointing immediately to save memory
        model.gradient_checkpointing_enable()
        
        return model, tokenizer
    except Exception as e:
        print(f"🔥 [PID {pid}, Rank {rank_idx}] ERROR loading model: {e}", flush=True)
        raise

def apply_lora(base_model, current_accelerator):
    rank_idx = current_accelerator.process_index
    pid = os.getpid()
    print(f"[PID {pid}, Rank {rank_idx}] Applying LoRA...", flush=True)
    
    try:
        lora_model = FastLanguageModel.get_peft_model(
            base_model,
            r=LORA_R,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Critical for memory savings
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        print(f"[PID {pid}, Rank {rank_idx}] LoRA applied successfully.", flush=True)
        return lora_model
    except Exception as e:
        print(f"🔥 [PID {pid}, Rank {rank_idx}] ERROR applying LoRA: {e}", flush=True)
        raise

def load_and_split_dataset(current_accelerator):
    rank_idx = current_accelerator.process_index
    pid = os.getpid()
    print(f"[PID {pid}, Rank {rank_idx}] Loading dataset...", flush=True)
    
    try:
        ds = load_dataset("json", data_files={"train": f"file://{LOCAL_JSONL}"}, trust_remote_code=True)["train"]
        splits = ds.train_test_split(test_size=TEST_SPLIT_RATIO, seed=42)
        return splits["train"], splits["test"]
    except Exception as e:
        print(f"🔥 [PID {pid}, Rank {rank_idx}] ERROR loading dataset: {e}", flush=True)
        raise

def main():
    pid = os.getpid()
    print(f"[PID {pid}] Initializing Accelerator...", flush=True)
    
    # Initialize with mixed precision for memory efficiency
    accelerator = Accelerator(mixed_precision="bf16")
    rank_idx = accelerator.process_index
    
    print(f"[PID {pid}, Rank {rank_idx}] Accelerator initialized.", flush=True)

    # Clear cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    model, tokenizer = load_model(accelerator)
    model = apply_lora(model, accelerator)
    
    train_ds_raw, val_ds_raw = load_and_split_dataset(accelerator)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process datasets
    DATASET_MAP_NUM_PROC = 1
    train_ds = train_ds_raw.map(
        lambda ex: {"text": ex["text"]}, 
        num_proc=DATASET_MAP_NUM_PROC, 
        remove_columns=[col for col in train_ds_raw.features if col != 'text']
    )
    
    val_ds = val_ds_raw.map(
        lambda batch: {"text": batch["text"]}, 
        batched=True, 
        num_proc=DATASET_MAP_NUM_PROC, 
        remove_columns=[col for col in val_ds_raw.features if col != 'text']
    )

    # Training arguments optimized for RTX 3090
    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        optim="adamw_8bit",  # 8-bit optimizer to save memory
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=os.path.join(OUTPUT_ROOT, "training_outputs"),
        report_to="none",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        bf16=True,
        fp16=False,
        # RTX 3090 specific
        dataloader_pin_memory=False,  # Can help with memory fragmentation
        dataloader_num_workers=0,     # Reduce memory overhead
        remove_unused_columns=True,
        load_best_model_at_end=False,  # Save memory by not keeping best model
    )

    print(f"[PID {pid}, Rank {rank_idx}] Initializing SFTTrainer...", flush=True)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        packing=False,  # Packing can increase memory usage
        args=training_args,
    )

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(trainer.model)
        if hasattr(unwrapped_model, "config"):
            unwrapped_model.config.use_cache = False

    accelerator.wait_for_everyone()
    print(f"[PID {pid}, Rank {rank_idx}] Starting training...", flush=True)

    try:
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        metrics = trainer.train()
        print(f"[PID {pid}, Rank {rank_idx}] Training completed.", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f"🔥 [PID {pid}, Rank {rank_idx}] CUDA OOM Error: {e}", flush=True)
        print("Consider reducing batch size, sequence length, or using gradient accumulation.")
        raise
    except Exception as e:
        print(f"🔥 [PID {pid}, Rank {rank_idx}] Training error: {e}", flush=True)
        raise

    accelerator.wait_for_everyone()

    # Saving logic (only on main process)
    if accelerator.is_main_process:
        print(f"[PID {pid}, Rank {rank_idx}] Saving model...", flush=True)
        
        # Save paths
        lora_adapter_path = os.path.join(OUTPUT_ROOT, "lora_adapters_final")
        merged_model_16bit_path = os.path.join(OUTPUT_ROOT, "merged_model_16bit")
        
        # Save LoRA adapters
        try:
            trainer.model.save_pretrained(lora_adapter_path)
            tokenizer.save_pretrained(lora_adapter_path)
            print(f"[PID {pid}, Rank {rank_idx}] LoRA adapters saved.", flush=True)
        except Exception as e:
            print(f"⚠️ Error saving LoRA adapters: {e}", flush=True)

        # For full model save, be very careful with memory
        try:
            # Clear everything possible first
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            
            # Load model in a memory-efficient way for merging
            print(f"[PID {pid}, Rank {rank_idx}] Merging and saving full model...", flush=True)
            
            # Note: You might need to save only LoRA adapters and merge offline
            # if memory is too constrained
            
        except Exception as e:
            print(f"⚠️ Could not save full merged model due to memory constraints: {e}", flush=True)
            print("Consider merging the model offline on a system with more RAM.")

    print(f"[PID {pid}, Rank {rank_idx}] Process completed successfully.", flush=True)
    return metrics if 'metrics' in locals() else None

if __name__ == "__main__":
    try:
        results = main()
        print(f"[PID {os.getpid()}] Training complete.", flush=True)
    except Exception as e:
        print(f"🔥 FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
