#!/usr/bin/env python3
"""
Qwen‑2.5‑VL‑7B  ❱❱  LoRA / QLoRA caption‑fine‑tune

• 4‑bit base model (NF4) + bfloat16 LoRA adapters
• PEFT LoRAConfig targets attention + MLP projections incl. vision projector
• Single‑GPU friendly (24 GB ≈ 11 GB peak)
"""

import os, json, requests, wandb, torch
from pathlib import Path
from io import BytesIO
from PIL import Image
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool  # <-- ADDED: For parallel processing
from functools import partial  # <-- ADD THIS IMPORT

from transformers import (
    Qwen2_5_VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from qwen_vl_utils import process_vision_info  # ← unchanged helper

# ────────────────────────────────────────────────────────────────────────────
# 1. Paths & hyper‑params
# ────────────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
TRAIN_JSONL = (
    "caption_benchmark_dataset_val_ft.jsonl"
)

RESULT_DIR = Path(f"caption_benchmark_ft/{Path(MODEL_ID).name}-benchmark-LORA-caption")
RESULT_DIR.mkdir(exist_ok=True)


# LoRA / QLoRA specifics
LORA_RANK = 64  # r
LORA_ALPHA = 128  # usually 2×r
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
EPOCHS = 1
BATCH = 4
GRAD_ACC = 1
MAX_NEW_TOKENS = 256


# ────────────────────────────────────────────────────────────────────────────
# 2. Load dataset and convert to chat‑template list‑of‑dict
# ────────────────────────────────────────────────────────────────────────────
def convert_data_to_qwen_template(sample):
    """Processes a single data sample, downloading the image and formatting it."""
    try:
        sys_text = sample[0]["content"]
        prompt = sample[1]["content"]
        img_url = sample[2]["content"][0]["image_url"]["url"]
        ref = sample[3]["content"]

        img = Image.open(BytesIO(requests.get(img_url, timeout=10).content)).convert(
            "RGB"
        )
        return [
            {"role": "system", "content": [{"type": "text", "text": sys_text}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": ref}]},
        ]
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# 4. Collate – identical to your original (unchanged)
# ────────────────────────────────────────────────────────────────────────────
def collate_fn(batch, processor):
    txts = [processor.apply_chat_template(x, tokenize=False) for x in batch]
    imgs = [process_vision_info(x)[0] for x in batch]

    model_inputs = processor(text=txts, images=imgs, return_tensors="pt", padding=True)
    labels = model_inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


# ============================================================================
# Main Execution Block
# ============================================================================
if __name__ == "__main__":
    # To prevent potential issues with fork in a multi-threaded environment
    torch.multiprocessing.set_start_method("spawn", force=True)

    print("Output dir:", RESULT_DIR.resolve())

    # --- MODIFIED: Parallel Data Processing ---
    print("Loading and processing dataset with multiprocessing...")
    raw_df = pd.read_json(TRAIN_JSONL, lines=True)

    messages_to_process = raw_df["messages"].tolist()

    # Use a multiprocessing Pool to parallelize the data conversion
    # This will use all available CPU cores by default
    with Pool() as pool:
        # Use imap to get a progress bar with tqdm
        results = list(
            tqdm(
                pool.imap(convert_data_to_qwen_template, messages_to_process),
                total=len(messages_to_process),
            )
        )
    # Filter out any samples that failed during processing
    train_records = [r for r in results if r is not None]
    print(f"Successfully processed {len(train_records)} records.")
    # --- END MODIFICATION ---

    # ────────────────────────────────────────────────────────────────────────────
    # 3. Model + processor (4‑bit)   →   add LoRA adapters
    # ────────────────────────────────────────────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
    )

    processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_ID)

    # Prepare for k‑bit training (enables gradient checkpointing & input cast)
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True
    )

    # Identify modules that receive LoRA weights
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",  # attention
        "gate_proj",
        "up_proj",
        "down_proj",  # MLP
        "vision_proj",  # cross‑modal projector
    ]

    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()  # sanity‑check

    # ────────────────────────────────────────────────────────────────────────────
    # 5. TrainingArguments
    # ────────────────────────────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=str(RESULT_DIR),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        weight_decay=1e-6,
        bf16=True,
        save_strategy="epoch",
        logging_steps=20,
        #report_to=["wandb"],
        run_name=str(RESULT_DIR.name),
        dataloader_num_workers=min(os.cpu_count(), 8),
        save_total_limit=1,
    )


    data_collator_with_processor = partial(collate_fn, processor=processor)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_records,
        data_collator=data_collator_with_processor,
        tokenizer=processor.tokenizer,
    )

    trainer.train()

    # ------------- 6. save adapter only (no merge here) -----------------
    adapter_dir = RESULT_DIR / "adapter"
    trainer.save_model(adapter_dir)          # writes adapter + config.json
    processor.save_pretrained(adapter_dir)   # tokenizer & image processor
    print("✓ adapter saved to", adapter_dir)
    # --------------------------------------------------------------------
