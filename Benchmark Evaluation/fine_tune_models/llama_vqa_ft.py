#!/usr/bin/env python3
"""
Llama‑3.2‑11B‑Vision ❱❱ LoRA / QLoRA VQA fine‑tune
• 4‑bit base model (NF4) + bfloat16 LoRA adapters
• PEFT LoRAConfig targets attention + MLP projections
• Single‑GPU friendly
"""

import requests, torch
import wandb
from pathlib import Path
from io import BytesIO
from PIL import Image
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from transformers import (
    MllamaForConditionalGeneration,  # ← use the vision‑aware class
    MllamaProcessor,  # ← **fix 1**
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ────────────────────────────────────────────────────────────────────────────
# 1. Paths & hyper‑params
# ────────────────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
TRAIN_JSONL = (
    "vqa_benchmark_dataset_val_ft.jsonl"
)

RESULT_DIR = Path(
    f"vqa_benchmark_ft/{Path(MODEL_ID).name.replace('meta-llama/', '')}-llama-benchmark-LORA-vqa"
)
RESULT_DIR.mkdir(exist_ok=True)

# LoRA / QLoRA specifics
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
EPOCHS = 1
BATCH = 8
GRAD_ACC = 1
MAX_NEW_TOKENS = 48

# ────────────────────────────────────────────────────────────────────────────
# 2. Initialise **processor** first so we can hand the image token to workers
# ────────────────────────────────────────────────────────────────────────────
processor = MllamaProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")  # fix 1
IMAGE_TOKEN = processor.image_token  # "<|image|>"


# ────────────────────────────────────────────────────────────────────────────
# 3. Dataset helper – receives the image_token as an argument
# ────────────────────────────────────────────────────────────────────────────
def convert_data_to_llama_template(sample, image_token: str = IMAGE_TOKEN):
    """Download image & wrap record in Llama‑3.2 chat template."""
    try:
        sys_text = sample[0]["content"]
        prompt = sample[1]["content"]
        img_url = sample[2]["content"][0]["image_url"]["url"]
        ref = sample[3]["content"]

        img = Image.open(BytesIO(requests.get(img_url, timeout=10).content)).convert(
            "RGB"
        )
        return {
            "images": [img],  # list → future‑proof for multi‑image
            "messages": [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": f"{image_token}\n{prompt}"},
                {"role": "assistant", "content": ref},
            ],
        }
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# 4. Build the training list (still using multiprocessing, but only a string
#    travels to each worker, never the heavy processor object)
# ────────────────────────────────────────────────────────────────────────────
raw_df = pd.read_json(TRAIN_JSONL, lines=True)

messages_to_process = raw_df["messages"].tolist()

with Pool() as pool:
    fn = partial(convert_data_to_llama_template, image_token=IMAGE_TOKEN)
    train_records = [
        r
        for r in tqdm(
            pool.imap(fn, messages_to_process),
            total=len(messages_to_process),
            desc="Downloading & templating",
        )
        if r is not None
    ]

print(f"Successfully processed {len(train_records)} records.")

# ────────────────────────────────────────────────────────────────────────────
# 5. Quantised base model  ➜  add LoRA adapters
# ────────────────────────────────────────────────────────────────────────────
bnb_cfg = BitsAndBytesConfig(  
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


base_model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
)

base_model = prepare_model_for_kbit_training(  #
    base_model, use_gradient_checkpointing=True
)

target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",  # attention
    "gate_proj",
    "up_proj",
    "down_proj",  
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
model.print_trainable_parameters()


# ────────────────────────────────────────────────────────────────────────────
# 6. Collate‑function factory – avoids “unexpected keyword 'processor'”
# ────────────────────────────────────────────────────────────────────────────
def make_collate_fn(proc: MllamaProcessor):
    image_token_id = proc.image_token_id
    pad_id = proc.tokenizer.pad_token_id

    def collate_fn(features):
        texts = [
            proc.apply_chat_template(x["messages"], tokenize=False) for x in features
        ]
        images = [x["images"] for x in features]

        batch = proc(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == pad_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


data_collator = make_collate_fn(processor)

# ────────────────────────────────────────────────────────────────────────────
# 7. TrainingArguments + Trainer
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
    run_name=str(RESULT_DIR.name),
    dataloader_num_workers=0, 
    remove_unused_columns=False,  
    save_total_limit=1,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_records,  
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)

trainer.train()
# model.save_pretrained(RESULT_DIR / "adapter")
# processor.save_pretrained(RESULT_DIR / "adapter")

adapter_dir = RESULT_DIR / "adapter"
trainer.save_model(adapter_dir)          # writes adapter + config.json
processor.save_pretrained(adapter_dir)   # tokenizer & image processor
print("✓ adapter saved to", adapter_dir)

