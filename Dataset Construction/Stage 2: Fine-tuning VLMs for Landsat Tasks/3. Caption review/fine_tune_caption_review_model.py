#!/usr/bin/env python3
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import requests
import torch
import wandb
from PIL import Image
from tqdm import tqdm
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    TrainingArguments,
    Trainer,
)


from qwen_vl_utils import process_vision_info

def caption_keep_or_delete_prompt(given_sentence):
    """
    Generates prompts to verify if a caption accurately reflects an image.

    Args:
        given_caption (str): The caption to be evaluated.

    Returns:
        tuple: A tuple containing the system prompt and the user prompt.
    """
    system_prompt = """Check the given image and its corresponding caption (a single sentence) to determine if the caption accurately reflects the content of the image. Respond with either "delete" if the caption does not match the image content or "keep" if it does.

# Output Format

- Respond with a single word: "delete" or "keep".

# Steps

1. Analyze the content of the provided image to understand its main elements and context.
2. Read the given caption and evaluate its accuracy and relevance to the image content.
3. Decide whether the caption accurately represents the image.
4. Respond accordingly with "delete" or "keep".

# Notes

- The caption should be a clear and direct reflection of the image's primary content.
- Consider the main focus of the image, including any prominent objects, actions, or emotions.
- "Keep" the caption if it correctly and completely represents the image without ambiguity. Otherwise, choose "delete"."""

    user_prompt = f"The given caption: {given_sentence}\n"

    return system_prompt, user_prompt


# --- Configuration ---

class Config:
    """A single class to manage all hyperparameters and paths."""
    # Model and Tokenizer
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Dataset Paths
    TRAIN_DATA_PATH = Path("caption_review_gt/caption_ft_train.csv")

    # Training Hyperparameters
    PER_DEVICE_BATCH_SIZE = 3
    ACCUM_STEPS = 1
    MAX_EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_NEW_TOKENS = 256  # Max tokens for generation during evaluation

    # Output and Logging
    MODEL_NAME = MODEL_ID.split("/")[-1]
    RESULT_PATH = Path(f"{MODEL_NAME}-caption-review-ft-model")
    LOGGING_DIR = RESULT_PATH / "logs"

    BASE_URL = (
    "PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE"
    )


# --- Data Handling ---

def convert_to_qwen_format(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts a sample from the dataset into the Qwen-VL chat template format
    and downloads the associated image.
    """
    try:

        given_sentence = row["given_sentence"]

        image_path = row["image_path"]
        image_url = Config.BASE_URL + image_path

        decision = row["decision"]

        response = requests.get(image_url, timeout=20)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")

        system_prompt, user_prompt = caption_keep_or_delete_prompt(given_sentence)

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": decision}]},
        ]
    except (requests.RequestException, IOError, KeyError, IndexError) as e:
        print(f"Skipping sample due to error: {e}")
        return None

def load_and_preprocess_data(file_path: Path) -> List[Dict]:
    """Reads a JSONL file and preprocesses its content."""
    df = pd.read_csv(file_path)
    processed_data = [
        convert_to_qwen_format(row)
        for _, row in tqdm(df.iterrows(), desc=f"Processing {file_path.name}")
    ]
    return [item for item in processed_data if item is not None]

# --- Model & Tokenizer ---

def initialize_model_and_processor(model_id: str):
    """Loads the model and processor from Hugging Face."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(model_id)
    return model, processor


def create_collate_fn(processor):
    """Creates a data collator for batching and tokenization."""
    pad_token_id = processor.tokenizer.pad_token_id
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)

    def collate_fn(examples: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]
        image_inputs = [process_vision_info(ex)[0] for ex in examples]

        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )

        labels = batch["input_ids"].clone()
        labels[labels == pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    return collate_fn

# --- Main Execution ---

def main():
    """Main function to run the entire training and evaluation pipeline."""
    cfg = Config()
    cfg.RESULT_PATH.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {cfg.RESULT_PATH}")

    # 1. Initialize Model and Processor
    model, processor = initialize_model_and_processor(cfg.MODEL_ID)

    # 2. Load and Prepare Datasets
    train_dataset = load_and_preprocess_data(cfg.TRAIN_DATA_PATH)

    training_args = TrainingArguments(
        output_dir=str(cfg.RESULT_PATH),
        num_train_epochs=cfg.MAX_EPOCHS,
        per_device_train_batch_size=cfg.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=cfg.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=cfg.ACCUM_STEPS,
        learning_rate=cfg.LEARNING_RATE,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        logging_dir=str(cfg.LOGGING_DIR),
        logging_steps=5,
        bf16=True,
        gradient_checkpointing=True,
        save_total_limit=3,
        dataloader_num_workers=min(os.cpu_count(), 8),
        dataloader_pin_memory=True,
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=create_collate_fn(processor),
    )

    # 5. Train and Save Model
    print("Starting training...")
    trainer.train()
    trainer.save_model(str(cfg.RESULT_PATH))
    print(f"Model saved to {cfg.RESULT_PATH}")

    wandb.finish()
    print("✅ Training and evaluation complete.")

if __name__ == "__main__":
    main()