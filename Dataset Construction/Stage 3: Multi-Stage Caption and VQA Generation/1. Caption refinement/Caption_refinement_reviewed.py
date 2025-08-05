#!/usr/bin/env python3
"""
qwen_batch_caption_review_v9_refactored.py
-----------------------------------------
A refactored, crash-resistant, and deduplicated caption-review pipeline for Qwen-VL.
This version enhances modularity and readability by breaking down the core logic.
"""
import argparse
import csv
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from contextlib import contextmanager

@contextmanager
def open_csv_writer(path: Path, fields):
    mode = "a" if path.exists() else "w"
    with open(path, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if mode == "w":
            w.writeheader()
        yield w, fh

def load_image(image_path, mode="PIL"):
    """
    Loads an image from a URL or local file.

    Parameters:
      - image_path (str): URL or local path to the image.
      - mode (str): 'PIL' returns a PIL.Image object; 'base64' returns a base64 encoded string.

    Returns:
      - A PIL.Image object or a base64-encoded string, depending on the mode.
    """

    import requests
    import base64
    from io import BytesIO
    from PIL import Image

    if image_path.startswith("http"):
        response = requests.get(image_path)
        response.raise_for_status()
        image_bytes = response.content
    else:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

    if mode == "base64":
        return base64.b64encode(image_bytes).decode("utf-8")
    elif mode == "PIL":
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    else:
        raise ValueError("Unsupported mode. Use 'PIL' or 'base64'.")

def caption_keep_or_delete_prompt(sentence):
    """
    Generates prompts to verify if a caption accurately reflects an image.

    Args:
        sentence (str): The caption to be evaluated.

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

    user_prompt = f"The given caption: {sentence}\n"

    return system_prompt, user_prompt

    return system_prompt, user_prompt

# --- Configuration Class ---

class ReviewConfig:
    """Manages all configuration settings for the review process."""
    BASE_URL = "PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE"
    SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
    CHINESE_RE = re.compile(r'[\u4E00-\u9FFF]')
    
    # CSV fields for the output file
    CSV_FIELDS = [
        "image_id", "image_path", "landuse", "old_caption", "caption", "geohash"
    ]

    def __init__(self, args: argparse.Namespace):
        self.model_id: str = args.model
        self.csv_patterns: List[str] = args.csv
        self.batch_size: int = args.batch
        self.max_new_tokens: int = args.max_new_tokens
        self.temperature: float = args.temperature
        self.top_p: float = args.top_p
        self.num_threads: int = args.threads

# --- Helper Functions ---

def has_chinese(text: str) -> bool:
    """Checks if a string contains Chinese characters."""
    return bool(ReviewConfig.CHINESE_RE.search(text))

def split_and_clean_sentences(text: str) -> List[str]:
    """Splits text into clean, valid sentences."""
    sentences = [s.strip() for s in ReviewConfig.SENTENCE_RE.split(text.strip()) if s.strip()]
    # Filter out sentences containing Chinese characters or specific keywords
    return [
        s for s in sentences
        if not has_chinese(s) and 'caption' not in s.lower() and 'addcriterion' not in s.lower()
    ]

def fetch_images_parallel(urls: List[str], num_threads: int) -> List[Image.Image]:
    """Downloads and decodes images in parallel, returning PIL.Image or None on failure."""
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        return list(pool.map(lambda u: load_image(u, mode="PIL"), urls))

def get_output_path(input_path: str) -> str:
    """Generates a standardized output CSV path from an input path."""
    return input_path.replace("_add_extra.csv", "_reviewed.csv")

# --- Core Logic Functions ---

def load_and_filter_data(csv_in: str, csv_out: str) -> pd.DataFrame:
    """Loads input CSV and filters out rows that are already processed."""
    df = pd.read_csv(csv_in)
    if os.path.exists(csv_out):
        with open(csv_out, newline="", encoding="utf-8") as fp:
            processed_ids = {row["image_id"] for row in csv.DictReader(fp) if row.get("caption", "").strip()}
        df = df[~df["image_id"].isin(processed_ids)]
    return df

def create_sentence_tasks(df: pd.DataFrame) -> Tuple[List[Dict], Dict]:
    """Creates a flat list of sentence tasks and a tracker for row state."""
    tasks = []
    tracker = {}
    for idx, row in df.iterrows():
        sentences = split_and_clean_sentences(row["caption"])
        if not sentences:
            continue
        tracker[idx] = {"total": len(sentences), "done": 0, "kept": []}
        for sentence in sentences:
            tasks.append({"row_idx": idx, "sentence": sentence, "url": f"{ReviewConfig.BASE_URL}{row['image_path'].strip()}"})
    return tasks, tracker

def process_batch(batch: List[Dict], model, processor, config: ReviewConfig) -> List[Dict]:
    """Processes a single batch: fetches images, runs inference, and returns decisions."""
    results = []
    images = fetch_images_parallel([task["url"] for task in batch], config.num_threads)
    
    valid_tasks = []
    prompts = []
    images_in = []

    for i, task in enumerate(batch):
        img = images[i]
        if img is None:
            # Treat image load failure as a "delete" decision for this sentence
            results.append({"row_idx": task["row_idx"], "sentence": task["sentence"], "decision": "delete"})
            continue
            
        sys_prompt, user_prompt = caption_keep_or_delete_prompt(task["sentence"])
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_prompt}]},
        ]
        prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        images_in.append(img)
        valid_tasks.append(task)

    if not prompts:
        return results

    # Run model inference
    inputs = processor(text=prompts, images=images_in, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=config.max_new_tokens, temperature=config.temperature, top_p=config.top_p
        )
    
    decoded_outputs = processor.batch_decode(
        [out[len(inp):] for out, inp in zip(outputs, inputs.input_ids)], skip_special_tokens=True
    )

    # Process decisions
    for i, decision in enumerate(decoded_outputs):
        task = valid_tasks[i]
        final_decision = "keep" if decision.strip().lower() == "keep" else "delete"
        results.append({"row_idx": task["row_idx"], "sentence": task["sentence"], "decision": final_decision})
        
    return results

def review_file(csv_in: str, csv_out: str, model, processor, config: ReviewConfig):
    """Orchestrates the caption review process for a single file."""
    df = load_and_filter_data(csv_in, csv_out)
    if df.empty:
        print(f"✅  {os.path.basename(csv_in)} is already fully processed.")
        return

    tasks, tracker = create_sentence_tasks(df)
    if not tasks:
        print(f"✓  No valid sentences to process in {os.path.basename(csv_in)}.")
        return

    with open_csv_writer(csv_out, ReviewConfig.CSV_FIELDS) as (writer, fp), \
         tqdm(total=len(tasks), unit="sent", desc=os.path.basename(csv_in)) as pbar:
        
        for i in range(0, len(tasks), config.batch_size):
            batch = tasks[i : i + config.batch_size]
            batch_results = process_batch(batch, model, processor, config)

            # Update tracker and commit completed rows
            for result in batch_results:
                row_idx = result["row_idx"]
                if row_idx not in tracker: continue

                tracker[row_idx]["done"] += 1
                if result["decision"] == "keep":
                    tracker[row_idx]["kept"].append(result["sentence"])
                
                # Check if all sentences for this row are processed
                if tracker[row_idx]["done"] == tracker[row_idx]["total"]:
                    row = df.loc[row_idx]
                    writer.writerow({
                        "image_id": row["image_id"], "image_path": row["image_path"],
                        "landuse": row["landuse"], "old_caption": row["caption"],
                        "caption": " ".join(tracker[row_idx]["kept"]),
                        "geometry_geohash": row["geometry_geohash"],
                    })
                    fp.flush()
                    del tracker[row_idx] # Free up memory
            
            pbar.update(len(batch))
            
    print(f"✓  Finished processing {os.path.basename(csv_in)}. Results are in {csv_out}")

# --- Main Execution ---

def main():
    """Parses arguments, initializes the model, and starts the review process."""
    parser = argparse.ArgumentParser(description="Crash-resistant, deduplicated caption-review pipeline for Qwen-VL.")
    parser.add_argument("--model", required=True, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the pre-trained model.")
    parser.add_argument("--csv", required=True, default="caption_ft_train.csv_add_extra.csv", help="One CSV file path.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for processing.")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Max new tokens for generation.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads for parallel image fetching.")
    
    args = parser.parse_args()
    config = ReviewConfig(args)

    print(f"Loading model: {config.model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    ).eval()
    model = torch.compile(model)

    processor = AutoProcessor.from_pretrained(config.model_id)
    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    for pattern in config.csv_patterns:
        for csv_in_path in glob.glob(pattern):
            csv_out_path = get_output_path(csv_in_path)
            print(f"\n--- Starting file: {os.path.basename(csv_in_path)} ---")
            review_file(csv_in_path, csv_out_path, model, processor, config)

if __name__ == "__main__":
    main()