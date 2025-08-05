#!/usr/bin/env python
import os
import csv
from pathlib import Path
from typing import List, Dict, Any, Callable

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

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


def add_missing_object_prompt(caption):
    """
    Generates prompts to identify missing objects in a caption.

    Args:
        caption (str): The existing image caption.

    Returns:
        tuple: A tuple containing the system prompt and the user prompt.
    """
    system_prompt = """You are an advanced assistant specializing in analyzing optical satellite images. You will get a caption about one image. Your task is to find and describe special missing patterns or objects that appear in the image but not in caption.

Only describe what is clearly visible - do NOT mention anything that is absent or not shown in the image. Avoid making statements about what is not present.

Do not start the response with phrases like "In addition to the features described in the caption," or similar wording—just directly state the missing object information.

Mention only one key missing object or pattern that is clearly visible in the image but not in the caption; keep it concise and ideally contained within a single sentence.

This will instruct me to avoid referencing absent features in my responses."""

    user_prompt = f"The given caption: {caption}\n"

    return system_prompt, user_prompt


def add_missing_connection_prompt(caption):
    """
    Generates prompts to identify missing connections between objects in a caption.

    Args:
        caption (str): The existing image caption.

    Returns:
        tuple: A tuple containing the system prompt and the user prompt.
    """
    system_prompt = """You are an advanced assistant specializing in analyzing optical satellite images. You will get a caption about one image. Your task is to find and describe special missing connections between objects that appear in the image but not in caption.

Only describe what is clearly visible - do NOT mention anything that is absent or not shown in the image. Avoid making statements about what is not present.

Do not start the response with phrases like "In addition to the features described in the caption," or similar wording—just directly state the missing connection or relationship information.

Mention only one key missing connection or relationship that is clearly visible in the image but not in the caption; keep it concise and ideally contained within a single sentence.

This will instruct me to avoid referencing absent features in my responses."""

    user_prompt = f"The given caption: {caption}\n"

    return system_prompt, user_prompt

# --- Configuration ---

class Config:
    """Manages all hyperparameters, paths, and settings."""
    # --- Model and Generation Parameters ---
    BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    MAX_NEW_TOKENS = 128
    TEMPERATURE = 0.7
    TOP_P = 0.8
    BATCH_SIZE = 8
    FLUSH_INTERVAL = 10

    # --- Paths and URLs ---
    BASE_URL = (
        "PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE"
    )
    INPUT_DATA_PATH = Path("caption_gt/captioning_ft_full.csv")

    # --- CSV Fieldnames ---
    FIELDNAMES = [
        "image_id", "image_path", "landuse", "old_caption",
        "caption", "geohash"
    ]

# --- Helper Functions ---

def load_and_prepare_dataframe(processing_file: Path, output_csv_path: Path) -> pd.DataFrame:
    """Loads input data, handles JSONL conversion, and filters out already processed entries."""
    if processing_file.suffix == ".csv":
        df = pd.read_csv(processing_file)
    
    print(f"Total entries in input file: {len(df)}")

    if output_csv_path.exists():
        processed_ids = pd.read_csv(output_csv_path)["image_id"].unique()
        df = df[~df["image_id"].isin(processed_ids)]
        print(f"Resuming run. {len(df)} entries remaining.")
    
    return df

def prepare_batch_data(batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Loads images for a batch and prepares the initial data structure."""
    batch_data = []
    for _, row in batch_df.iterrows():
        full_url = f"{Config.BASE_URL}{row['image_path'].strip()}"
        data_item = row.to_dict()
        try:
            rgb_image = load_image(full_url, mode="PIL")
            if rgb_image is None: raise IOError("load_image returned None")
            data_item["rgb_image"] = rgb_image
        except (UnidentifiedImageError, IOError, Exception) as e:
            print(f"Error loading image {full_url} (ID: {row['image_id']}): {e}")
            data_item["rgb_image"] = "ERROR"
        batch_data.append(data_item)
    return batch_data

def run_llm_inference_pass(
    model, processor, batch_data: List[Dict], prompt_creator: Callable
) -> List[str]:
    """Runs a single pass of LLM inference for a batch."""
    texts_for_llm, images_for_llm, valid_indices = [], [], []
    
    for i, item in enumerate(batch_data):
        if item["rgb_image"] != "ERROR":
            system_prompt, user_prompt = prompt_creator(item)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["rgb_image"]},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts_for_llm.append(text)
            images_for_llm.append(item["rgb_image"])
            valid_indices.append(i)

    generated_texts = [""] * len(batch_data)
    if not texts_for_llm:
        return generated_texts

    inputs = processor(text=texts_for_llm, images=images_for_llm, padding=True, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
        )
    
    trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    decoded_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True)

    for i, text in enumerate(decoded_text):
        original_idx = valid_indices[i]
        generated_texts[original_idx] = text.strip().replace("\n", " ")
        
    return generated_texts

# --- Main Processing Function ---

def generate_captions_batched(
    df: pd.DataFrame, output_path: Path, model, processor
):
    """Processes a DataFrame in batches to generate captions."""
    with open_csv_writer(output_path, Config.FIELDNAMES) as (writer, csvfile):
        for i_start in tqdm(range(0, len(df), Config.BATCH_SIZE), desc="Processing Batches"):
            batch_df = df.iloc[i_start : i_start + Config.BATCH_SIZE]
            
            batch_data = prepare_batch_data(batch_df)

            # First Pass: Identify missing objects
            missing_objects = run_llm_inference_pass(
                model, processor, batch_data,
                lambda item: add_missing_object_prompt(item["caption"])
            )
            for i, text in enumerate(missing_objects):
                batch_data[i]["missing_object"] = text

            # Second Pass: Identify missing connections
            missing_connections = run_llm_inference_pass(
                model, processor, batch_data,
                lambda item: add_missing_connection_prompt(
                    f"{item['caption']} {item['missing_object']}".strip()
                )
            )

            # Write results for the batch
            for i, item in enumerate(batch_data):
                if item["rgb_image"] == "ERROR":
                    final_caption = "ERROR: Image loading failed."
                else:
                    final_parts = [item["caption"], item["missing_object"], missing_connections[i]]
                    final_caption = " ".join(filter(None, final_parts)).strip()
                
                writer.writerow({
                    "image_id": item["image_id"], "image_path": item["image_path"],
                    "landuse": item["landuse"], "old_caption": item["caption"],
                    "caption": final_caption, "geohash": item["geohash"],
                })

            if (i_start // Config.BATCH_SIZE + 1) % Config.FLUSH_INTERVAL == 0:
                csvfile.flush()

    print(f"\n✅ Results saved to {output_path}")

# --- Main Execution ---

def main():
    """Main function to initialize the model and run the processing pipeline."""
    cfg = Config()

    print(f"Loading model: {cfg.BASE_MODEL_NAME}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    ).eval()
    model = torch.compile(model)
    
    processor = AutoProcessor.from_pretrained(cfg.BASE_MODEL_NAME)
    processor.tokenizer.padding_side = "left"

    processing_file = cfg.INPUT_DATA_PATH

    output_csv_path = processing_file.with_name(
        processing_file.name.replace(".csv", "_add_extra.csv")
    )

    print(f"\n--- Processing: {processing_file.name} ---")
    print(f"Output will be saved to: {output_csv_path}")
    
    df_to_process = load_and_prepare_dataframe(processing_file, output_csv_path)

    generate_captions_batched(df_to_process, output_csv_path, model, processor)

if __name__ == "__main__":
    main()