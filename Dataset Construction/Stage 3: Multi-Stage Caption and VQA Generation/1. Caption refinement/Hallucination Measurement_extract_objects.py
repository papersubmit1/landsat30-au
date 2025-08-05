#!/usr/bin/env python3
"""
Refactored multiprocess script for extracting key objects from captions using the OpenAI API.
This version enhances modularity, centralizes configuration, and clarifies the data processing pipeline.
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import httpx
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

def extract_key_objects_prompt(given_caption):
    """
    Generates prompts to extract key earth observation objects from a caption.

    Args:
        given_caption (str): The caption from which to extract objects.

    Returns:
        tuple: A tuple containing the system prompt and the user prompt.
    """
    system_prompt = """Extract the key objects directly from the provided caption, focusing on earth observation elements such as natural features, human-made structures, and land use areas. These objects must explicitly appear in the caption and should emphasize notable earth science patterns like oxbow bends.

# Steps

1. Carefully read the provided caption, identifying each object explicitly mentioned.
2. Cross-check each identified object to ensure it directly appears in the caption and falls under categories related to earth observation such as natural features, human-made structures, or land use areas.
3. Give particular attention to identifying and naming distinct natural patterns, such as oxbow bends, formed by meandering rivers.
4. Compile validated objects into a list format.

# Output Format

- Return a JSON array containing strings of all identified key objects relevant to earth observation, directly extracted from the caption.

# Examples

### Example 1
**Input:** "The image shows a large river bending through a dense forest with a small urban area visible on the horizon."
**Output:** ["river", "forest", "urban area"]

### Example 2
**Input:** "A solar farm bordered by a highway with adjacent cropland and a small lake."
**Output:** ["solar farm", "highway", "cropland", "lake"]

### Example 3
**Input:** "Mountains rise in the distance beyond stretches of desert and a nearby reservoir."
**Output:** ["mountains", "desert", "reservoir"]

### Example 4
**Input:** "The landscape is dominated by natural vegetation, featuring oxbow bends in the river path, with cultivated fields and wetlands nearby."
**Output:** ["natural vegetation", "oxbow bends", "river", "cultivated fields", "wetlands"]

# Notes

- Only include objects explicitly mentioned in the caption.
- If an object does not appear word-for-word in the caption, it should be omitted.
- Pay special attention to terminology and synonyms that may describe earth observation features but ensure they appear exactly as in the caption."""

    user_prompt = f"The given caption: {given_caption}\n"

    return system_prompt, user_prompt

# --- Configuration ---

class Config:
    """A single class to manage all configuration and settings."""
    # --- Model and API Parameters ---
    MODEL_NAME = "gpt-4.1"
    MAX_NEW_TOKENS = 560
    TEMPERATURE = 0.0
    TOP_P = 0.9
    API_TIMEOUT = httpx.Timeout(30.0, read=20.0, write=15.0, connect=6.0)

    # --- File Paths ---
    BASE_DIR = Path("PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE")
    
    # Input files for processing
    CAPTION_FILES: List[Path] = [ "caption_ft_train_reviewed.csv"
        # Add other files as needed
    ]

    # --- Multiprocessing Settings ---
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", mp.cpu_count()))

# --- Worker and Helper Functions ---

def load_seen_ids(csv_path: Path) -> set:
    """Collects 'image_id's from an existing JSONL file to allow resuming a run."""
    if not csv_path.is_file():
        return set()
    
    seen = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                seen.add(str(record["image_id"]))
            except json.JSONDecodeError as e:
                sys.stderr.write(f"Error decoding JSON line: {e}\n")

    return seen

def _init_worker(api_timeout: httpx.Timeout):
    """Initializes an OpenAI client for each worker process."""
    global oai_client
    oai_client = OpenAI(timeout=api_timeout)

def _process_row(task_args: Tuple) -> Tuple[str, Dict[str, Any]] | None:
    """
    Worker function to process a single row: calls the OpenAI API and formats the result.
    """
    row_dict, model_name, max_tokens, temp, top_p = task_args
    try:
        image_id = str(row_dict["image_id"])
        given_caption = row_dict["caption"]

        system_prompt, user_prompt = extract_key_objects_prompt(given_caption)
        
        response = oai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
        )
        key_objects = response.choices[0].message.content
        
        recording = {
            "image_id": image_id,
            "caption": given_caption,
            "key_objects": key_objects,
        }
        return image_id, recording
        
    except Exception as e:
        sys.stderr.write(f"Error processing item (ID: {row_dict.get('image_id', 'N/A')}): {e}\n")
        return None

# --- Main Pipeline Functions ---

def run_extraction_pipeline(input_path: Path, config: Config):
    """
    Manages the multiprocessing pool to extract key objects for a given input file.
    
    Args:
        input_path: Path to the input CSV file.
        config: The configuration object.
    """
    output_path = f"{input_path.stem}_key_object.csv"
    
    df = pd.read_csv(input_path)
    seen_ids = load_seen_ids(output_path)

    tasks = [
        (row.to_dict(), config.MODEL_NAME, config.MAX_NEW_TOKENS, config.TEMPERATURE, config.TOP_P)
        for _, row in df.iterrows() if str(row["image_id"]) not in seen_ids
    ]

    if not tasks:
        print(f"✅ No new items to process in {input_path.name}.")
        return

    print(f"--- Starting extraction for {input_path.name} ---")
    with mp.Pool(processes=config.NUM_WORKERS, initializer=_init_worker, initargs=(config.API_TIMEOUT,)) as pool, \
         open(output_path, "a", encoding="utf-8") as f, \
         tqdm(total=len(tasks), desc=f"Extracting ({config.NUM_WORKERS} workers)") as pbar:
        
        for result in pool.imap_unordered(_process_row, tasks, chunksize=4):
            pbar.update(1)
            if result:
                _, recording = result
                f.write(json.dumps(recording, ensure_ascii=False) + "\n")
                f.flush()

# --- Entry Point ---

def main():
    """
    Main function to orchestrate the two-stage pipeline:
    1. Extract key objects from captions via OpenAI API.
    2. Merge the extracted objects with a ground truth file.
    """
    config = Config()

    # --- Stage 1: Extraction ---
    for input_path in config.CAPTION_FILES:
        run_extraction_pipeline(input_path, config)
    
if __name__ == "__main__":
    main()