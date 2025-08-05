#!/usr/bin/env python3
"""
Refactored multiprocess script for generating VQA responses using the OpenAI API.
Supports both single-image and multi-image (time-series) processing modes
with de-duplication and streaming JSONL output.
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Callable

import httpx
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

def generate_vqa_prompt(object_list):
    """
    Generates prompts to create VQA questions from an object list.

    Args:
        object_list (list): A list of key object strings.

    Returns:
        tuple: A tuple containing the system prompt and the user prompt.
    """
    system_prompt = """You are an AI that generates multiple-choice questions based on a given image and a key object list. Your questions should focus on four aspects: scene/land-cover identification, object presence, counting, and spatial relations.

Instructions:

Given an input of an image and its associated key object list, follow these steps:

Scene/Land-Cover Identification:

Analyze the environment or landscape in the image.
Generate a multiple-choice question that helps identify or classify the scene type.
Object Presence:

Assess which key objects from the list are visible in the image.
Create a question to confirm or deny the presence of these objects.
Counting:

Count the number of specified key objects visible in the image.
Formulate a question to verify this count.
Spatial Relation:

Evaluate the spatial relationships between notable objects.
Generate a question to describe or identify these relationships.
Output Format:

Produce your output as a Python dictionary with the following structure:

{
    "questions": [
        {
            "type": "scene_land_cover",
            "question": "<question_text>",
            "choices": ["<choice_1>", "<choice_2>", "<choice_3>", "<choice_4>"],
            "answer": "<correct_choice>"
        },
        {
            "type": "object_presence",
            "question": "<question_text>",
            "choices": ["<choice_1>", "<choice_2>", "<choice_3>", "<choice_4>"],
            "answer": "<correct_choice>"
        },
        {
            "type": "counting",
            "question": "<question_text>",
            "choices": ["<choice_1>", "<choice_2>", "<choice_3>", "<choice_4>"],
            "answer": "<correct_choice>"
        },
        {
            "type": "spatial_relation",
            "question": "<question_text>",
            "choices": ["<choice_1>", "<choice_2>", "<choice_3>", "<choice_4>"],
            "answer": "<correct_choice>"
        }
    ]
}
Each question dictionary contains:
"type": The aspect being questioned.
"question": The question text.
"choices": A list of four answer choices (including three distractors and one correct answer in random order).
"answer": The correct answer (as it appears in "choices").
Additional Notes:

Do not indicate the correct answer in the choices themselves.
Ensure distractors are plausible and clearly distinct from the correct answer.
Tailor questions and choices to the context of the image and the provided key object list.
Consider possible visual ambiguities and logical contrasts in distractors.
Example Output:

{
    "questions": [
        {
            "type": "scene_land_cover",
            "question": "What type of environment is shown in the image?",
            "choices": ["Desert", "Beach", "Forest", "Mountain"],
            "answer": "Beach"
        },
        {
            "type": "object_presence",
            "question": "Which of the following objects is visible in the image?",
            "choices": ["Cactus", "Snowman", "Palm Tree", "Skyscraper"],
            "answer": "Palm Tree"
        },
        {
            "type": "counting",
            "question": "How many umbrellas are present in the image?",
            "choices": ["One", "Two", "Three", "Four"],
            "answer": "Two"
        },
        {
            "type": "spatial_relation",
            "question": "What is the spatial relation between the sea and the sand in the image?",
            "choices": [
                "The sea is above the sand",
                "The sea is next to the sand",
                "The sea is under the sand",
                "The sea is far away from the sand"
            ],
            "answer": "The sea is next to the sand"
        }
    ]
}
Use this structure for all outputs."""
    user_prompt = f"object list: [{object_list}]"

    return system_prompt, user_prompt

# --- 1. Configuration ---

class Config:
    """A single class to manage all configuration and settings."""
    # --- API and Model Parameters ---
    MODEL_NAME = "gpt-4.1"
    MAX_NEW_TOKENS_SINGLE = 560
    MAX_NEW_TOKENS_MULTI = 1024  # Often needs more tokens
    TEMPERATURE = 0.7
    TOP_P = 0.8
    API_TIMEOUT = httpx.Timeout(30.0, read=20.0, write=15.0, connect=6.0)

    # --- File Paths and URLs ---
    BASE_URL = "PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE"
    INPUT_CSV = Path("caption_ft_train.csv")
    OUTPUT_DIR = Path("vqa_result")
    
    # --- Multiprocessing ---
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", mp.cpu_count()))

# --- 2. Worker Functions ---

def _init_worker(api_timeout: httpx.Timeout):
    """Initializes an OpenAI client for each worker process."""
    global oai_client
    oai_client = OpenAI(timeout=api_timeout)

def _process_single_image_task(task_args: Tuple) -> Dict | None:
    """Worker for single-image VQA generation."""
    row_dict, config = task_args
    try:
        system_prompt, user_prompt = generate_vqa_prompt(row_dict["landuse"])
        full_image_url = f"{config.BASE_URL}{row_dict['image_path'].strip()}"
        
        response = oai_client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": full_image_url}}]},
            ],
            max_tokens=config.MAX_NEW_TOKENS_SINGLE,
            temperature=config.TEMPERATURE, top_p=config.TOP_P
        )
        
        return {
            "image_id": str(row_dict["image_id"]),
            "vqa": response.choices[0].message.content,
            "key_objects": row_dict["key_objects"],
            "caption": row_dict["caption"],
            "image_path": row_dict["image_path"],
        }
    except Exception as e:
        sys.stderr.write(f"Error in single-image worker (ID: {row_dict.get('image_id', 'N/A')}): {e}\n")
        return None

# --- 3. Task Preparation ---

def load_seen_ids(jsonl_path: Path) -> set:
    """Loads already processed image_id's from an output file to allow resumption."""
    if not jsonl_path.is_file(): return set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return {json.loads(line).get("image_id") for line in f if line.strip()}

def prepare_single_image_tasks(df: pd.DataFrame, config: Config) -> List[Tuple]:
    """Filters for single-image areas and prepares the task list."""
    df_single = df[df["multi_image_areas"] == "No"].drop_duplicates(subset=['image_id'])
    return [(row.to_dict(), config) for _, row in df_single.iterrows()]


# --- 4. Main Pipeline ---

def run_pipeline(tasks: List[Tuple], worker_function: Callable, output_path: Path, config: Config):
    """
    A generic driver for running a multiprocessing pipeline.
    
    Args:
        tasks: A list of tasks, where each task is a tuple of arguments for the worker.
        worker_function: The function to be executed by each worker process.
        output_path: The path to the output JSONL file.
        config: The main configuration object.
    """
    seen_ids = load_seen_ids(output_path)
    filtered_tasks = [task for task in tasks if str(task[0].iloc[0]['geometry_geohash'] if isinstance(task[0], pd.DataFrame) else task[0]['image_id']) not in seen_ids]

    if not filtered_tasks:
        print("✅ No new items to process.")
        return

    output_path.parent.mkdir(exist_ok=True)
    with mp.Pool(processes=config.NUM_WORKERS, initializer=_init_worker, initargs=(config.API_TIMEOUT,)) as pool, \
         open(output_path, "a", encoding="utf-8") as f, \
         tqdm(total=len(filtered_tasks), desc=f"Processing ({config.NUM_WORKERS} workers)") as pbar:
        
        for result in pool.imap_unordered(worker_function, filtered_tasks, chunksize=4):
            pbar.update(1)
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

# --- 5. Entry Point ---

def main():
    config = Config()
    df = pd.read_csv(config.INPUT_CSV)

    output_path = config.OUTPUT_DIR / "vqa_output.jsonl"
    tasks = prepare_single_image_tasks(df, config)
    run_pipeline(tasks, _process_single_image_task, output_path, config)


    print(f"✅ Pipeline finished. Results are in {output_path}")

if __name__ == "__main__":
    main()