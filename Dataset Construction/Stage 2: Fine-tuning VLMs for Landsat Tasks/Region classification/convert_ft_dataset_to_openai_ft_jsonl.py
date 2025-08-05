import json
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

# Import shared utilities
# from shared_utils.shared_utils import build_prompts
# from prompt_folder.rs_prompt_list import prompt_version_name

# --- Configuration ---
BATCH_SIZE = 1000
IMAGE_DATASET_FOLDER = "BETTER_SETUP_A_SHAREDRIVER_URL_FOR_OEPNAI_FINETUNE"
REGION_CLS_GT_PATH = "region_cls_ft_train.csv"
OUTPUT_FILENAME = "region_cls_ft_trai_batch.jsonl"


# --- Helper Functions ---

def region_classification_prompt():
    """Generates prompts for detailed land cover classification."""
    system_prompt = """You are an advanced assistant for analyzing an optical satellite image. Your role is using the information from image to accurate answers to the questions to the scene.
Analyze an optical satellite image to classify land cover types. Focus on six classifications: Cultivated Terrestrial Vegetation, Natural Terrestrial Vegetation, Natural Aquatic Vegetation, Artificial Surface, Natural Bare Surfaces, and Water. Pay particular attention to Cultivated Terrestrial Vegetation, Artificial Surface, and Water.
"""

    user_prompt_txt = """Answer these questions:
1. What land cover classifications can be found in the image?
2. Divide the image into five sections: top-left, bottom-left, top-right, bottom-right, and center. For each, list classifications in order of area occupied.
Use this structured format as output:
{'Land Cover Classifications in Optical Image': [list], 'Top-Left Area': [list], 'Top-Right Area': [list], 'Bottom-Left Area': [list], 'Bottom-Right Area': [list], 'Centre Area': [list]}

Examples
{"Land Cover Classifications in Optical Image": ["Natural Terrestrial Vegetation", "Cultivated Terrestrial Vegetation", "Artificial Surface"],"Top-Left Area": ["Cultivated Terrestrial Vegetation", "Artificial Surface"], "Top-Right Area": ["Natural Terrestrial Vegetation", "Cultivated Terrestrial Vegetation"], "Bottom-Left Area": ["Cultivated Terrestrial Vegetation", "Natural Bare Surface"], "Bottom-Right Area": ["Natural Terrestrial Vegetation", "Natural Bare Surface"], "Centre Area": ["Natural Bare Surface"]}"""

    return system_prompt, user_prompt_txt

def validate_jsonl_file(file_path: Path):
    """
    Validates a JSONL file for required format and content.

    Args:
        file_path: The path to the JSONL file.
    """
    format_errors = defaultdict(int)
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                format_errors[f"line_{i}_invalid_json"] += 1
                continue

            if not isinstance(ex, dict) or "messages" not in ex:
                format_errors["missing_messages_list"] += 1
                continue

            for message in ex["messages"]:
                if "role" not in message or "content" not in message:
                    format_errors["message_missing_key"] += 1
                if message.get("role") not in ("system", "user", "assistant"):
                    format_errors["unrecognized_role"] += 1

            if not any(msg.get("role") == "assistant" for msg in ex["messages"]):
                format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for error, count in format_errors.items():
            print(f"- {error}: {count}")
    else:
        print("No errors found. The file is valid.")


def create_message(
    system_prompt: str,
    user_prompt: str,
    image_path: str,
    caption: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Creates a message dictionary for a single data entry.

    Args:
        system_prompt: The system prompt text.
        user_prompt: The user prompt text.
        image_path: The full URL to the image.
        caption: The expected caption for the image.

    Returns:
        A dictionary containing the formatted message list.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    image_content = [{"type": "image_url", "image_url": {"url": image_path}}]
    messages.append({"role": "user", "content": json.dumps(image_content)})
    messages.append({"role": "assistant", "content": caption})
    return {"messages": messages}


# --- Main Function ---

def generate_fine_tune_batch(
    image_folder: str,
    gt_csv_path: Path,
    output_filename: Path,
    batch_size: int,
):
    """
    Generates a JSONL file for fine-tuning based on a CSV of ground truth data.
    """
    try:
        gt_df = pd.read_csv(gt_csv_path)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_csv_path}")
        return

    system_prompt, user_prompt = region_classification_prompt()
    messages_to_write = []

    batch_df = gt_df.head(batch_size)
    for _, row in batch_df.iterrows():
        full_image_path = f"{image_folder}{row['image_path']}"
        message = create_message(
            system_prompt,
            user_prompt,
            full_image_path,
            str(row["gt_region_cls"]),
        )
        messages_to_write.append(message)

    with open(output_filename, "w", encoding="utf-8") as f:
        for message in messages_to_write:
            f.write(json.dumps(message) + "\n")

    print(f"Generated {len(messages_to_write)} messages in {output_filename}")
    validate_jsonl_file(output_filename)


if __name__ == "__main__":
    generate_fine_tune_batch(
        IMAGE_DATASET_FOLDER,
        REGION_CLS_GT_PATH,
        OUTPUT_FILENAME,
        BATCH_SIZE,
    )