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
REGION_CLS_GT_PATH = "caption_ft_train.csv"
OUTPUT_FILENAME = "caption_ft_train_batch.jsonl"


# --- Helper Functions ---

def image_captioning_prompt(landcover, landuse):
    """
    Generates prompts for creating a detailed caption from image and metadata.

    Args:
        landcover (str): A string containing land cover information.
        landuse (str): A string containing land use information.

    Returns:
        tuple: A tuple containing the system prompt and the user prompt.
    """
    system_prompt = """Generate a detailed and concise caption from an optical satellite image using provided metadata.

* Please use image content and land cover information to cross-validate land use information. Please use verifed land use information to finish the caption.
* Identify all visible water bodies: rivers (describe their path), lakes, ponds; mention locations and relative sizes using area cues (30m x 30m per pixel).
* Distinguish dominant land cover in each area (bare surface, vegetated, cropland), specify approximate extent or pattern if possible.
* Identify and size artificial areas (“small town”, “city”) and reference exact locations (e.g., “Top-Left Area”) when relevant.
* Describe visible urban features only if seen or confirmed in metadata.
* Note any irrigated field patterns.
* Describe road corridors, specifying directions and links to urban areas, if visible.
* Include any spatial references for features (top, bottom, left, right, center).
* Summarize the balance and dominance between bare and vegetated surfaces.
* Use the land use metadata to offer insights on overall landscape use.
* Incorporate color information for the overall image or specific areas (e.g., "The forests appear dark green," or "The river reflects shades of blue"), describing observed hues and any notable color patterns.
* Caption should be in plain text, clear, and concise without markdown or line breaks."""

    user_prompt = (
        "The following are the metadata to this satellite image:\n"
        "Land Cover Information:\n"
        f"{landcover}\n"
        "Land Use Information:\n"
        f"{landuse}"
    )

    return system_prompt, user_prompt

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

    messages_to_write = []

    batch_df = gt_df.head(batch_size)
    for _, row in batch_df.iterrows():
        full_image_path = f"{image_folder}{row['image_path']}"
        landcover = row['region_cls']
        landuse = row['landuse']
        
        system_prompt, user_prompt = image_captioning_prompt(landcover, landuse)

        message = create_message(
            system_prompt,
            user_prompt,
            full_image_path,
            str(row["caption"]),
        )
        messages_to_write.append(message)

    with open(output_filename, "w", encoding="utf-8") as f:
        for message in messages_to_write:
            f.write(json.dumps(message) + "\n")

    print(f"Generated {len(messages_to_write)} messages in {output_filename}")


if __name__ == "__main__":
    generate_fine_tune_batch(
        IMAGE_DATASET_FOLDER,
        REGION_CLS_GT_PATH,
        OUTPUT_FILENAME,
        BATCH_SIZE,
    )