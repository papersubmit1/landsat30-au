#!/usr/bin/env python3

import os
import csv
import math
import pandas as pd
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError  

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

EVALUATE_VLM_CAPTION_SYSTEM_PROMPT = """You are an expert model for describing satellite or aerial images of landscapes, 
where each image pixel represents a 30-meter ground resolution. Use detailed, domain-specific language 
to describe the visible land covers, features, surface types (e.g., vegetation, artificial surfaces, water, etc.), 
and spatial relationships appropriate for the given spatial scale. Your goal is to give an analytical, objective 
caption that covers both the dominant and minor elements in the image, referencing spatial orientation 
(top-left, center, etc.) and notable connections (such as roads, patch boundaries, etc.). 
Base your descriptions only on observable features in the image, keeping the pixel resolution in mind."""

EVALUATE_VLM_CAPTION_USER_PROMPT = """Each image pixel corresponds to 30 meters on the ground. Respond in plain 
text only, with no formatting, lists, or special markup—just a single paragraph. 
Now, describe the following image in the same detailed manner, considering that each pixel represents 
30 meters:"""

def evaluate_vlm_caption_zero_shot():
    """Returns prompts for zero-shot VLM caption evaluation."""
    system_prompt = EVALUATE_VLM_CAPTION_SYSTEM_PROMPT
    user_prompt = EVALUATE_VLM_CAPTION_USER_PROMPT
    return system_prompt, user_prompt


def evaluate_vlm_caption_one_shot():
    """Returns prompts and an example for one-shot VLM caption evaluation."""
    system_prompt = EVALUATE_VLM_CAPTION_SYSTEM_PROMPT
    user_prompt = EVALUATE_VLM_CAPTION_USER_PROMPT

    # Note: 'VLM_TO_CAPTION_USER_PROMPT' was in the original code. This might be a
    # typo for 'EVALUATE_VLM_CAPTION_USER_PROMPT'. It is left as is per instructions.
    shot_blocks = [
        {
            "image_path": "DEA_VLM_images/ga_ls9c_ard_3-x60y28-2024-patches/ga_ls9c_ard_3-x60y28-2024-r331nmx-2024-07-29-raw.png",
            "question": EVALUATE_VLM_CAPTION_USER_PROMPT,
            "caption": "The image shows a landscape dominated by dark green natural terrestrial vegetation, covering most of the area. Patches of cultivated terrestrial vegetation are visible in the top-left, bottom-left, and center areas, appearing as lighter green or brownish zones. Artificial surfaces, likely small clearings or structures, are present in the top-right and bottom-right areas, with one distinct light patch in the bottom-right. The overall balance is heavily in favor of vegetated surfaces, with artificial and bare areas being minor. The color palette is dominated by dark greens with occasional lighter and brownish patches. The road corridor in the top-right connects directly to the artificial surface and cultivated vegetation patches, forming a clear link between infrastructure and land use zones. A narrow, winding path or track is visible near the bottom-center, cutting through the vegetation and extending slightly into the middle section of the image.",
        }
    ]
    return shot_blocks, system_prompt, user_prompt


def cloud_filter_prompt():
    """Generates prompts to classify a satellite image as 'cloudy' or 'clear'."""
    system_prompt = """You are an advanced assistant specializing in analyzing optical satellite images. Your task is to classify each satellite image as either "cloudy" or "clear".

Definitions:
Cloudy: The majority of the image is covered by clouds, obscuring most of the Earth's surface, OR if the image is dominated by features or artifacts (such as sensor bands, stripes, or areas with missing data) that prevent a clear view of ground features.
Clear: The image is mostly free of clouds, and the surface of the Earth is clearly visible.

Instructions:

If clouds or visual obstructions (e.g. striping, missing data, sensor artifacts, over-exposure) cover most of the image and you cannot clearly see the ground features, classify as "cloudy".
If the ground and surface features are mostly visible, classify as "clear".
Respond only with "cloudy" or "clear"."""

    user_prompt = "Please classify the image as either 'cloudy' or 'clear'."

    return system_prompt, user_prompt

# Base URL to form the full image URL from the CSV field.
BASE_URL = (
    "PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE"
)


def generate_cloud_filter_result_by_qwen(
    raw_result_file,
    output_csv_path,
    model,
    processor,
    max_new_tokens,
    temperature,
    top_p,
):
    """
    Process a CSV file of metadata, load images from remote storage, build prompts
    using the standard prompt template, and generate captions using Qwen2.5-VL.
    """
    # Load the CSV file
    df = pd.read_csv(raw_result_file)


    fieldnames = [
        "image_id",
        "image_path",
        "cloud_filter",
    ]

    # Gather already-processed IDs
    if os.path.exists(output_csv_path):
        processed_ids = set()
        with open(output_csv_path, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("cloud_filter", "").strip():
                    processed_ids.add(row["image_id"])
        df = df[~df["image_url"].isin(processed_ids)]

    # Open CSV in append mode (or create new file) and process features.
    with open_csv_writer(output_csv_path, fieldnames) as (writer, csvfile):
        with tqdm(
            total=len(df),
            desc="Processing GeoJSON files",
            unit="feature",
        ) as overall_pbar:
            for idx, row in df.iterrows():
                image_id = row["image_url"]
                image_path = row["image_url"]

                system_prompt, user_prompt_txt = cloud_filter_prompt()

                # Load the RGB image using the full image URL.
                full_image_url = BASE_URL + image_path.strip()

                try:
                    rgb_image = load_image(full_image_url, mode="PIL")

                    if rgb_image is None:
                        print(
                            f"Could not load image at {full_image_url} (load_image returned None)"
                        )
                        overall_pbar.update(1)
                        continue

                    # Build messages list for Qwen.
                    # Here we assume a single image is used (adjust if you need two images).
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": rgb_image},
                                {"type": "text", "text": user_prompt_txt},
                            ],
                        },
                    ]

                    # Prepare the text prompt using the processor's chat template.
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    # Process visual inputs from the messages.
                    image_inputs, video_inputs = process_vision_info(messages)

                    # Build final inputs for the model.
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(model.device)

                    with torch.inference_mode():
                        # Generate caption using Qwen2.5-VL.
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    # Optional: remove prompt tokens from the generated output.
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    cloud_filter = output_text[0]
                    # generated_captions.append(caption)
                    cloud_filter = cloud_filter.replace("\n", " ").replace("  ", " ")

                    row_to_write = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "cloud_filter": cloud_filter,
                    }

                    writer.writerow(row_to_write)
                    csvfile.flush()

                except (UnidentifiedImageError, IOError, Exception) as e:
                    print(f"Error loading or processing image {full_image_url}: {e}")
                finally:
                    overall_pbar.update(1) 

            try:
                pass
            except KeyboardInterrupt:
                print(
                    "\nKeyboardInterrupt detected. Terminating processing early. CSV file has been updated with current data."
                )
                return

    print(f"Results saved to {output_csv_path}")


# ---------- Main ----------
if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_new_tokens = 10

    # Load the Qwen2.5-VL model and its processor.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )

    model.eval()  # >>> ADDED — turn off training-mode bookkeeping
    model = torch.compile(model)  # >>> ADDED — fuse kernels for faster inference

    processor = AutoProcessor.from_pretrained(model_name)

    for temperature in [
        0.000001,
    ]:
        max_new_tokens = 560
        top_p = 1.0

        region_result_files = [
            "caption_gt/captioning_ft_full.csv"
        ]

        for region_result_file in region_result_files:
            output_csv_path = f"cloud_filter_result/{region_result_file.split('/')[-1].replace('.csv', '_cloud_filter.csv')}"

            # Generate captions using the Qwen model.
            generate_cloud_filter_result_by_qwen(
                region_result_file,
                output_csv_path,
                model,
                processor,
                max_new_tokens,
                temperature,
                top_p,
            )
