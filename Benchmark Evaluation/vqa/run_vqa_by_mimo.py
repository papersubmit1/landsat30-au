#!/usr/bin/env python
import os
import pandas as pd
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # Utility to process visual inputs
from tqdm import tqdm
from pathlib import Path

import csv

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
    

# Base URL to form the full image URL from the CSV field.
BASE_URL = (
    "PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE"
)


def run_vlm_by_vqa(question: str, options: str):
    sys = (
        "You are an evaluation agent for remote–sensing VQA.\n"
        "Your ONLY job is to look at a satellite image, read the multiple‑choice question and its options, "
        "and pick exactly ONE best answer. The image pixel resolution is 30 m.\n\n"
        "Output rules\n────────────\nReturn **only** the text in current option.\n• Do **NOT** output words, punctuation, or explanations.\n• Trim whitespace.\n"
    )
    usr = f"<image>\nQuestion:\n{question}\n\nOptions: {options}\n"
    return sys, usr

def generate_caption_mimo(
    region_cls_result_csv_file, output_csv_path, model, processor
):
    """
    Process a CSV file of metadata, load images from remote storage, build prompts
    using the standard prompt template, and generate captions using Qwen2.5-VL.
    """
    # Load the CSV file
    df = pd.read_csv(region_cls_result_csv_file, keep_default_na=False)

    fieldnames = [
        "qa_id",
        "image_path",
        "question",
        "answer",
        "options",
        "gt_answer",
    ]

    # Gather already‑processed IDs
    if os.path.exists(output_csv_path):
        processed_ids = set()
        with open(output_csv_path, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("answer", "").strip():
                    processed_ids.add(row["qa_id"])
        df = df[~df["qa_id"].isin(processed_ids)]

    # Open CSV in append mode (or create new file) and process features.
    with open_csv_writer(output_csv_path, fieldnames) as (writer, csvfile):
        with tqdm(
            total=len(df),
            desc="MiMo-VL VQA",
            unit="feature",
        ) as overall_pbar:
            try:
                for idx, row in df.iterrows():
                    image_path = row["image_path"]
                    question = row["question"]
                    options = row["options"]

                    system_prompt, user_prompt_txt = run_vlm_by_vqa(question, options)

                    # Load the RGB image using the full image URL.
                    full_image_url = BASE_URL + image_path.strip()
                    rgb_image = load_image(full_image_url, mode="PIL")
                    if rgb_image is None:
                        print(f"Could not load image at {full_image_url}")
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
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=8192,
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

                    answer = output_text[0]
                    # generated_captions.append(caption)
                    answer = answer.replace("\n", " ").replace("  ", " ")

                    row = {
                        "qa_id": row["qa_id"],
                        "image_path": image_path,
                        "question": question,
                        "answer": answer,
                        "options": options,
                        "gt_answer": row["answer"],
                    }

                    writer.writerow(row)
                    csvfile.flush()
                    overall_pbar.update(1)
            except KeyboardInterrupt:
                print(
                    "\nKeyboardInterrupt detected. Terminating processing early. CSV file has been updated with current data."
                )
                return

    print(f"Results saved to {output_csv_path}")


if __name__ == "__main__":
    for model_name in [
        "XiaomiMiMo/MiMo-VL-7B-RL",
    ]:
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

        human_read_able_name = "vqa_answer"

        # Define input and output CSV file paths.
        region_cls_result_csv_file = "Landsat30-AU-VQA-test.csv"
        output_csv_path = f"mimo_vqa_zero_shot.csv"

        # Generate captions using the Qwen model.
        generate_caption_mimo(
            region_cls_result_csv_file,
            output_csv_path,
            model,
            processor,
        )
