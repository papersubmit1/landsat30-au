#!/usr/bin/env python3
"""
glm4v_batch_vqa.py
──────────────────
Batch VQA inference with the GLM‑4.1V‑9B‑Thinking model (image‑text‑to‑text).

Usage
-----
python glm4v_batch_vqa.py \
    --csv  Landsat30-AU-VQA-test.csv \
    --out  results/glm4v_answers.csv
"""

import os
import math
import csv
import argparse
from pathlib import Path
from typing import List

import torch
import pandas as pd
from tqdm import tqdm
from transformers import Glm4vForConditionalGeneration, AutoProcessor


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


BASE_URL = (
    "PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE"
)

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


def generate_answers_glmv(
    input_csv: Path,
    output_csv: Path,
    model,
    processor,
):
    """Iterate through a VQA CSV and write model answers."""
    df = pd.read_csv(input_csv, keep_default_na=False)

    # avoid double work if we resume
    processed_ids = set()
    if output_csv.exists():
        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            processed_ids = {row["qa_id"] for row in reader if row.get("answer")}
        df = df[~df["qa_id"].isin(processed_ids)]

    fieldnames = ["qa_id", "image_path", "question", "answer", "options", "gt_answer"]

    df = df[1:]

    with open_csv_writer(output_csv, fieldnames) as (writer, csvfile):
        progress = tqdm(total=len(df), unit="sample", desc="GLM‑4V VQA")
        try:
            for _, row in df.iterrows():
                # 1️⃣ build prompt
                question, options = row["question"], row["options"]
                system_prompt, user_prompt = run_vlm_by_vqa(question, options)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": None},  # placeholder
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ]

                # 2️⃣ load image & replace placeholder
                full_url = BASE_URL + row["image_path"].strip()
                image = load_image(full_url, mode="PIL")
                if image is None:
                    print(f"[WARN] could not read {full_url}")
                    progress.update()
                    continue
                messages[1]["content"][0]["image"] = image

                # 3️⃣ tokenise
                chat_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[chat_text],
                    images=[image],  # vision handled internally
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                # 4️⃣ generation (no_grad)
                with torch.inference_mode():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=8192,
                    )

                # 5️⃣ decode (strip the prompt part)
                answer_ids = gen_ids[:, inputs.input_ids.shape[1] :]
                answer = (
                    processor.batch_decode(
                        answer_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    .replace("\n", " ")
                    .strip()
                )

                writer.writerow(
                    {
                        "qa_id": row["qa_id"],
                        "image_path": row["image_path"],
                        "question": question,
                        "answer": answer,
                        "options": options,
                        "gt_answer": row["answer"],
                    }
                )
                csvfile.flush()
                progress.update()
        except KeyboardInterrupt:
            print("\n[INFO] interrupted – partial results saved.")
        finally:
            progress.close()
    print(f"[✓] Results written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    input_csv = "Landsat30-AU-VQA-test.csv"
    output_csv = "vqa_answer/GLM_vqa_zero_shot.csv"

    MODEL_NAME = "THUDM/GLM-4.1V-9B-Thinking"
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    ).eval()
    model = torch.compile(model)  # optional, Torch ≥2.4

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    generate_answers_glmv(
        Path(input_csv),
        Path(output_csv),
        model,
        processor,
    )
