#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import List, Tuple
from urllib.parse import quote

from shared_utils.shared_utils import (
    open_csv_writer,
)

import pandas as pd
import requests
import torch
from tqdm.auto import tqdm

try:
    # pillow‑simd gives 2–3× faster JPEG decode; falls back to normal Pillow
    from PIL import Image
except ImportError as e:
    print(
        "Pillow/Pillow‑SIMD missing — install with `pip install pillow-simd`",
        file=sys.stderr,
    )
    raise e

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# ---------------------------------------------------------------------------
# 1. Global config & logging
# ---------------------------------------------------------------------------
BASE_URL = (
    "PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE"
)

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


def caption_gemma3(
    csv_in: Path,
    csv_out: Path,
    model,
    processor,
    inference_model,
    max_new,
    temperature,
    top_p,
):
    df = pd.read_csv(csv_in, keep_default_na=False)

    done = set()
    if csv_out.exists():
        with open(csv_out, newline="", encoding="utf-8") as f:
            done = {r["image_id"] for r in csv.DictReader(f) if r.get("caption")}
        df = df[~df["image_id"].isin(done)]

    fields = [
        "image_id",
        "image_path",
        "caption",
        "gt_caption",
    ]

    with open_csv_writer(csv_out, fields) as (writer, fh):
        bar = tqdm(total=len(df), unit="caption", desc="Gemma3 Caption")
        try:
            for _, row in df.iterrows():
                if inference_model == "zero_shot":
                    sys_p, usr_p = evaluate_vlm_caption_zero_shot()
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": sys_p}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "url": BASE_URL + row["image_path"].strip(),
                                },
                                {"type": "text", "text": usr_p},
                            ],
                        },
                    ]
                else:
                    shot_blocks, sys_p, usr_p = evaluate_vlm_caption_one_shot()
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": sys_p}],
                        }
                    ]
                    for sb in shot_blocks:
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "url": BASE_URL + sb["image_path"].strip(),
                                    },
                                    {"type": "text", "text": sb["question"]},
                                ],
                            }
                        )
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": sb["caption"]},
                                ],
                            }
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "url": BASE_URL + row["image_path"].strip(),
                                },
                                {"type": "text", "text": usr_p},
                            ],
                        }
                    )

                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                    )
                generation = generation[0][input_len:]

                vlm_caption = processor.decode(
                    generation,
                    skip_special_tokens=True,
                )

                writer.writerow(
                    {
                        "image_id": row["image_id"],
                        "image_path": row["image_path"],
                        "caption": vlm_caption,
                        "gt_caption": row["caption"],
                    }
                )
                fh.flush()
                bar.update()
        except KeyboardInterrupt:
            print("[INFO] Interrupted—partial results saved.")
        finally:
            bar.close()
    print(f"[✓] Answers written → {csv_out}")


def main():
    torch.backends.cuda.matmul.allow_tf32 = True  # free speed on Ampere/ADA

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument(
        "--mode",
        choices=["zero_shot", "one_shot"],
        default="one_shot",
        help="Inference mode (default: one_shot)",
    )
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.8)
    args = ap.parse_args()

    input_csv = "caption_gt/captioning_ft_full.csv"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    if os.getenv("JIT_COMPILE", "1") == "1":
        model = torch.compile(model)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model)

    output_csv = f"caption_answer/gemma3_caption_{args.mode}_t-{args.temperature}_p-{args.top_p}.csv"

    caption_gemma3(
        Path(input_csv),
        Path(output_csv),
        model,
        processor,
        inference_model={args.mode},
        max_new=args.max_new,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
