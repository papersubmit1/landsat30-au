#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info  # utility from repo


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. Core generation routine
# ---------------------------------------------------------------------------


def generate_caption_mimo(
    input_csv: str | Path,
    output_csv: str | Path,
    model,
    processor,
):
    """Run Caption generation on a single CSV file."""

    df = pd.read_csv(input_csv, keep_default_na=False)

    fieldnames = [
        "image_id",
        "image_path",
        "caption",
        "gt_caption",
    ]

    # resume support ---------------------------------------------------------
    processed_ids: set[str] = set()
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8", newline="") as f:
            processed_ids = {
                r["image_id"] for r in csv.DictReader(f) if r.get("caption")
            }
        df = df[~df["image_id"].isin(processed_ids)]

    if df.empty:
        print("Nothing to process – already complete.")
        return

    # open writer ------------------------------------------------------------
    with open_csv_writer(output_csv, fieldnames) as (writer, outfile):
        pbar = tqdm(total=len(df), desc="Mimo MiMo-VL-7B Caption", unit="caption")
        try:
            for _, row in df.iterrows():
                # ----------------------------------------------------------------
                # 1. Build conversation messages (zero‑ vs one‑shot)
                # ----------------------------------------------------------------
                tgt_img_url = BASE_URL + row["image_path"].strip()
                tgt_img = load_image(tgt_img_url, mode="PIL")
                if tgt_img is None:
                    pbar.update()
                    continue

                sys_p, usr_p = evaluate_vlm_caption_zero_shot()
                messages = [
                    {"role": "system", "content": sys_p},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": tgt_img},
                            {"type": "text", "text": usr_p},
                        ],
                    },
                ]

                # ----------------------------------------------------------------
                # 2. Prepare inputs
                # ----------------------------------------------------------------
                text_prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                img_inputs, vid_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_prompt],
                    images=img_inputs,
                    videos=vid_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                # ----------------------------------------------------------------
                # 3. Generate answer
                # ----------------------------------------------------------------
                with torch.inference_mode():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=8192,
                    )
                gen_trim = gen[:, inputs.input_ids.shape[1] :]
                vlm_caption = (
                    processor.batch_decode(
                        gen_trim,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    .replace("\n", " ")
                    .strip()
                )

                writer.writerow(
                    {
                        "image_id": row["image_id"],
                        "image_path": row["image_path"],
                        "caption": vlm_caption,
                        "gt_caption": row["caption"],
                    }
                )
                outfile.flush()
                pbar.update()
        except KeyboardInterrupt:
            print("Interrupted — partial results saved.")
        finally:
            pbar.close()

    print(f"[✓] Results saved to {output_csv}")


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------


def main():
    torch.backends.cuda.matmul.allow_tf32 = True

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="XiaomiMiMo/MiMo-VL-7B-RL")
    ap.add_argument(
        "--mode",
        choices=["zero_shot"],
        default="zero_shot",
        help="Inference mode",
    )

    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    # ------------------ load model ------------------
    print(f"Loading {args.model} …")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    model = torch.compile(model)

    processor = AutoProcessor.from_pretrained(args.model)

    # ------------------ paths -----------------------
    input_csv = "caption_gt/captioning_ft_full.csv"
    out_dir = Path("caption_answer")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / (f"{args.model.split('/')[-1]}_caption_{args.mode}.csv")

    generate_caption_mimo(
        input_csv,
        out_csv,
        model,
        processor,
    )


if __name__ == "__main__":
    main()
