#!/usr/bin/env python3
"""
qwen2_batch_vqa_one_shot.py
---------------------------
Batch VQA inference with **Qwen‑2.5‑VL** supporting both *zero‑shot* and
*one‑shot* modes (category‑specific exemplars).

• Use `--mode one_shot` (default) to inject a single exemplar Q/A + image
  matching `question_type` for each target sample.
• Falls back to zero‑shot automatically when no exemplar exists.

Example
-------
python qwen2_batch_vqa_one_shot.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --mode one_shot \
    --temperature 0.3 --top-p 0.8
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import math
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info  # utility from repo


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



BASE_URL = (
    "PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE"
)

one_shot_df = pd.read_csv(
    "one_shot/one_shot.csv"
)

EVALUATE_VLM_CAPTION_USER_PROMPT = """Each image pixel corresponds to 30 meters on the ground. Respond in plain 
text only, with no formatting, lists, or special markup—just a single paragraph. 
Now, describe the following image in the same detailed manner, considering that each pixel represents 
30 meters:"""

EVALUATE_VLM_VQA_SYSTEM_PROMPT = """
You are an evaluation agent for remote–sensing VQA.
Your ONLY job is to look at a satellite image, read
the multiple‑choice question and its options, and pick
exactly ONE best answer. The image pixel resolution is
30x30m.

────────────
Task
────────────
1. Inspect the image carefully.
2. Read the question and the list of answer options
3. Choose the single option that best answers the
   question, based solely on visual evidence.

────────────
Output rules
────────────
• Return **only** the text in current option.
• Do **NOT** output words, punctuation, or explanations.
• Trim whitespace;.

────────────
Example
────────────
(User supplies an image that clearly shows a branching
network of channels entering a muddy coastline.)

Question:
Which land‑cover type is dominant in this image?

Options: ['Dense forest', 'Bare surface', 'Urban area', 'River delta'].

Answer: River delta
"""

def evaluate_vlm_vqa_one_shot(one_shot_df, question_type, question_text, option_text):
    """
    Returns prompts and an example for one-shot VLM VQA evaluation.

    Args:
        one_shot_df (pd.DataFrame): DataFrame with one-shot examples.
        question_type (str): The type of question for filtering examples.
        question_text (str): The question to be answered.
        option_text (str): The multiple-choice options for the question.

    Returns:
        tuple: A tuple of (shot_blocks, system_prompt, user_prompt).
    """
    system_prompt = EVALUATE_VLM_VQA_SYSTEM_PROMPT
    one_shot_df = one_shot_df[one_shot_df["question_type"] == question_type]

    shot_blocks = []
    for _, row in one_shot_df.iterrows():
        shot_blocks.append(
            {
                "image_path": row["image_path"],
                "full_question": f"Question:{row['question']}\n\nOptions: {row['options']}\n",
                "answer": row["answer"],
            }
        )

    user_prompt = f"Question:{question_text}\n\nOptions: {option_text}\n"
    return shot_blocks, system_prompt, user_prompt


def evaluate_vlm_vqa_zero_shot(question_txt, option_txt):
    """
    Returns prompts for zero-shot VLM VQA evaluation.

    Args:
        question_txt (str): The question to be answered.
        option_txt (str): The multiple-choice options for the question.

    Returns:
        tuple: A tuple containing the system prompt and the user prompt.
    """
    system_prompt = EVALUATE_VLM_VQA_SYSTEM_PROMPT
    user_prompt = f"Question:{question_txt}\n\nOptions: {option_txt}\n"
    return system_prompt, user_prompt


ONE_SHOT_CSV = Path(
    "one_shot/one_shot.csv"
)


# lazy‑load exemplars once
def _load_one_shot_df() -> pd.DataFrame:
    if not ONE_SHOT_CSV.exists():
        raise FileNotFoundError(
            "one_shot.csv not found → switch to zero‑shot or supply the file."
        )
    return pd.read_csv(ONE_SHOT_CSV)


_one_shot_df: pd.DataFrame | None = None


def get_one_shot_df() -> pd.DataFrame:
    global _one_shot_df
    if _one_shot_df is None:
        _one_shot_df = _load_one_shot_df()
    return _one_shot_df


# ---------------------------------------------------------------------------
# 2. Core generation routine
# ---------------------------------------------------------------------------


def generate_vqa_qwen(
    input_csv: str | Path,
    output_csv: str | Path,
    model,
    processor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    mode: str,
):
    """Run VQA generation on a single CSV file."""

    df = pd.read_csv(input_csv, keep_default_na=False)

    fieldnames = [
        "qa_id",
        "image_path",
        "question",
        "answer",
        "options",
        "gt_answer",
    ]

    # resume support ---------------------------------------------------------
    processed_ids: set[str] = set()
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8", newline="") as f:
            processed_ids = {r["qa_id"] for r in csv.DictReader(f) if r.get("answer")}
        df = df[~df["qa_id"].isin(processed_ids)]

    if df.empty:
        print("Nothing to process – already complete.")
        return

    # open writer ------------------------------------------------------------
    with open_csv_writer(output_csv, fieldnames) as (writer, outfile):
        pbar = tqdm(total=len(df), desc="Qwen‑VL VQA", unit="qa")
        try:
            one_shot_df = get_one_shot_df() if mode == "one_shot" else None

            for _, row in df.iterrows():
                # ----------------------------------------------------------------
                # 1. Build conversation messages (zero‑ vs one‑shot)
                # ----------------------------------------------------------------
                tgt_img_url = BASE_URL + row["image_path"].strip()
                tgt_img = load_image(tgt_img_url, mode="PIL")
                if tgt_img is None:
                    pbar.update()
                    continue

                if mode == "one_shot":
                    q_type = row.get("question_type", None)
                    shot_blocks, sys_p, usr_p = evaluate_vlm_vqa_one_shot(
                        one_shot_df, q_type, row["question"], row["options"]
                    )
                    if not shot_blocks:  # fallback
                        sys_p, usr_p = evaluate_vlm_vqa_zero_shot(row["question"], row["options"])
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
                    else:
                        messages: List[dict] = [
                            {"role": "system", "content": sys_p},
                        ]
                        # add exemplar conversation
                        for sb in shot_blocks:
                            shot_img_url = BASE_URL + sb["image_path"].strip()
                            shot_img = load_image(shot_img_url, mode="PIL")
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": shot_img},
                                        {"type": "text", "text": sb["full_question"]},
                                    ],
                                }
                            )
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": sb["answer"]}],
                                }
                            )
                        # final target turn
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": tgt_img},
                                    {"type": "text", "text": usr_p},
                                ],
                            }
                        )
                else:  # zero‑shot
                    sys_p, usr_p = evaluate_vlm_vqa_zero_shot(row["question"], row["options"])
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
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                gen_trim = gen[:, inputs.input_ids.shape[1] :]
                answer = (
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
                        "qa_id": row["qa_id"],
                        "image_path": row["image_path"],
                        "question": row["question"],
                        "answer": answer,
                        "options": row["options"],
                        "gt_answer": row["answer"],
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
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument(
        "--mode",
        choices=["zero_shot", "one_shot"],
        default="zero_shot",
        help="Inference mode",
    )
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=120)
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

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # ------------------ paths -----------------------
    input_csv = "Landsat30-AU-VQA-test.csv"
    out_dir = Path("vqa_answer")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / (
        f"{args.model.split('/')[-1]}-ft_vqa_{args.mode}_t-{args.temperature}_p-{args.top_p}.csv"
    )

    generate_vqa_qwen(
        input_csv,
        out_csv,
        model,
        processor,
        args.max_tokens,
        args.temperature,
        args.top_p,
        args.mode,
    )


if __name__ == "__main__":
    main()
