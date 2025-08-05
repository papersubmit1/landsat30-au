#!/usr/bin/env python3
"""
llava_onevision_batch_vqa_one_shot.py
-------------------------------------
Batch VQA inference with **LLaVA‑OneVision** supporting both *zero‑shot* and
*one‑shot* exemplar prompting.

Usage
-----
python llava_onevision_batch_vqa_one_shot.py \
    --model llava-hf/llava-onevision-qwen2-7b-si-hf \
    --mode one_shot \
    --temperature 0.3 --top-p 0.8
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

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


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
BASE_URL = (
    "PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE"
)
ONE_SHOT_CSV = Path(
    "one_shot.csv"
)

# lazy‑load exemplars once
_one_shot_df: pd.DataFrame | None = None


def get_one_shot_df() -> pd.DataFrame:
    global _one_shot_df
    if _one_shot_df is None:
        if not ONE_SHOT_CSV.exists():
            raise FileNotFoundError(
                "one_shot.csv not found → supply file or use zero‑shot mode."
            )
        _one_shot_df = pd.read_csv(ONE_SHOT_CSV)
    return _one_shot_df


# ---------------------------------------------------------------------------
# 2. Core routine
# ---------------------------------------------------------------------------


def vqa_llava_onevision(
    csv_in: Path,
    csv_out: Path,
    model,
    processor,
    mode: str = "one_shot",
    max_new: int = 120,
    temperature: float = 0.3,
    top_p: float = 0.8,
):
    """Iterate a VQA CSV and write answers generated by LLaVA‑OneVision."""
    df = pd.read_csv(csv_in, keep_default_na=False)

    # resume‑able run ---------------------------------------------------------
    done = set()
    if csv_out.exists():
        with open(csv_out, newline="", encoding="utf-8") as f:
            done = {r["qa_id"] for r in csv.DictReader(f) if r.get("answer")}
        df = df[~df["qa_id"].isin(done)]

    if df.empty:
        print("Nothing to process — already complete!")
        return

    fields = [
        "qa_id",
        "image_path",
        "question",
        "answer",
        "options",
        "gt_answer",
    ]

    with open_csv_writer(csv_out, fields) as (writer, fh):
        bar = tqdm(total=len(df), unit="sample", desc="LLaVA‑OneVision VQA")
        try:
            shot_df = get_one_shot_df() if mode == "one_shot" else None

            for _, row in df.iterrows():
                # ----------------------------------------------------------------
                # 1. Build conversation messages (zero‑ vs one‑shot)
                # ----------------------------------------------------------------
                tgt_img_url = BASE_URL + row["image_path"].strip()
                tgt_img = load_image(tgt_img_url, mode="PIL")
                if tgt_img is None:
                    bar.update()
                    continue

                if mode == "one_shot":
                    q_type = row.get("question_type", None)
                    shot_blocks, sys_p, usr_p = (
                        shot_df, q_type, row["question"], row["options"]
                    )
                    if not shot_blocks:  # fallback to zero‑shot
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
                        # Add exemplar blocks
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
                        # final user turn
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": tgt_img},
                                    {"type": "text", "text": usr_p},
                                ],
                            }
                        )
                else:  # zero‑shot path
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
                # 2. Convert to prompt & encode
                # ----------------------------------------------------------------
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # The processor expects **all** images in a list, in order of appearance.
                # We can extract them directly from the messages list.
                imgs_ordered: List = [
                    c["image"]
                    for m in messages[1:] # skip the system prompt
                    for c in m.get("content", [])
                    if c["type"] == "image"
                ]

                inputs = processor(
                    images=imgs_ordered,
                    text=[prompt],
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                # ----------------------------------------------------------------
                # 3. Generate answer
                # ----------------------------------------------------------------
                with torch.inference_mode():
                    out_ids = model.generate(
                        **inputs,
                        do_sample=True,
                        max_new_tokens=max_new,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )

                ans_ids = out_ids[:, inputs.input_ids.shape[1] :]
                answer = (
                    processor.batch_decode(
                        ans_ids,
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
                fh.flush()
                bar.update()
        except KeyboardInterrupt:
            print("[INFO] Interrupted — partial results saved.")
        finally:
            bar.close()
    print(f"[✓] Answers written → {csv_out}")


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------


def main():
    torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+ speed boost

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-onevision-qwen2-7b-si-hf")
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

    # ------------------ load model ----------------------------
    print(f"Loading {args.model} …")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    model = torch.compile(model)

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"

    # ------------------ paths & run ---------------------------
    input_csv = "Landsat30-AU-VQA-test.csv"
    out_dir = Path("vqa_answer")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / (
        f"{args.model.split('/')[-1]}_vqa_{args.mode}_t-{args.temperature}_p-{args.top_p}.csv"
    )

    vqa_llava_onevision(
        Path(input_csv),
        out_csv,
        model,
        processor,
        mode=args.mode,
        max_new=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
