#!/usr/bin/env python3
"""
gemma_batch_caption_review_v5_oneshot.py
----------------------------------------
Batch caption‑review / VQA pipeline for **Gemma‑3 VLM** with optional **one‑shot**
examples per question‑type.

Key additions vs v4
-------------------
✦ `--mode {zero_shot,one_shot}` CLI flag (default *one_shot*)
✦ Loads a small CSV of exemplar shots (`one_shot.csv`) once at start‑up
✦ New helper `make_payloads()` builds either zero‑shot or one‑shot messages;
  shot images are fetched on‑the‑fly with the existing `fetch_and_decode()`
✦ End‑to‑end logic in `review_batch()` & `process_file()` updated to pass the
  mode flag through – **no other code-paths were touched**

Usage (unchanged except for the new flag)
----------------------------------------
python gemma_batch_caption_review_v5_oneshot.py \
    --model google/gemma-3-12b-it \
    --mode one_shot \
    --batch 8 --temperature 0.3 --top-p 0.8
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import List, Tuple
from urllib.parse import quote
import pandas as pd

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
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TOP_P = 0.8
DEFAULT_TEMPERATURE = 0.3
CSV_FLUSH_INTERVAL = 10  # flush every N successful batches

MAX_WORKERS = 16  # threads for image prefetch; tune per machine
_image_pool: ThreadPoolExecutor | None = None
_session: requests.Session | None = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

ONE_SHOT_CSV = Path(
    "one_shot/one_shot.csv"
)
_one_shot_df: pd.DataFrame | None = None

# ---------------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------------


def get_pool() -> ThreadPoolExecutor:
    global _image_pool
    if _image_pool is None:
        _image_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    return _image_pool


def get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def dtype_name(model):
    return {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(
        model.dtype, str(model.dtype)
    )


def chunk_df(df: pd.DataFrame, size: int):
    for start in range(0, len(df), size):
        yield df.iloc[start : start + size].copy()


def fetch_and_decode(rel_path: str, timeout: int = 15):
    """Download + decode an image. Raises if anything is wrong."""
    url = BASE_URL + quote(rel_path.strip(), safe="/")
    r = get_session().get(url, timeout=timeout)
    r.raise_for_status()
    try:
        img = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as exc:  # pillow catches corrupt image
        raise RuntimeError(f"{url} is not a valid image: {exc}") from exc
    return img


# Cache system‑prompt tokens (they never change)
SYS_PROMPT_TEXT, _ = evaluate_vlm_vqa_zero_shot("dummy", "dummy")  # first element is system prompt
SYS_PROMPT_TOKENS: torch.Tensor | None = None  # filled after processor is available

# ---------------------------------------------------------------------------
# 3. Inference helpers
# ---------------------------------------------------------------------------


def load_one_shot_df() -> pd.DataFrame:
    global _one_shot_df
    if _one_shot_df is None:
        if not ONE_SHOT_CSV.exists():
            raise FileNotFoundError(f"One‑shot CSV not found at {ONE_SHOT_CSV}")
        _one_shot_df = pd.read_csv(ONE_SHOT_CSV)
    return _one_shot_df


def make_payloads(batch_rows, images, mode: str):
    """Build chat template payloads for *either* zero‑shot or one‑shot."""
    payloads: List[list] = []

    one_shot_df = load_one_shot_df() if mode == "one_shot" else None

    for row, image in zip(batch_rows, images):
        if mode == "one_shot":
            # Gather exemplar(s) for the current question type
            q_type = getattr(row, "question_type", None)
            shot_blocks, _sys, usr_prompt = evaluate_vlm_vqa_one_shot(
                one_shot_df, q_type, row.question, row.options
            )
            if not shot_blocks:  # fallback to zero‑shot if nothing found
                _sys, usr_prompt = evaluate_vlm_vqa_zero_shot(row.question, row.options)
                payloads.append(
                    [
                        {"role": "system", "content": [{"type": "text", "text": _sys}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": usr_prompt},
                            ],
                        },
                    ]
                )
                continue

            # Build one‑shot conversation: system → shot(s) → target
            conv: list = [
                {"role": "system", "content": [{"type": "text", "text": _sys}]}
            ]

            for sb in shot_blocks:
                shot_img = fetch_and_decode(sb["image_path"])
                conv.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": shot_img},
                            {"type": "text", "text": sb["full_question"]},
                        ],
                    }
                )
                conv.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": sb["answer"]}],
                    }
                )

            # Append target question & image
            conv.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": usr_prompt},
                    ],
                }
            )
            payloads.append(conv)
        else:
            # zero‑shot
            _sys, usr_prompt = evaluate_vlm_vqa_zero_shot(row.question, row.options)
            payloads.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": _sys}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": usr_prompt},
                        ],
                    },
                ]
            )
    return payloads


def review_batch(
    model,
    processor,
    batch_df: pd.DataFrame,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tokenizer_max_len: int,
    mode: str,
) -> Tuple[List[pd.Series], List[str]]:
    """Prefetch images → build payloads → generate → return (rows, answers)."""
    if batch_df.empty:
        return [], []

    # --- 1. Bucket rows by approximate prompt length to reduce padding --------
    prompt_len = batch_df["question"].str.len() + batch_df["options"].str.len()
    batch_df = batch_df.iloc[prompt_len.argsort()].reset_index(drop=True)

    # --- 2. Kick off parallel downloads --------------------------------------
    futures = [get_pool().submit(fetch_and_decode, p) for p in batch_df.image_path]

    # --- 3. Collect results while building payload list -----------------------
    good_rows, good_images = [], []
    for row, fut in zip(batch_df.itertuples(index=False), futures):
        try:
            img = fut.result()
        except Exception as exc:
            logging.warning("Skip %s — %s", row.qa_id, exc)
            continue
        good_rows.append(row)
        good_images.append(img)

    if not good_rows:
        return [], []  # everything failed

    payloads = make_payloads(good_rows, good_images, mode)

    # --- 4. Tokenise ----------------------------------------------------------
    inputs = processor.apply_chat_template(
        payloads,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer_max_len,
    )

    # Non‑blocking copy to GPU (pin_memory implied by HF tokenizer)
    inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}

    # --- 5. Generate ----------------------------------------------------------
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=model.dtype):
        gens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    new_tokens = gens[:, inputs["input_ids"].shape[-1] :]
    answers = processor.batch_decode(new_tokens, skip_special_tokens=True)
    return good_rows, answers


# ---------------------------------------------------------------------------
# 4. Driver
# ---------------------------------------------------------------------------


def process_file(
    df: pd.DataFrame,
    out_path: Path,
    model,
    processor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    tokenizer_max_len: int,
    mode: str,
):
    fieldnames = ["qa_id", "image_path", "question", "options", "answer", "gt_answer"]

    # Resume support ----------------------------------------------------------
    processed = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8", newline="") as f:
            processed = {r["qa_id"] for r in csv.DictReader(f) if r.get("answer")}
        df = df[~df["qa_id"].isin(processed)]
        logging.info("Resuming — %d rows already done.", len(processed))

    if df.empty:
        logging.info("Nothing to do — file complete.")
        return

    batch_counter = 0
    with open_csv_writer(out_path, fieldnames) as (writer, csvfile):
        pbar = tqdm(total=len(df), desc="Reviewing", unit="qa")
        for chunk in chunk_df(df, batch_size):
            try:
                # OOM‑aware retry ------------------------------------------------
                for attempt in (1, 2):
                    try:
                        rows, answers = review_batch(
                            model,
                            processor,
                            chunk,
                            max_new_tokens,
                            temperature,
                            top_p,
                            tokenizer_max_len,
                            mode,
                        )
                        break
                    except torch.cuda.OutOfMemoryError:
                        if len(chunk) == 1 or attempt == 2:
                            raise
                        torch.cuda.empty_cache()
                        chunk = chunk.iloc[: len(chunk) // 2]
                        logging.warning("OOM — retrying with batch size %d", len(chunk))

                # Write surviving rows ---------------------------------------
                for row, ans in zip(rows, answers):
                    writer.writerow(
                        {
                            "qa_id": row.qa_id,
                            "image_path": row.image_path,
                            "question": row.question,
                            "options": row.options,
                            "answer": ans,
                            "gt_answer": row.answer,
                        }
                    )
                batch_counter += 1
                if batch_counter % CSV_FLUSH_INTERVAL == 0:
                    csvfile.flush()
                pbar.update(len(rows))

            except Exception as ex:  # pylint: disable=broad-except
                logging.error(
                    "Error on batch starting %s: %s", chunk.iloc[0]["qa_id"], ex
                )
                pbar.update(len(chunk))
        csvfile.flush()
        pbar.close()
    logging.info("Saved to %s", out_path)


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------


def main():
    torch.backends.cuda.matmul.allow_tf32 = True  # free speed on Ampere/ADA

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument(
        "--mode",
        choices=["zero_shot", "one_shot"],
        default="zero_shot",
        help="Inference mode (default: zero_shot)",
    )
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    args = ap.parse_args()

    input_csv = "Landsat30-AU-VQA-test.csv"

    # ------------------  Load model -------------------------------
    logging.info("Loading model %s …", args.model)
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

    ctx = getattr(model.config, "max_position_embeddings", 8192)
    logging.info("dtype %s | ctx %d | mode %s", dtype_name(model), ctx, args.mode)

    # Instantiate cached system prompt tokens (optional future use)
    global SYS_PROMPT_TOKENS  # noqa: PLW0603
    SYS_PROMPT_TOKENS = processor.tokenizer(
        SYS_PROMPT_TEXT, return_tensors="pt"
    ).input_ids[0]

    # ------------------  Output directory ------------------------
    out_dir = Path("vqa_answer")
    out_dir.mkdir(exist_ok=True)

    # ------------------  Main loop over files --------------------
    for cap_file in [input_csv]:
        logging.info("Processing %s …", cap_file)
        df = pd.read_csv(cap_file)

        human_readable_name = f"vqa_{args.mode}"
        out_csv = out_dir / (
            f"{args.model.split('/')[-1]}_{human_readable_name}"
            f"_t-{args.temperature}_p-{args.top_p}.csv"
        )

        process_file(
            df,
            out_csv,
            model,
            processor,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.batch,
            ctx,
            args.mode,
        )

    logging.info("All files processed.")


if __name__ == "__main__":
    main()
