#!/usr/bin/env python3
"""
rsllava_batch_vqa_one_shot.py (patched)
--------------------------------------
Fixes for multi‑image alignment:
* **No duplicate `<image>` markers** – if a prompt already contains the token we
  replace it with the model wrapper instead of prepending a second one.
* Explicit **`attention_mask`** passed to `model.generate` to suppress the
  warning about identical `pad`/`eos` tokens.
* The image tensor stack length now exactly matches the number of wrappers,
  preventing the `IndexError: index 1 is out of bounds` crash.
"""

from __future__ import annotations

import argparse, csv, os, base64, requests, torch
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Set

import pandas as pd
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────────────────────────────────────


def run_vlm_by_vqa(question: str, options: str) -> Tuple[str, str]:
    sys = (
        "You are an evaluation agent for remote–sensing VQA.\n"
        "Your ONLY job is to look at a satellite image, read the multiple‑choice question "
        "and its options, and pick exactly ONE best answer.\n\n"
        "Output rules\n────────────\n• Return **only** the text in current option.\n"
        "• Do **NOT** output words, punctuation, or explanations.\n• Trim whitespace.\n"
    )
    usr = f"<image>\nQuestion:\n{question}\n\nOptions: {options}\n"
    return sys, usr


# One‑shot CSV
ONE_SHOT_CSV = Path(
    "one_shot_gt/one_shot.csvv"
)

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
BASE_URL = (
    "https://dea-public-data-dev.s3.ap-southeast-2.amazonaws.com/projects/dea-vqa/"
)


@contextmanager
def open_csv_writer(path: Path, fieldnames):
    mode = "a" if path.exists() else "w"
    with open(path, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        yield writer, fh


def fetch_image(url_or_path: str) -> Image.Image | None:
    try:
        buf = (
            requests.get(url_or_path, timeout=15).content
            if url_or_path.startswith("http")
            else Path(url_or_path).read_bytes()
        )
        return Image.open(BytesIO(buf)).convert("RGB")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Chat helper (patched)
# ─────────────────────────────────────────────────────────────────────────────


def llava_chat(
    model,
    tokenizer,
    image_processor,
    conv_mode: str,
    system_prompt: str,
    turns: List[Tuple[str, Image.Image | None]],
    temperature: float,
    top_p: float,
) -> str:
    """Build a conversation with correct image/token alignment and query RS‑LLaVA."""
    conv = conv_templates[conv_mode].copy()
    conv.system = system_prompt
    all_images: List[Image.Image] = []

    role_cycle = [conv.roles[0], conv.roles[1]] * ((len(turns) + 1) // 2)
    for role, (txt, img) in zip(role_cycle, turns):
        if img is not None:
            wrapper = (
                f"{DEFAULT_IM_START_TOKEN} {DEFAULT_IMAGE_TOKEN} {DEFAULT_IM_END_TOKEN}"
                if model.config.mm_use_im_start_end
                else DEFAULT_IMAGE_TOKEN
            )
            if "<image>" in txt:
                txt = txt.replace("<image>", wrapper, 1)
            else:
                txt = f"{wrapper}\n" + txt
            all_images.append(img)
        conv.append_message(role, txt)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    # use all‑ones mask to avoid dtype issues
    attention_mask = torch.ones_like(input_ids)  # shape == input_ids

    if not all_images:
        raise ValueError("No images matched to <image> tokens.")
    image_batch = (
        torch.stack(
            [
                image_processor(im, return_tensors="pt")["pixel_values"][0]
                for im in all_images
            ]
        )
        .half()
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            images=image_batch,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_beams=1,
            max_new_tokens=512,
            stopping_criteria=[stopping],
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return tokenizer.decode(
        out[0, input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main batch generator
# ─────────────────────────────────────────────────────────────────────────────


def generate_rsllava(
    input_csv: str | Path,
    model_path: str,
    model_base: str,
    mode: str,
    temperature: float,
    top_p: float,
):
    conv_mode = "llava_v1"
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        model_base,
        get_model_name_from_path(model_path),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    out_dir = Path("vqa_answer")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"rsllava_vqa_{mode}_t-{temperature}_p-{top_p}.csv"
    df = pd.read_csv(input_csv, keep_default_na=False)
    if out_csv.exists():
        done = {r["qa_id"] for r in csv.DictReader(out_csv.open()) if r.get("answer")}
        df = df[~df["qa_id"].isin(done)]

    exemplar_df = pd.read_csv(ONE_SHOT_CSV) if mode == "one_shot" else None
    fields = ["qa_id", "image_path", "question", "answer", "options", "gt_answer"]

    with open_csv_writer(out_csv, fields) as (writer, fh):
        bar = tqdm(total=len(df), desc="RS‑LLaVA VQA")
        for _, row in df.iterrows():
            tgt_img = fetch_image(BASE_URL + row["image_path"].strip())
            sys_p, tgt_q = run_vlm_by_vqa(row["question"], row["options"])
            if tgt_img is None:
                bar.update()
                continue
            turns: List[Tuple[str, Image.Image | None]] = []
            if mode == "one_shot" and exemplar_df is not None:
                sub = exemplar_df[
                    exemplar_df["question_type"] == row.get("question_type")
                ]
                if not sub.empty:
                    ex = sub.sample(1, random_state=42).iloc[0]
                    ex_img = fetch_image(BASE_URL + ex["image_path"].strip())
                    if ex_img is not None:
                        _, ex_q = run_vlm_by_vqa(ex["question"], ex["options"])
                        turns.extend([(ex_q, ex_img), (ex["answer"], None)])
            turns.append((tgt_q, tgt_img))
            answer = llava_chat(
                model,
                tokenizer,
                image_processor,
                conv_mode,
                sys_p,
                turns,
                temperature,
                top_p,
            )
            writer.writerow(
                {
                    "qa_id": row["qa_id"],
                    "image_path": row["image_path"],
                    "question": row["question"],
                    "answer": answer.replace("\n", " "),
                    "options": row["options"],
                    "gt_answer": row["answer"],
                }
            )
            fh.flush()
            bar.update(1)
    print(f"✓ Results saved to {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="Landsat30-AU-VQA-test.csv",
    )
    ap.add_argument("--mode", choices=["zero_shot", "one_shot"], default="zero_shot")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.8)
    args = ap.parse_args()

    generate_rsllava(
        args.input,
        "BigData-KSU/RS-llava-v1.5-7b-LoRA",
        "Intel/neural-chat-7b-v3-3",
        args.mode,
        args.temperature,
        args.top_p,
    )


if __name__ == "__main__":
    main()
