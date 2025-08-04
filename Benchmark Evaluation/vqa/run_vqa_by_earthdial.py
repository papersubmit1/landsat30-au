#!/usr/bin/env python3

from __future__ import annotations

# --- lock process to GPU 3 --------------------------------------------------
import os


import argparse, csv, base64, requests, torch
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from earthdial.model.internvl_chat import InternVLChatModel
from earthdial.train.dataset import build_transform

# ────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ────────────────────────────────────────────────────────────────────────────


def run_vlm_by_vqa(question: str, options: str):
    sys = (
        "You are an evaluation agent for remote–sensing VQA.\n"
        "Your ONLY job is to look at a satellite image, read the multiple‑choice question and its options, "
        "and pick exactly ONE best answer. The image pixel resolution is 30 m.\n\n"
        "Output rules\n────────────\nReturn **only** the text in current option.\n• Do **NOT** output words, punctuation, or explanations.\n• Trim whitespace.\n"
    )
    usr = f"<image>\nQuestion:\n{question}\n\nOptions: {options}\n"
    return sys, usr


ONE_SHOT_CSV = Path(
    "one_shot_gt/one_shot.csv"
)

BASE_URL = (
    "PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE"
)

# ────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────
@contextmanager
def open_csv_writer(path: Path, fields):
    mode = "a" if path.exists() else "w"
    with open(path, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader() if mode == "w" else None
        yield w, fh


def fetch_image(url: str):
    buf = (
        requests.get(url, timeout=20).content
        if url.startswith("http")
        else Path(url).read_bytes()
    )
    return Image.open(BytesIO(buf)).convert("RGB")


# ────────────────────────────────────────────────────────────────────────────
# Core generator
# ────────────────────────────────────────────────────────────────────────────


def generate(
    input_csv: str | Path, model_name: str, mode: str, temp: float, top_p: float
):
    out_dir = Path("vqa_answer")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"earthdial_vqa_{mode}_t-{temp}_p-{top_p}.csv"

    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False
    )
    model = InternVLChatModel.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()
    device = model.device  # cuda:0 (which is physical GPU 3)
    transform = build_transform(
        False, model.config.vision_config.image_size, "imagenet"
    )

    df = pd.read_csv(input_csv, keep_default_na=False)
    if out_csv.exists():
        done = {r["qa_id"] for r in csv.DictReader(out_csv.open()) if r.get("answer")}
        df = df[~df["qa_id"].isin(done)]

    exemplar_df = pd.read_csv(ONE_SHOT_CSV) if mode == "one_shot" else None
    fields = ["qa_id", "image_path", "question", "answer", "options", "gt_answer"]

    with open_csv_writer(out_csv, fields) as (writer, fh):
        bar = tqdm(total=len(df), desc="EarthDial VQA")
        for _, row in df.iterrows():
            tgt_img = fetch_image(BASE_URL + row["image_path"].strip())
            if tgt_img is None:
                bar.update()
                continue

            tgt_tensor = transform(tgt_img).to(device, dtype=model.dtype)
            pixel_values = tgt_tensor.unsqueeze(0)
            num_patches = [1]
            history: List[Tuple[str, str]] | None = None

            if mode == "one_shot" and exemplar_df is not None:
                sub = exemplar_df[
                    exemplar_df["question_type"] == row.get("question_type")
                ]
                if not sub.empty:
                    ex = sub.sample(1).iloc[0]
                    ex_img = fetch_image(BASE_URL + ex["image_path"].strip())
                    if ex_img is not None:
                        ex_tensor = transform(ex_img).to(device, dtype=model.dtype)
                        pixel_values = torch.cat([ex_tensor.unsqueeze(0), pixel_values])
                        num_patches = [1, 1]
                        _, ex_q = run_vlm_by_vqa(ex["question"], ex["options"])
                        history = [(ex_q, ex["answer"])]

            _, tgt_prompt = run_vlm_by_vqa(row["question"], row["options"])
            gcfg = {
                "num_beams": 5,
                "max_new_tokens": 120,
                "do_sample": temp > 0,
                "temperature": temp,
                "top_p": top_p,
            }
            answer = model.chat(
                tokenizer=tok,
                pixel_values=pixel_values,
                num_patches_list=num_patches,
                question=tgt_prompt,
                history=history,
                generation_config=gcfg,
                verbose=False,
            )
            answer = answer.replace("\n", " ")
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
            bar.update(1)
    print("✓ Done →", out_csv)


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="Landsat30-AU-VQA-test.csv",
    )
    ap.add_argument("--model", default="akshaydudhane/EarthDial_4B_RGB")
    ap.add_argument("--mode", choices=["zero_shot", "one_shot"], default="one_shot")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.8)
    args = ap.parse_args()
    generate(args.input, args.model, args.mode, args.temperature, args.top_p)


if __name__ == "__main__":
    main()
