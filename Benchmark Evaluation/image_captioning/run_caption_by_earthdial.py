#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, requests, torch
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
VLM_TO_CAPTION_SYSTEM_PROMPT = """You are an expert model for describing satellite or aerial images of landscapes, 
where each image pixel represents a 30-meter ground resolution. Use detailed, domain-specific language 
to describe the visible land covers, features, surface types (e.g., vegetation, artificial surfaces, water, etc.), 
and spatial relationships appropriate for the given spatial scale. Your goal is to give an analytical, objective 
caption that covers both the dominant and minor elements in the image, referencing spatial orientation 
(top-left, center, etc.) and notable connections (such as roads, patch boundaries, etc.). 
Base your descriptions only on observable features in the image, keeping the pixel resolution in mind."""

VLM_TO_CAPTION_USER_PROMPT = """Each image pixel corresponds to 30 meters on the ground. Respond in plain 
text only, with no formatting, lists, or special markup—just a single paragraph. 
Now, describe the following image in the same detailed manner, considering that each pixel represents 
30 meters:"""


def run_vlm_by_caption_zero_shot():
    system_prompt = VLM_TO_CAPTION_SYSTEM_PROMPT
    user_prompt = VLM_TO_CAPTION_USER_PROMPT
    return system_prompt, user_prompt


shot_blocks = [
    {
        "image_path": "DEA_VLM_images/ga_ls9c_ard_3-x60y28-2024-patches/ga_ls9c_ard_3-x60y28-2024-r331nmx-2024-07-29-raw.png",
        "question": VLM_TO_CAPTION_USER_PROMPT,
        "caption": "The image shows a landscape dominated by dark green natural terrestrial vegetation, covering most of the area. Patches of cultivated terrestrial vegetation are visible in the top-left, bottom-left, and center areas, appearing as lighter green or brownish zones. Artificial surfaces, likely small clearings or structures, are present in the top-right and bottom-right areas, with one distinct light patch in the bottom-right. The overall balance is heavily in favor of vegetated surfaces, with artificial and bare areas being minor. The color palette is dominated by dark greens with occasional lighter and brownish patches. The road corridor in the top-right connects directly to the artificial surface and cultivated vegetation patches, forming a clear link between infrastructure and land use zones. A narrow, winding path or track is visible near the bottom-center, cutting through the vegetation and extending slightly into the middle section of the image.",
    }
]


BASE_URL = (
    "PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE"
)


# ────────────────────────────────────────────────────────── util helpers ──
def ensure_image_token(text: str) -> str:
    """Guarantee a '<image>' placeholder at top of the prompt."""
    return text if "<image>" in text else "<image>\n" + text


@contextmanager
def open_csv_writer(path: Path, fields):
    mode = "a" if path.exists() else "w"
    with open(path, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if mode == "w":
            w.writeheader()
        yield w, fh


def fetch_image(url_or_path: str) -> Image.Image:
    buf = (
        requests.get(url_or_path, timeout=20).content
        if url_or_path.startswith("http")
        else Path(url_or_path).read_bytes()
    )
    return Image.open(BytesIO(buf)).convert("RGB")


# ───────────────────────────────────────────────────────── core routine ──
def generate(
    input_csv: str | Path, model_name: str, mode: str, temp: float, top_p: float
):
    out_dir = Path(
        "caption_answer"
    )
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"earthdial_caption_{mode}_t-{temp}_p-{top_p}.csv"

    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False
    )
    model = InternVLChatModel.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()
    device = model.device
    transform = build_transform(
        False, model.config.vision_config.image_size, "imagenet"
    )

    df = pd.read_csv(input_csv, keep_default_na=False)
    if out_csv.exists():
        done = {
            r["image_id"] for r in csv.DictReader(out_csv.open()) if r.get("caption")
        }
        df = df[~df["image_id"].isin(done)]

    fields = ["image_id", "image_path", "caption", "gt_caption"]

    with open_csv_writer(out_csv, fields) as (writer, fh):
        bar = tqdm(total=len(df), desc="EarthDial caption")
        for _, row in df.iterrows():
            tgt_img = fetch_image(BASE_URL + row["image_path"].strip())
            if tgt_img is None:
                bar.update()
                continue

            tgt_tensor = transform(tgt_img).to(device, dtype=model.dtype)
            pixel_values = tgt_tensor.unsqueeze(0)
            num_patches = [1]
            history: List[Tuple[str, str]] | None = None

            # build target question with <image> placeholder
            sys_p, usr_p = run_vlm_by_caption_zero_shot()
            question_prompt = ensure_image_token(f"{sys_p}\n\n{usr_p}")

            # one‑shot exemplar
            if mode == "one_shot":
                ex = shot_blocks[0]
                ex_img = fetch_image(BASE_URL + ex["image_path"])
                ex_tensor = transform(ex_img).to(device, dtype=model.dtype)

                pixel_values = torch.cat([ex_tensor.unsqueeze(0), pixel_values])
                num_patches = [1, 1]

                # exemplar Q must also include <image>
                ex_prompt = ensure_image_token(usr_p)
                history = [(ex_prompt, ex["caption"])]

            gcfg = {
                "num_beams": 5,
                "do_sample": True,  # beam‑search is deterministic; disable sampling
                "max_new_tokens": 1024,
                "temperature": temp,
                "top_p": top_p,
            }

            vlm_caption = model.chat(
                tokenizer=tok,
                pixel_values=pixel_values,
                num_patches_list=num_patches,
                question=question_prompt,
                history=history,
                generation_config=gcfg,
                verbose=False,
            ).replace("\n", " ")

            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "caption": vlm_caption,
                    "gt_caption": row["caption"],
                }
            )
            fh.flush()
            bar.update(1)

    print("✓ Done →", out_csv)


# ────────────────────────────────────────────────────────────── CLI ──────
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=(
            "caption_gt/captioning_ft_full.csv"
        ),
    )
    ap.add_argument("--model", default="akshaydudhane/EarthDial_4B_RGB")
    ap.add_argument("--mode", choices=["zero_shot", "one_shot"], default="one_shot")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.8)
    args = ap.parse_args()
    generate(args.input, args.model, args.mode, args.temperature, args.top_p)


if __name__ == "__main__":
    main()
