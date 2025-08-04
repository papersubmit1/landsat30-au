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


def run_vlm_by_caption_one_shot():
    system_prompt = VLM_TO_CAPTION_SYSTEM_PROMPT

    user_prompt = VLM_TO_CAPTION_USER_PROMPT

    shot_blocks = [
        {
            "image_path": "DEA_VLM_images/ga_ls9c_ard_3-x60y28-2024-patches/ga_ls9c_ard_3-x60y28-2024-r331nmx-2024-07-29-raw.png",
            "question": VLM_TO_CAPTION_USER_PROMPT,
            "caption": "The image shows a landscape dominated by dark green natural terrestrial vegetation, covering most of the area. Patches of cultivated terrestrial vegetation are visible in the top-left, bottom-left, and center areas, appearing as lighter green or brownish zones. Artificial surfaces, likely small clearings or structures, are present in the top-right and bottom-right areas, with one distinct light patch in the bottom-right. The overall balance is heavily in favor of vegetated surfaces, with artificial and bare areas being minor. The color palette is dominated by dark greens with occasional lighter and brownish patches. The road corridor in the top-right connects directly to the artificial surface and cultivated vegetation patches, forming a clear link between infrastructure and land use zones. A narrow, winding path or track is visible near the bottom-center, cutting through the vegetation and extending slightly into the middle section of the image.",
        }
    ]
    return shot_blocks, system_prompt, user_prompt

BASE_URL = (
    "PLEASE PUT YOUR IMAGE FOLER FULL PATH HERE"
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
            max_new_tokens=1024,
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

    out_dir = Path(
        "caption_answer"
    )
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"rsllava_caption_{mode}_t-{temperature}_p-{top_p}.csv"
    df = pd.read_csv(input_csv, keep_default_na=False)
    if out_csv.exists():
        done = {
            r["image_id"] for r in csv.DictReader(out_csv.open()) if r.get("caption")
        }
        df = df[~df["image_id"].isin(done)]

    fields = [
        "image_id",
        "image_path",
        "caption",
        "gt_caption",
    ]

    with open_csv_writer(out_csv, fields) as (writer, fh):
        bar = tqdm(total=len(df), desc="RS‑LLaVA Caption")
        for _, row in df.iterrows():
            tgt_img = fetch_image(BASE_URL + row["image_path"].strip())
            sys_p, tgt_q = run_vlm_by_caption_zero_shot()
            if tgt_img is None:
                bar.update()
                continue
            turns: List[Tuple[str, Image.Image | None]] = []
            if mode == "one_shot":
                sub, _, _ = run_vlm_by_caption_one_shot()

                ex = sub[0]
                ex_img = fetch_image(BASE_URL + ex["image_path"].strip())
                if ex_img is not None:
                    _, ex_q = run_vlm_by_caption_zero_shot()
                    turns.extend([(ex_q, ex_img), (ex["caption"], None)])

            turns.append((tgt_q, tgt_img))
            vlm_caption = llava_chat(
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
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "caption": vlm_caption,
                    "gt_caption": row["caption"],
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
        default=(
            "caption_gt/captioning_ft_full.csv"
        ),
    )
    ap.add_argument("--mode", choices=["zero_shot", "one_shot"], default="one_shot")
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
