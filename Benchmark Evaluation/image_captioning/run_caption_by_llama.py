#!/usr/bin/env python3
"""
llama32_batch_vqa.py
────────────────────
Batch VQA inference with meta‑llama/Llama‑3.2‑11B‑Vision.

Example
-------
python llama32_batch_vqa.py \
    --csv  Landsat30-AU-VQA-test.csv \
    --out  results/llama32_answers.csv
"""

import argparse, csv
from pathlib import Path

import torch, pandas as pd
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoProcessor


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
            "question": VLM_TO_CAPTION_USER_PROMPT,
            "caption": "The image shows a landscape dominated by dark green natural terrestrial vegetation, covering most of the area. Patches of cultivated terrestrial vegetation are visible in the top-left, bottom-left, and center areas, appearing as lighter green or brownish zones. Artificial surfaces, likely small clearings or structures, are present in the top-right and bottom-right areas, with one distinct light patch in the bottom-right. The overall balance is heavily in favor of vegetated surfaces, with artificial and bare areas being minor. The color palette is dominated by dark greens with occasional lighter and brownish patches. The road corridor in the top-right connects directly to the artificial surface and cultivated vegetation patches, forming a clear link between infrastructure and land use zones. A narrow, winding path or track is visible near the bottom-center, cutting through the vegetation and extending slightly into the middle section of the image.",
        }
    ]
    return shot_blocks, system_prompt, user_prompt

def caption_llama32(
    csv_in: Path,
    csv_out: Path,
    model,
    processor,
    inference_model,
    max_new,
    temperature,
    top_p,
):
    """Iterate the Caption CSV and dump answers produced by Llama 3.2‑Vision."""
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
        bar = tqdm(total=len(df), unit="caption", desc="Llama 3.2 Caption")
        try:
            for _, row in df.iterrows():
                images_to_encode = []

                img_url = BASE_URL + row["image_path"].strip()
                tgt_img = load_image(img_url, mode="PIL")
                if tgt_img is None:
                    bar.update()
                    continue

                if inference_model == "zero_shot":
                    sys_p, usr_p = evaluate_vlm_caption_zero_shot()

                    images_to_encode.append(tgt_img)

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
                    shot_blocks, sys_p, usr_p = evaluate_vlm_caption_one_shot()

                    messages = [
                        {"role": "system", "content": sys_p},
                    ]

                    for shot_block in shot_blocks:
                        img_url = BASE_URL + shot_block["image_path"].strip()
                        shot_img = load_image(img_url, mode="PIL")

                        images_to_encode.append(shot_img)

                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": shot_img},
                                    {
                                        "type": "text",
                                        "text": shot_block["question"],
                                    },
                                ],
                            }
                        )
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": shot_block["caption"]}
                                ],
                            },
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": tgt_img},
                                {"type": "text", "text": usr_p},
                            ],
                        }
                    )

                    images_to_encode.append(tgt_img)

                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = processor(
                    images=images_to_encode,
                    text=[prompt],
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        temperature=temperature,
                        top_p=top_p,
                    )

                ans_ids = out[:, inputs.input_ids.shape[1] :]
                vlm_caption = (
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


if __name__ == "__main__":
    temperature = 0.3
    top_p = 0.8
    max_new = 512

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["zero_shot", "one_shot"],
        default="zero_shot",
        help="Inference mode",
    )

    args = ap.parse_args()

    input_csv = "caption_gt/captioning_ft_full.csv"

    MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    model = torch.compile(model)

    processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
    processor.tokenizer.padding_side = "left"

    output_csv = (
        f"caption_answer/llama_caption_{args.mode}_t-{temperature}_p-{top_p}.csv"
    )

    caption_llama32(
        Path(input_csv),
        Path(output_csv),
        model,
        processor,
        inference_model={args.mode},
        max_new=max_new,
        temperature=temperature,
        top_p=top_p,
    )
