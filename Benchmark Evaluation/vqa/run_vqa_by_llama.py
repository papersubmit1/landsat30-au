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


def vqa_llama32(
    csv_in: Path,
    csv_out: Path,
    model,
    processor,
    inference_model: str = "zero_shot",
    max_new: int = 120,
    temperature: float = 0.3,
    top_p: float = 0.8,
):
    """Iterate the VQA CSV and dump answers produced by Llama 3.2‑Vision."""
    df = pd.read_csv(csv_in, keep_default_na=False)

    done = set()
    if csv_out.exists():
        with open(csv_out, newline="", encoding="utf-8") as f:
            done = {r["qa_id"] for r in csv.DictReader(f) if r.get("answer")}
        df = df[~df["qa_id"].isin(done)]

    fields = ["qa_id", "image_path", "question", "answer", "options", "gt_answer"]

    with open_csv_writer(csv_out, fields) as (writer, fh):
        bar = tqdm(total=len(df), unit="sample", desc="Llama 3.2 VQA")
        try:
            for _, row in df.iterrows():
                images_to_encode = []

                question, opts = row["question"], row["options"]

                img_url = BASE_URL + row["image_path"].strip()
                tgt_img = load_image(img_url, mode="PIL")
                if tgt_img is None:
                    bar.update()
                    continue

                if inference_model == "zero_shot":
                    sys_p, usr_p = evaluate_vlm_vqa_zero_shot(question, opts)

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
                    question_type = row["question_type"]
                    shot_blocks, sys_p, usr_p = evaluate_vlm_vqa_one_shot(
                        one_shot_df, question_type, question, opts
                    )

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
                                        "text": shot_block["full_question"],
                                    },
                                ],
                            }
                        )
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": shot_block["answer"]}
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
                        "question": question,
                        "answer": answer,
                        "options": opts,
                        "gt_answer": row["answer"],
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
    max_new_tokens = 120

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["zero_shot", "one_shot"],
        default="zero_shot",
        help="Inference mode",
    )

    args = ap.parse_args()

    input_csv = "Landsat30-AU-VQA-test.csv"

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

    output_csv = f"vqa_answer/llama_vqa_{args.mode}_t-{temperature}_p-{top_p}.csv"

    vqa_llama32(
        Path(input_csv), Path(output_csv), model, processor, inference_model={args.mode}
    )
