import argparse
import ast
import datetime
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import evaluate
import nltk
import pandas as pd
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# --- 1. Configuration ---

class Config:
    """A single class to manage all configuration and settings."""
    # --- File Paths ---
    BASE_DIR = Path(".")  # Assumes files are in the same directory
    STOP_WORDS_FILE = BASE_DIR / "object_from_caption_stop_word_list.txt"
    SYNONYM_MAP_FILE = BASE_DIR / "object_from_caption_categroy_mapping.json"
    GT_FILE = BASE_DIR / "caption_ft_test.csv"
    
    # --- Model Result Files to Evaluate ---
    MODEL_RESULT_FILES = [
        BASE_DIR / "caption_ft_test_reviewed.csv",
        # Add more model result file paths here
    ]
    
    # --- Evaluation Parameters ---
    PRED_OBJECTS_COL = "key_objects"
    REF_OBJECTS_COL = "gt_key_objects"
    PRED_CAPTION_COL = "caption"
    REF_CAPTION_COL = "gt_caption"
    IMAGE_ID_COL = "image_id"
    
    # Max tokens for SPICE/CIDEr CoreNLP compatibility
    CAPTION_TRUNCATE_LIMIT = 50
    
    # --- NLTK Resources ---
    NLTK_RESOURCES = ["punkt", "wordnet", "omw-1.4"]

# --- 2. Resource Loading and Setup ---

def setup_nltk(resources: List[str]):
    """Checks for and downloads required NLTK resources."""
    print("Checking NLTK resources...")
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
            print(f"  - '{resource}' found.")
        except LookupError:
            print(f"  - '{resource}' not found, downloading...")
            nltk.download(resource)
    print("NLTK resource check complete.")

def load_resources(config: Config) -> Tuple[set, Dict]:
    """Loads stop words and synonym map from files."""
    stop_terms = set(config.STOP_WORDS_FILE.read_text(encoding="utf-8").splitlines())
    synonym_map = json.loads(config.SYNONYM_MAP_FILE.read_text(encoding="utf-8"))
    return stop_terms, synonym_map

# --- 3. Metric Calculation Functions ---

def normalize_objects(objects: Iterable[str], stop_terms: set, synonym_map: Dict) -> List[str]:
    """Applies synonym mapping and filters out stop terms."""
    out = []
    for obj in objects:
        canon = synonym_map.get(obj.lower().strip(), obj.lower().strip())
        if canon and canon not in stop_terms:
            out.append(obj if canon == "Remote Sensing VQA pattern" else canon)
    return out

def _safely_eval_list(val: any) -> List[str]:
    """Safely evaluates a string representation of a list."""
    if isinstance(val, float) and math.isnan(val):
        return []
    try:
        return ast.literal_eval(val) if isinstance(val, str) else []
    except (ValueError, SyntaxError):
        return []

def calculate_chair_scores(df: pd.DataFrame, config: Config, stop_terms: set, synonym_map: Dict) -> Dict:
    """Calculates CHAIR-s and CHAIR-i scores for object hallucination."""
    hallucinated_captions = 0
    hallucinated_tokens = 0
    total_tokens = 0

    for _, row in df.iterrows():
        pred_objs_raw = _safely_eval_list(row[config.PRED_OBJECTS_COL])
        ref_objs_raw = _safely_eval_list(row[config.REF_OBJECTS_COL])
        
        pred_objs = set(normalize_objects(pred_objs_raw, stop_terms, synonym_map))
        ref_objs = set(normalize_objects(ref_objs_raw, stop_terms, synonym_map))
        
        hallucinated = pred_objs - ref_objs
        
        if pred_objs:
            total_tokens += len(pred_objs)
            if hallucinated:
                hallucinated_captions += 1
                hallucinated_tokens += len(hallucinated)

    chair_s = 1.0 - (hallucinated_captions / len(df)) if len(df) > 0 else 0.0
    chair_i = 1.0 - (hallucinated_tokens / total_tokens) if total_tokens > 0 else 0.0
    
    return {"CHAIR-s": chair_s, "CHAIR-i": chair_i}

def truncate_caption(text: str, max_tokens: int) -> str:
    """Truncates a caption to a max token limit without splitting sentences where possible."""
    sentences = re.split(r"(?<=[.!?;:])\s+", str(text).strip())
    kept, tokens_so_far = [], 0
    for sent in sentences:
        toks = sent.split()
        if tokens_so_far + len(toks) > max_tokens:
            break
        kept.append(sent)
        tokens_so_far += len(toks)
    return " ".join(kept) if kept else " ".join(sentences[0].split()[:max_tokens])

def calculate_coco_metrics(preds: List[str], refs: List[str], ids: List[str], config: Config) -> Dict:
    """Calculates SPICE, CIDEr, and SPIDEr using the COCO evaluation scripts."""
    gts = {"annotations": [], "images": []}
    res = []
    
    for i, (img_id, ref, pred) in enumerate(zip(ids, refs, preds)):
        gts["images"].append({"id": str(img_id)})
        gts["annotations"].append({"image_id": str(img_id), "id": i, "caption": truncate_caption(ref, config.CAPTION_TRUNCATE_LIMIT)})
        res.append({"image_id": str(img_id), "caption": truncate_caption(pred, config.CAPTION_TRUNCATE_LIMIT)})
    
    # Suppress verbose output from COCO tools
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        coco = COCO()
        coco.dataset = gts
        coco.createIndex()
        coco_res = coco.loadRes(res)
        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.evaluate()
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout # Restore stdout

    spice = coco_eval.eval.get("SPICE", 0.0)
    cider = coco_eval.eval.get("CIDEr", 0.0)
    spider = (spice + cider) / 2.0
    
    return {"SPICE": spice, "CIDEr": cider, "SPIDEr": spider}

def calculate_standard_metrics(preds: List[str], refs: List[str], metrics: Dict) -> Dict:
    """Calculates standard NLP metrics like BLEU and BERTScore."""
    refs_nested = [[r] for r in refs]
    bleu_score = metrics["bleu"].compute(predictions=preds, references=refs_nested)["bleu"]
    bert_scores = metrics["bertscore"].compute(predictions=preds, references=refs, lang="en")
    bert_f1 = sum(bert_scores["f1"]) / len(bert_scores["f1"]) if bert_scores["f1"] else 0.0
    
    return {"BLEU-4": bleu_score, "BERTScore-F1": bert_f1}

# --- 4. Main Pipeline ---

def preprocess_model_output(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    """Applies model-specific cleaning to the caption column."""
    file_name = file_path.name.lower()
    if "glm" in file_name:
        df["caption"] = df["caption"].apply(lambda x: str(x).split("<answer>")[-1].replace("</answer>", ""))
    elif "mimo" in file_name:
        df["caption"] = df["caption"].apply(lambda x: str(x).split("</think>")[-1])
    return df

def main():
    """Main function to orchestrate the entire evaluation pipeline."""
    config = Config()
    
    # --- Setup ---
    setup_nltk(config.NLTK_RESOURCES)
    stop_terms, synonym_map = load_resources(config)
    
    print("Loading evaluation metrics...")
    loaded_metrics = {
        "bleu": evaluate.load("bleu"),
        "bertscore": evaluate.load("bertscore"),
    }
    
    gt_df = pd.read_csv(config.GT_FILE, keep_default_na=False)
    gt_df = gt_df.rename(columns={"caption": config.REF_CAPTION_COL})
    gt_df = gt_df[[config.REF_CAPTION_COL, config.IMAGE_ID_COL]]

    overall_results = []
    
    # --- Evaluation Loop ---
    for model_file in config.MODEL_RESULT_FILES:
        print(f"\n--- Evaluating: {model_file.name} ---")
        model_name = model_file.stem.split("_caption_")[0]
        
        df = pd.read_csv(model_file, keep_default_na=False)
        df = preprocess_model_output(df, model_file)
        df = pd.merge(df, gt_df, on=config.IMAGE_ID_COL, how="left")
        df.dropna(subset=[config.REF_CAPTION_COL, config.PRED_CAPTION_COL], inplace=True)
        
        preds = df[config.PRED_CAPTION_COL].tolist()
        refs = df[config.REF_CAPTION_COL].tolist()
        img_ids = df[config.IMAGE_ID_COL].tolist()
        
        # --- Compute All Metrics ---
        standard_scores = calculate_standard_metrics(preds, refs, loaded_metrics)
        coco_scores = calculate_coco_metrics(preds, refs, img_ids, config)
        chair_scores = calculate_chair_scores(df, config, stop_terms, synonym_map)
        
        avg_len = sum(len(p.split()) for p in preds) / len(preds) if preds else 0.0

        # --- Aggregate and Store Results ---
        all_scores = {
            "model_name": model_name,
            **standard_scores,
            **coco_scores,
            **chair_scores,
            "Caption Length": avg_len
        }
        overall_results.append(all_scores)
        print(pd.DataFrame([all_scores]))

    # --- Save Final Report ---
    if overall_results:
        final_df = pd.DataFrame(overall_results)
        output_path = config.BASE_DIR / "vlm_captioning_evaluation_results.csv"
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ Evaluation complete. Results saved to {output_path}")
    else:
        print("\nNo models were evaluated.")

if __name__ == "__main__":
    main()