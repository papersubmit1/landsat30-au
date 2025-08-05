# -*- coding: utf-8 -*-
"""
This script defines functions to generate structured prompts for various Vision
Language Model (VLM) tasks related to satellite imagery analysis, including
cloud filtering, land cover classification, image captioning, and Visual
Question Answering (VQA) generation and evaluation.
"""

# =============================================================================
# Stage 1: Imagery and Metadata Preparation
# =============================================================================

# -----------------
# Landsat imagery
# -----------------

def cloud_filter_prompt():
    """Generates prompts to classify a satellite image as 'cloudy' or 'clear'."""
    system_prompt = """You are an advanced assistant specializing in analyzing optical satellite images. Your task is to classify each satellite image as either "cloudy" or "clear".

Definitions:
Cloudy: The majority of the image is covered by clouds, obscuring most of the Earth's surface, OR if the image is dominated by features or artifacts (such as sensor bands, stripes, or areas with missing data) that prevent a clear view of ground features.
Clear: The image is mostly free of clouds, and the surface of the Earth is clearly visible.

Instructions:

If clouds or visual obstructions (e.g. striping, missing data, sensor artifacts, over-exposure) cover most of the image and you cannot clearly see the ground features, classify as "cloudy".
If the ground and surface features are mostly visible, classify as "clear".
Respond only with "cloudy" or "clear"."""

    user_prompt = "Please classify the image as either 'cloudy' or 'clear'."

    return system_prompt, user_prompt


# =============================================================================
# Stage 2: Fine-tuning VLMs for Landsat Tasks
# =============================================================================

# -------------------------
# Region classification
# -------------------------

def region_classification_prompt():
    """Generates prompts for detailed land cover classification."""
    system_prompt = """You are an advanced assistant for analyzing an optical satellite image. Your role is using the information from image to accurate answers to the questions to the scene.
Analyze an optical satellite image to classify land cover types. Focus on six classifications: Cultivated Terrestrial Vegetation, Natural Terrestrial Vegetation, Natural Aquatic Vegetation, Artificial Surface, Natural Bare Surfaces, and Water. Pay particular attention to Cultivated Terrestrial Vegetation, Artificial Surface, and Water.
"""

    user_prompt_txt = """Answer these questions:
1. What land cover classifications can be found in the image?
2. Divide the image into five sections: top-left, bottom-left, top-right, bottom-right, and center. For each, list classifications in order of area occupied.
Use this structured format as output:
{'Land Cover Classifications in Optical Image': [list], 'Top-Left Area': [list], 'Top-Right Area': [list], 'Bottom-Left Area': [list], 'Bottom-Right Area': [list], 'Centre Area': [list]}

Examples
{"Land Cover Classifications in Optical Image": ["Natural Terrestrial Vegetation", "Cultivated Terrestrial Vegetation", "Artificial Surface"],"Top-Left Area": ["Cultivated Terrestrial Vegetation", "Artificial Surface"], "Top-Right Area": ["Natural Terrestrial Vegetation", "Cultivated Terrestrial Vegetation"], "Bottom-Left Area": ["Cultivated Terrestrial Vegetation", "Natural Bare Surface"], "Bottom-Right Area": ["Natural Terrestrial Vegetation", "Natural Bare Surface"], "Centre Area": ["Natural Bare Surface"]}"""

    return system_prompt, user_prompt_txt


# ------------------
# Image captioning
# ------------------

def image_captioning_prompt(landcover, landuse):
    """
    Generates prompts for creating a detailed caption from image and metadata.

    Args:
        landcover (str): A string containing land cover information.
        landuse (str): A string containing land use information.

    Returns:
        tuple: A tuple containing the system prompt and the user prompt.
    """
    system_prompt = """Generate a detailed and concise caption from an optical satellite image using provided metadata.

* Please use image content and land cover information to cross-validate land use information. Please use verifed land use information to finish the caption.
* Identify all visible water bodies: rivers (describe their path), lakes, ponds; mention locations and relative sizes using area cues (30m x 30m per pixel).
* Distinguish dominant land cover in each area (bare surface, vegetated, cropland), specify approximate extent or pattern if possible.
* Identify and size artificial areas (“small town”, “city”) and reference exact locations (e.g., “Top-Left Area”) when relevant.
* Describe visible urban features only if seen or confirmed in metadata.
* Note any irrigated field patterns.
* Describe road corridors, specifying directions and links to urban areas, if visible.
* Include any spatial references for features (top, bottom, left, right, center).
* Summarize the balance and dominance between bare and vegetated surfaces.
* Use the land use metadata to offer insights on overall landscape use.
* Incorporate color information for the overall image or specific areas (e.g., "The forests appear dark green," or "The river reflects shades of blue"), describing observed hues and any notable color patterns.
* Caption should be in plain text, clear, and concise without markdown or line breaks."""

    user_prompt = (
        "The following are the metadata to this satellite image:\n"
        "Land Cover Information:\n"
        f"{landcover}\n"
        "Land Use Information:\n"
        f"{landuse}"
    )

    return system_prompt, user_prompt

# ------------------
# Caption review
# ------------------

def caption_keep_or_delete_prompt(sentence):
    """
    Generates prompts to verify if a caption accurately reflects an image.

    Args:
        sentence (str): The caption to be evaluated.

    Returns:
        tuple: A tuple containing the system prompt and the user prompt.
    """
    system_prompt = """Check the given image and its corresponding caption (a single sentence) to determine if the caption accurately reflects the content of the image. Respond with either "delete" if the caption does not match the image content or "keep" if it does.

# Output Format

- Respond with a single word: "delete" or "keep".

# Steps

1. Analyze the content of the provided image to understand its main elements and context.
2. Read the given caption and evaluate its accuracy and relevance to the image content.
3. Decide whether the caption accurately represents the image.
4. Respond accordingly with "delete" or "keep".

# Notes

- The caption should be a clear and direct reflection of the image's primary content.
- Consider the main focus of the image, including any prominent objects, actions, or emotions.
- "Keep" the caption if it correctly and completely represents the image without ambiguity. Otherwise, choose "delete"."""

    user_prompt = f"The given caption: {sentence}\n"

    return system_prompt, user_prompt


# =============================================================================
# Stage 3: Multi-Stage Caption and VQA Generation
# =============================================================================

# --------------------
# Caption refinement
# --------------------

def add_missing_object_prompt(caption):
    """
    Generates prompts to identify missing objects in a caption.

    Args:
        caption (str): The existing image caption.

    Returns:
        tuple: A tuple containing the system prompt and the user prompt.
    """
    system_prompt = """You are an advanced assistant specializing in analyzing optical satellite images. You will get a caption about one image. Your task is to find and describe special missing patterns or objects that appear in the image but not in caption.

Only describe what is clearly visible - do NOT mention anything that is absent or not shown in the image. Avoid making statements about what is not present.

Do not start the response with phrases like "In addition to the features described in the caption," or similar wording—just directly state the missing object information.

Mention only one key missing object or pattern that is clearly visible in the image but not in the caption; keep it concise and ideally contained within a single sentence.

This will instruct me to avoid referencing absent features in my responses."""

    user_prompt = f"The given caption: {caption}\n"

    return system_prompt, user_prompt


def add_missing_connection_prompt(caption):
    """
    Generates prompts to identify missing connections between objects in a caption.

    Args:
        caption (str): The existing image caption.

    Returns:
        tuple: A tuple containing the system prompt and the user prompt.
    """
    system_prompt = """You are an advanced assistant specializing in analyzing optical satellite images. You will get a caption about one image. Your task is to find and describe special missing connections between objects that appear in the image but not in caption.

Only describe what is clearly visible - do NOT mention anything that is absent or not shown in the image. Avoid making statements about what is not present.

Do not start the response with phrases like "In addition to the features described in the caption," or similar wording—just directly state the missing connection or relationship information.

Mention only one key missing connection or relationship that is clearly visible in the image but not in the caption; keep it concise and ideally contained within a single sentence.

This will instruct me to avoid referencing absent features in my responses."""

    user_prompt = f"The given caption: {caption}\n"

    return system_prompt, user_prompt


def extract_key_objects_prompt(given_caption):
    """
    Generates prompts to extract key earth observation objects from a caption.

    Args:
        given_caption (str): The caption from which to extract objects.

    Returns:
        tuple: A tuple containing the system prompt and the user prompt.
    """
    system_prompt = """Extract the key objects directly from the provided caption, focusing on earth observation elements such as natural features, human-made structures, and land use areas. These objects must explicitly appear in the caption and should emphasize notable earth science patterns like oxbow bends.

# Steps

1. Carefully read the provided caption, identifying each object explicitly mentioned.
2. Cross-check each identified object to ensure it directly appears in the caption and falls under categories related to earth observation such as natural features, human-made structures, or land use areas.
3. Give particular attention to identifying and naming distinct natural patterns, such as oxbow bends, formed by meandering rivers.
4. Compile validated objects into a list format.

# Output Format

- Return a JSON array containing strings of all identified key objects relevant to earth observation, directly extracted from the caption.

# Examples

### Example 1
**Input:** "The image shows a large river bending through a dense forest with a small urban area visible on the horizon."
**Output:** ["river", "forest", "urban area"]

### Example 2
**Input:** "A solar farm bordered by a highway with adjacent cropland and a small lake."
**Output:** ["solar farm", "highway", "cropland", "lake"]

### Example 3
**Input:** "Mountains rise in the distance beyond stretches of desert and a nearby reservoir."
**Output:** ["mountains", "desert", "reservoir"]

### Example 4
**Input:** "The landscape is dominated by natural vegetation, featuring oxbow bends in the river path, with cultivated fields and wetlands nearby."
**Output:** ["natural vegetation", "oxbow bends", "river", "cultivated fields", "wetlands"]

# Notes

- Only include objects explicitly mentioned in the caption.
- If an object does not appear word-for-word in the caption, it should be omitted.
- Pay special attention to terminology and synonyms that may describe earth observation features but ensure they appear exactly as in the caption."""

    user_prompt = f"The given caption: {given_caption}\n"

    return system_prompt, user_prompt


# ----------------
# VQA generation
# ----------------

def generate_vqa_prompt(object_list):
    """
    Generates prompts to create VQA questions from an object list.

    Args:
        object_list (list): A list of key object strings.

    Returns:
        tuple: A tuple containing the system prompt and the user prompt.
    """
    system_prompt = """You are an AI that generates multiple-choice questions based on a given image and a key object list. Your questions should focus on four aspects: scene/land-cover identification, object presence, counting, and spatial relations.

Instructions:

Given an input of an image and its associated key object list, follow these steps:

Scene/Land-Cover Identification:

Analyze the environment or landscape in the image.
Generate a multiple-choice question that helps identify or classify the scene type.
Object Presence:

Assess which key objects from the list are visible in the image.
Create a question to confirm or deny the presence of these objects.
Counting:

Count the number of specified key objects visible in the image.
Formulate a question to verify this count.
Spatial Relation:

Evaluate the spatial relationships between notable objects.
Generate a question to describe or identify these relationships.
Output Format:

Produce your output as a Python dictionary with the following structure:

{
    "questions": [
        {
            "type": "scene_land_cover",
            "question": "<question_text>",
            "choices": ["<choice_1>", "<choice_2>", "<choice_3>", "<choice_4>"],
            "answer": "<correct_choice>"
        },
        {
            "type": "object_presence",
            "question": "<question_text>",
            "choices": ["<choice_1>", "<choice_2>", "<choice_3>", "<choice_4>"],
            "answer": "<correct_choice>"
        },
        {
            "type": "counting",
            "question": "<question_text>",
            "choices": ["<choice_1>", "<choice_2>", "<choice_3>", "<choice_4>"],
            "answer": "<correct_choice>"
        },
        {
            "type": "spatial_relation",
            "question": "<question_text>",
            "choices": ["<choice_1>", "<choice_2>", "<choice_3>", "<choice_4>"],
            "answer": "<correct_choice>"
        }
    ]
}
Each question dictionary contains:
"type": The aspect being questioned.
"question": The question text.
"choices": A list of four answer choices (including three distractors and one correct answer in random order).
"answer": The correct answer (as it appears in "choices").
Additional Notes:

Do not indicate the correct answer in the choices themselves.
Ensure distractors are plausible and clearly distinct from the correct answer.
Tailor questions and choices to the context of the image and the provided key object list.
Consider possible visual ambiguities and logical contrasts in distractors.
Example Output:

{
    "questions": [
        {
            "type": "scene_land_cover",
            "question": "What type of environment is shown in the image?",
            "choices": ["Desert", "Beach", "Forest", "Mountain"],
            "answer": "Beach"
        },
        {
            "type": "object_presence",
            "question": "Which of the following objects is visible in the image?",
            "choices": ["Cactus", "Snowman", "Palm Tree", "Skyscraper"],
            "answer": "Palm Tree"
        },
        {
            "type": "counting",
            "question": "How many umbrellas are present in the image?",
            "choices": ["One", "Two", "Three", "Four"],
            "answer": "Two"
        },
        {
            "type": "spatial_relation",
            "question": "What is the spatial relation between the sea and the sand in the image?",
            "choices": [
                "The sea is above the sand",
                "The sea is next to the sand",
                "The sea is under the sand",
                "The sea is far away from the sand"
            ],
            "answer": "The sea is next to the sand"
        }
    ]
}
Use this structure for all outputs."""
    user_prompt = f"object list: [{object_list}]"

    return system_prompt, user_prompt


# =============================================================================
# Benchmark Evaluation
# =============================================================================

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