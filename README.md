# Landsat30-AU

## Data Availability

The image data used in this project is stored in a dedicated repository on the Hugging Face Hub. All images are provided as zip files.

### Accessing the Data

You can find and download the dataset from the following URL:

*   **Hugging Face Dataset:** [https://huggingface.co/datasets/supermarkioner/Landsat30-AU](https://huggingface.co/datasets/supermarkioner/Landsat30-AU)

Please download the zip files from the "Files and versions" tab in the repository and extract them to your designated data directory.

## Benchmark Evaluation

To ensure a smooth and error-free benchmark evaluation, we recommend maintaining four distinct Python environments. This approach helps to avoid potential conflicts between the dependencies of different Vision Language Models (VLMs).

### Environment 1: EarthDial

EarthDial requires a dedicated Python environment. Detailed setup instructions can be found at its official GitHub repository:

*   **EarthDial:** [https://github.com/hiyamdebary/EarthDial](https://github.com/hiyamdebary/EarthDial)

### Environment 2: RS-LLaVA

Similarly, RS-LLaVA needs its own isolated environment to function correctly. Please refer to the setup guide available at:

*   **RS-LLaVA:** [https://github.com/BigData-KSU/RS-LLaVA](https://github.com/BigData-KSU/RS-LLaVA)

### Environment 3: Gemma 3

For Gemma 3, a specific transformers environment is necessary. You can find the model and setup details on its Hugging Face page:

*   **Gemma 3:** [https://huggingface.co/google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)

### Environment 4: Shared Environment for Other VLMs

The remaining VLMs can share a single Python environment. We recommend setting up this shared environment using the guidance provided for Qwen, as it is compatible with the other models in this group.

The following VLMs can be installed in this shared environment:

*   **Qwen:** Setup guidance can be found on its Hugging Face page: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct.
*   **LLaVA:** For model documentation, refer to the Hugging Face documentation: https://huggingface.co/docs/transformers/en/model_doc/llava_onevision.
*   **Llama:** The model card is available on Hugging Face: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct.
*   **MiMo:** The official GitHub repository provides detailed information: https://github.com/XiaomiMiMo/MiMo-VL.
*   **GLM-V:** The model card can be accessed on Hugging Face: THUDM/GLM-4.1V-9B-Thinking.

### Evaluation Input Files

The following CSV files are required for the evaluation process. They contain the ground truth data for various tasks.

*   **Image Captioning (Image Captioning Test Set):**
    *   [captioning_ft_full.csv](https://raw.githubusercontent.com/papersubmit1/landsat30-au/main/lightweight_files/caption_gt/captioning_ft_full.csv)
*   **Visual Question Answering (VQA Test Set):**
    *   [Landsat30-AU-VQA-test.csv](https://raw.githubusercontent.com/papersubmit1/landsat30-au/main/lightweight_files/vqa_gt/Landsat30-AU-VQA-test.csv)
*   **Visual Question Answering (One-Shot Set):**
    *   [one_shot.csv](https://raw.githubusercontent.com/papersubmit1/landsat30-au/main/lightweight_files/one_shot_gt/one_shot.csv)

---

## Dataset Construction

### Landsat Imagery

The generation of the Landsat imagery dataset requires the Digital Earth Australia (DEA) Sandbox environment. [1] This is a specialized platform for analyzing Earth observation data.

**Requirements:**

1.  **DEA Sandbox Account:** You must first register for a free account to access the necessary environment. Please follow the official guidance to set up your account:
    *   **Registration:** [DEA Sandbox Homepage](https://app.sandbox.dea.ga.gov.au/) [1]

2.  **Execution Script:** The primary script for processing the imagery is designed to run exclusively within the DEA Sandbox.
    *   **Script:** [Landsat_imagery.py](https://github.com/papersubmit1/landsat30-au/blob/main/Dataset%20Construction/Stage%201%3A%20Imagery%20and%20Metadata%20Preparation/Landsat_imagery.py)

3.  **Input GeoJSON:** This script requires the following GeoJSON file, which defines the geographical areas of interest.
    *   **GeoJSON File:** [keep_4_landuse_metadata-categories.geojson](https://github.com/papersubmit1/landsat30-au/blob/main/lightweight_files/geojson_files/keep_4_landuse_metadata-categories.geojson)


