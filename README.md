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

The image-captioning captioning_ft_full CSV file: https://raw.githubusercontent.com/papersubmit1/landsat30-au/refs/heads/main/lightweight_files/caption_gt/captioning_ft_full.csv
The VQA one‑shot CSV file: https://raw.githubusercontent.com/papersubmit1/landsat30-au/refs/heads/main/lightweight_files/one_shot_gt/one_shot.csv