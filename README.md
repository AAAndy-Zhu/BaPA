# Balanced Position Assignment (BaPA)

This repository contains the implementation of [*From Bias to Balance: Exploring and Mitigating Spatial Bias in LVLMs*](https://arxiv.org/abs/2509.21984). In this paper, we systematically investigates spatial robustness in Large Vision-Language Models (LVLMs) and propose **Balanced Position Assignment (BaPA)**, a simple yet effective method to mitigate spatial bias in LVLMs to improve cross-modal reasoning.


## 📊 Data
### Probe Dataset
We design a probe dataset by randomly sampling 10,000 image-caption pairs from LAION. Each image is placed in different spatial positions in a 3×3 grid combined with distractor images, enabling us to evaluate the robustness of LVLMs to spatial variations.

- **The dataset will be made publicly available upon paper acceptance on 🤗Hugging Face.**
- **A subset of the data is already included in this repository for reference.**

### Similarity Dataset
We also construct an auxiliary dataset to measure the cosine similarity between image features and their corresponding caption embeddings across different spatial positions. We randomly select 10,000 image-caption pairs from LAION and constructed 90,000 samples in total as well.

- **The dataset will be made publicly available upon paper acceptance on 🤗Hugging Face.**
- **A subset of the data is already included in this repository for reference.**

### Fine-tuning Dataset
We randomly sample 10,000 instruction-tuning examples from the LLaVA-v1.5 dataset to fine-tune LVLMs for adapting BaPA to general multi-modal downstream tasks.

- **The dataset can be found in `scripts/LLaVA/playground/data/llava_v1_5_instruct_sample_10k_for_llava1.5.json` and `LLaMA-Factory/data/llava_v1_5_instruct_sample_10k.json`.**

### Downstream Benchmark

- **CRPE:** Download the dataset from [🤗Hugging Face](https://huggingface.co/datasets/OpenGVLab/CRPE).
- **HallusionBench:** Download the dataset from [🤗Hugging Face](https://huggingface.co/datasets/rayguan/HallusionBench).
- **ScienceQA:** We provide the formatted test data `llava_test_CQM-A.json` in `ScienceQA/`. You should download and unzip test images (test.zip) from [Google Driver](https://drive.google.com/drive/folders/16kuhXdM-MOhYcFIyRj91WvnDnjnF-xHw).
- **MMMU-Pro:** Clone the repository from [Github](https://github.com/MMMU-Benchmark/MMMU) and download the dataset from [🤗Hugging Face](https://huggingface.co/datasets/MMMU/MMMU_Pro). We also provide the inference scripts in `MMMU/mmmu-pro/infer`.
- **MME:** Download and unzip the [data](https://huggingface.co/datasets/darkyarding/MME/blob/main/MME_Benchmark_release_version.zip) and [eval tool](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/Evaluation/tools/eval_tool.zip) into `MME/`. We also provide formatted test data in JSON in `MME/`.
## 🧩 Models
You can download the following LVLMs directly from 🤗Hugging Face:

- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)  
- [Gemma3-12B](https://huggingface.co/google/gemma-3-12b-it)  
- [LLaVA-v1.6-Mistral-7B](https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b)  
- [Llama3-LLaVA-NeXT-8B](https://huggingface.co/lmms-lab/llama3-llava-next-8b)  
- [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b)  
  
> [!IMPORTANT] 
> After downloading, please modify each model’s `config.json` file by adding the following parameter:

```json
"position_balance": true
```

- Set ``"position_balance": false`` to train or evaluate the original model.
- Set ``"position_balance": true`` to train or evaluate the BaPA-enhanced model.


## ⚙️ Usage

### Environment Setup
```bash
# For Qwen2.5-VL, Gemma3, LLaVA-NeXT and LLaVA-v1.6
conda create -n bapa python=3.10 -y
conda activate bapa
pip install -r requirements.txt
cd ../scripts/LLaVA-NeXT
pip install -e ".[train]"
cd ../../transformers-4.51.3
pip install -e .

# For LLaVA-v1.5
conda create -n llava-bapa python=3.10 -y
conda activate llava-bapa
cd ./scripts/LLaVA
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
cd ../../transformers-4.37.2
pip install -e .

# For LLaMA-Factory
conda create -n llamafactory python=3.10 -y
conda activate llamafactory
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
# More details please refer to https://github.com/hiyouga/LLaMA-Factory.
cd ../transformers-4.51.3
pip install -e .
```

### Training with BaPA (LoRA Fine-tuning)
#### Image Preparation
Download the images of instruction tuning data provided by [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA). 

After downloading all of them, organize the data as follows in `scripts/LLaVA/playground/data`
```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

#### LoRA Fine-tuning with 10K instruction tuning samples
- **For Qwen2.5-VL, Gemma3 and LLaVA-v1.6:**
  1. In `LLaMA-Factory/examples/train_lora/*.yaml` and `LLaMA-Factory/examples/merge_lora/*.yaml`, update the base model path to your downloaded model.
  2. Ensure that the model’s config.json includes:
        ```json
        "position_balance": true
        ```
  3. Run LoRA-based fine-tuning and merge LoRA weights (More details please refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).)
        ```bash
        conda activate llamafactory
        cd LLaMA-Factory
        llamafactory-cli train examples/train_lora/*.yaml
        llamafactory-cli export examples/merge_lora/*.yaml
        ```
- **For LLaVA-v1.5:**
    1. Ensure that the model’s config.json includes:
        ```json
        "position_balance": true
        ```
    2. Run LoRA-based fine-tuning and merge LoRA weights
        ```bash
        conda activate llava-bapa
        cd ./scripts/LLaVA
        sh ./scripts/v1_5/finetune_task_lora.sh
        python ./scripts/merge_lora_weights.py --model-path path/to/fine_tuned_adapter --model-base path/to/base_model --save-model-path path/to/save/model
        ```

### Training LLaVA-v1.5-7B with BaPA from Scratch
If you want to train **LLaVA-v1.5** from scratch, please follow the official [LLaVA-v1.5 repository](https://github.com/haotian-liu/LLaVA) tutorial.  

1. Download the corresponding datasets into: `scripts/LLaVA/playground/data/`
2. Download [Vicuna-7B-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) into `scripts/LLaVA/checkpoints/vicuna-7b-v1.5`
3. Pretrain (feature alignment) and Visual Instruction Tuning
    ```bash
    conda activate llava-bapa
    cd ./scripts/LLaVA
    sh ./scripts/v1_5/pretrain.sh
    sh ./scripts/v1_5/finetune.sh
    ```

### Evaluation on Probe Dataset
```bash
# For Gemma3 and Qwen2.5-VL
conda activate bapa
cd scripts/Gemma3 
# cd scripts/Qwen2.5-VL
python eval_laion_bias_data.py --model_path path/to/model --eval_file path/to/probe/data --img_path path/to/probe/image --answer_file path/to/save/results --position_balance whether/to/use/BaPA

# For LLaVA-NeXT and LLaVA-v1.6
conda activate bapa
cd scripts/LLaVA-NeXT
python eval_laion_bias_data_llavanext.py --model_path path/to/model --eval_file path/to/probe/data --img_path path/to/probe/image --answer_file path/to/save/results --position_balance whether/to/use/BaPA
# python eval_laion_bias_data_llava1.6.py --model_path path/to/model --eval_file path/to/probe/data --img_path path/to/probe/image --answer_file path/to/save/results --position_balance whether/to/use/BaPA

# Generate heatmaps
cd scripts
pythom metrics.py --results_file path/to/saved/result --output_heatmap path/to/save/output/heatmap
```

### Evaluation on Perception Ability of Vision Encoder

```bash
# For Gemma3 
conda activate bapa
cd scripts/Gemma3 
python gemma3_laion_bias_prob.py --model_path path/to/model --eval_file path/to/probe/data --img_path path/to/probe/image --output_dir path/to/save/results

# For Qwen2.5-VL
conda activate bapa
cd scripts/Qwen2.5-VL
python qwen_laion_bias_prob.py --model_path path/to/model --eval_file path/to/probe/data --img_path path/to/probe/image --output_dir path/to/save/results

# For LLaVA-NeXT and LLaVA-v1.6
conda activate bapa
cd scripts/LLaVA-NeXT
python llavanext_laion_bias_prob.py --model_path path/to/model --eval_file path/to/probe/data --img_path path/to/probe/image --output_dir path/to/save/results
# python llava1.6_laion_bias_prob.py --model_path path/to/model --eval_file path/to/probe/data --img_path path/to/probe/image --output_dir path/to/save/results

# Generate heatmaps
cd scripts
pythom prob_results_position.py --prob_results_dir path/to/saved/result --output_path path/to/save/output/heatmaps
```

### Evaluation on Understanding Ability of Vision Encoder

```bash
# For Gemma3 and Qwen2.5-VL
conda activate bapa
cd scripts/Gemma3 
# cd scripts/Qwen2.5-VL
python similarity --model_path path/to/model --eval_file path/to/similarity/data --img_path path/to/similarity/image --answer_file path/to/save/results

# For LLaVA-NeXT and LLaVA-v1.6
conda activate bapa
cd scripts/LLaVA-NeXT
python similarity_llavanext.py --model_path path/to/model --eval_file path/to/similarity/data --img_path path/to/similarity/image --answer_file path/to/save/results
# python similarity_llava1.6.py --model_path path/to/model --eval_file path/to/similarity/data --img_path path/to/similarity/image --answer_file path/to/save/results

# Generate heatmaps
cd scripts
pythom metrics_similarity.py --results_file path/to/saved/results --output_heatmap path/to/save/output/heatmap
```

### Evaluation on ScienceQA, CRPE and HallusionBench
```bash
# For Gemma3 and Qwen2.5-VL
conda activate bapa
cd scripts/Gemma3 
# cd scripts/Qwen2.5-VL
# First modify the arguments in eval_downstream.sh, then
sh scripts/eval_downstream.sh

# For LLaVA-v1.6
conda activate bapa
cd scripts/LLaVA-NeXT
# First modify the arguments in eval_downstream_llava1.6.sh, then
sh scripts/eval_downstream_llava1.6.sh

# For LLaVA-v1.5
conda activate llava-bapa
cd scripts/LLaVA
# First modify the arguments in eval_downstream.sh, then
sh scripts/eval_downstream.sh
```

### Evaluation on MMMU-Pro
```bash
# For Gemma3, Qwen2.5-VL and LLaVA-v1.6
conda activate bapa
cd MMMU/mmmu-pro/infer
python infer_gemma3.py --model path/to/model --dataset_variant standard (10 options)/standard (4 options) --dataset_repo_id path/to/MMMU_Pro_datasets --position_balance True/False
# python infer_qwen2.5vl.py --model path/to/model --dataset_variant standard (10 options)/standard (4 options) --dataset_repo_id path/to/MMMU_Pro_datasets --position_balance True/False
# python infer_llava1.6.py --model path/to/model --dataset_variant standard (10 options)/standard (4 options) --dataset_repo_id path/to/MMMU_Pro_datasets --position_balance True/False

# For LLaVA-v1.5
conda activate llava-bapa
cd MMMU/mmmu-pro/infer
python infer_llava1.5.py --model path/to/model --dataset_variant standard (10 options)/standard (4 options) --dataset_repo_id path/to/MMMU_Pro_datasets --position_balance True/False
```

### Evaluation on MME
```bash
# For Gemma3 and Qwen2.5-VL
conda activate bapa
cd scripts/Gemma3 
# cd scripts/Qwen2.5-VL
python eval_mme.py --model_path path/to/model --path_to_mme path/to/MME/data --output_dir name/of/output/directory --position_balance whether/to/use/BaPA

# For LLaVA-v1.6
conda activate bapa
cd scripts/LLaVA-NeXT
python eval_mme_llava1.6.py --model_path path/to/model --path_to_mme path/to/MME/data --output_dir name/of/output/directory --position_balance whether/to/use/BaPA

# For LLaVA-v1.5
conda activate llava-bapa
cd scripts/LLaVA
python eval_mme.py --model_path path/to/model --path_to_mme path/to/MME/data --output_dir name/of/output/directory --position_balance whether/to/use/BaPA
```

### Visualization of Information Flow
```bash
conda activate bapa
cd scripts/LLaVA-NeXT
# First modify the arguments in attention_downstream_llava1.6.py, then
python attention_downstream_llava1.6.py
```



## 🙏 Acknowledgements
We thank the developers of [LAION](https://laion.ai/), [LLaVA](https://huggingface.co/lmsys/vicuna-7b-v1.5), [Qwen](https://github.com/QwenLM/Qwen3-VL), [Gemma](https://huggingface.co/google/gemma-3-12b-it), [Transformers](https://github.com/huggingface/transformers), [CCA](https://github.com/xing0047/cca-llava) and [Graphical Perception Evaluation](https://github.com/microsoft/lmm-graphical-perception) for open-sourcing their datasets, models and codes. This work builds upon their contributions.
