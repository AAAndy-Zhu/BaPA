#!/bin/bash

# python -m llava.eval.model_vqa_loader \
    # --model-path ./checkpoints/llava-v1.5-7b-lora-merged \
    # --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    # --image-folder ./playground/data/eval/textvqa/train_images \
    # --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-for-benchmark.jsonl \
    # --temperature 0 \
    # --conv-mode vicuna_v1

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/textvqa_val_v051_ocr_prediction_qwen2.5vl.json

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/textvqa_val_v051_ocr_prediction_qwen2.5vl_position_1.json

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/textvqa_val_v051_ocr_prediction_qwen2.5vl_position_1_lora.json

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/textvqa_val_v051_ocr_prediction_gemma3.json

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/textvqa_val_v051_ocr_prediction_gemma3_position_1.json

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/textvqa_val_v051_ocr_prediction_gemma3_position_1_lora.json

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/textvqa_val_v051_ocr_prediction_gemma3_position_1_lora_2e_4r.json


# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/textvqa_val_v051_ocr_prediction_phi4.json

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/textvqa_val_v051_ocr_prediction_phi4_position_1.json

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/textvqa_val_v051_ocr_prediction_llavanext.json

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/textvqa_val_v051_ocr_prediction_llavanext_position_1.json


# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/textvqa_val_v051_ocr_prediction_llava1.6.json

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/textvqa_val_v051_ocr_prediction_llava1.6_position_1_lora.json

# reproduce: Accuracy: 58.78%
# for benchmark: Accuracy: 58.35%


# Accuracy of qwen2.5vl: 77.81
# Accuracy of qwen2.5vl_lora: 83.26
# Accuracy of qwen2.5vl_position_1: 77.28
# Accuracy of qwen2.5vl_position_1_lora: 83.00
# Accuracy of gemma3: 69.37
# Accuracy of gemma3_position_1: 59.58
# Accuracy of gemma3_position_1_lora: 62.59
# Accuracy of gemma3_position_1_lora_2e: 63.81
# Accuracy of phi4: 76.09
# Accuracy of phi4_position_1: 30.34
# Accuracy of llavanext: 65.08  
# Accuracy of llavanext_position_1: 11.20