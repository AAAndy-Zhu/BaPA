#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/llava-v1.5-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_result.json

# reproduce: Total: 4241, Correct: 3000, Accuracy: 70.74%, IMG-Accuracy: 68.32%
# for benchmark: Total: 4241, Correct: 3017, Accuracy: 71.14%, IMG-Accuracy: 70.10%