import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import re
import ast
import json
import yaml
import argparse
import torch

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from qwen_vl_utils import process_vision_info



def replace_images_tokens(input_string):
    image_order = [int(num) for num in re.findall(r"<image\s+(\d+)>", input_string)]
    input_string = re.sub(r"<image\s+\d+>", "[image]", input_string)
    return input_string, image_order


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def construct_prompt(doc):
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    question = f"{question}\n{parsed_options}\n{prompt_config['standard']}"
    return question


def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)


def origin_mmmu_doc_to_visual(doc, image_order):
    visual = []
    for idx in image_order:
        visual.append(doc[f"image_{idx}"])
    return visual


def vision_mmmu_doc_to_visual(doc):
    return [doc["image"]]


def process_prompt(data, dataset_variant):
    if "standard" in dataset_variant:
        prompt, image_order = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data, image_order)
    elif "vision" in dataset_variant:
        prompt = prompt_config["vision"]
        images = vision_mmmu_doc_to_visual(data)
    return (prompt, images)


def run_inference_on_dataset(dataset, model_path, position_balance):
    print(f"Loading model {args.model}...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path)
    print(f"Model loaded. Device: {model.device}")

    model.position_balance = position_balance
    print("Position Balance:", model.position_balance)

    print(f"Processing dataset...")
    results = []
    for data in tqdm(dataset, desc=f"Processing {dataset.info.dataset_name}"):
        prompt, images = process_prompt(data, dataset.info.config_name)
        if len(images) != 1:
            print(f"Skipping sample with {len(images)} images (only 1 supported).")
            results.append("")
            continue
        # Construct conversation with one (initial) utterance

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": images[0],
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        # print(inputs.input_ids)
        input_len = inputs["input_ids"].shape[-1]

        input_ids_flat = inputs["input_ids"].flatten()
        idx_image_tokens = (input_ids_flat == 262144).nonzero(as_tuple=True)[0].tolist()
        first_image_token_index = idx_image_tokens[0] if len(idx_image_tokens) > 0 else -1
        len_image_token = len(idx_image_tokens)
        # print(f'262144: first {first_image_token_index}, len {len_image_token}')

        model.first_image_token_index = first_image_token_index
        model.len_image_token = len_image_token

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=128)
            generation = generation[0][input_len:]

        generated_text = processor.decode(generation, skip_special_tokens=True)

        # Add model response to data + remove images
        result_sample = {"response": generated_text, **data}
        result_sample = {k: v for k, v in result_sample.items() if not k.startswith("image")}
        results.append(result_sample)
        # print(result_sample)
        # exit()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='path/to/Gemma3')
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "cot"])
    parser.add_argument(
        "--dataset_variant",
        type=str,
        default="standard (4 options)",
        choices=["vision", "standard (10 options)", "standard (4 options)"],
    )
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--dataset_repo_id", type=str, default="path/to/MMMU_Pro_datasets")
    parser.add_argument("--position_balance", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset_repo_id} ({args.dataset_variant}) split='{args.dataset_split}'...")
    dataset = load_dataset(
        path=args.dataset_repo_id,
        name=args.dataset_variant,
        split=args.dataset_split + ("[:10]" if args.debug else ""),
    )
    print(f"Dataset loaded. Total samples: {len(dataset)}")

    # Load prompt configuration
    with open("../prompts.yaml", "r") as file:
        prompt_config = yaml.safe_load(file)[args.mode]
    print(f"Prompt configuration loaded:\n{prompt_config}")

    results = run_inference_on_dataset(
        dataset=dataset,
        model_path=args.model,
        position_balance=args.position_balance
    )
    print(f"Dataset processed. Total results: {len(results)}")

    # Output directory
    dataset_name = args.dataset_repo_id.split("/")[-1]
    model_name = args.model.split("/")[-1]
    output_path = f"./output/{dataset_name}/{model_name}_{args.dataset_variant}_{args.mode}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            str_result = json.dumps(result, ensure_ascii=False)
            f.write(str_result + "\n")
    print(f"Results saved.")
