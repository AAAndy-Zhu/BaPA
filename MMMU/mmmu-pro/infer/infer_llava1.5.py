import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import re
import ast
import json
import yaml
import argparse
import torch
import requests
from PIL import Image
from io import BytesIO

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from tqdm import tqdm
from datasets import load_dataset


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

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def run_inference_on_dataset(dataset, model_path, position_balance):
    print(f"Loading model {args.model}...")
    temperature = 0
    top_p = None
    num_beams = 1
    max_new_tokens = 5

    model_id = model_path

    disable_torch_init()

    model_name = get_model_name_from_path(model_id)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_id, None, "liuhaotian/llava-v1.5-7b"
    )
    print(f"Model loaded. Device: {model.device}")

    model.position_balance = position_balance
    print("Position Balance:", model.position_balance)

    print(f"Processing dataset...")
    results = []
    for data in tqdm(dataset, desc=f"Processing {dataset.info.dataset_name}"):
        qs, images = process_prompt(data, dataset.info.config_name)
        if len(images) != 1:
            print(f"Skipping sample with {len(images)} images (only 1 supported).")
            results.append("")
            continue
        # Construct conversation with one (initial) utterance
        # print(images)
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv_mode_ = None

        if conv_mode_ is not None and conv_mode != conv_mode_:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, conv_mode_, conv_mode_
                )
            )
        else:
            conv_mode_ = conv_mode

        conv = conv_templates[conv_mode_].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        images = [x.convert("RGB") for x in images]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)
        

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print("outputs:", outputs)
        # Add model response to data + remove images
        result_sample = {"response": outputs, **data}
        result_sample = {k: v for k, v in result_sample.items() if not k.startswith("image")}
        results.append(result_sample)
        # print(result_sample)
        # exit()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='path/to/llava-v1.5-7b')
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "cot"])
    parser.add_argument(
        "--dataset_variant",
        type=str,
        default="standard (10 options)",
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
