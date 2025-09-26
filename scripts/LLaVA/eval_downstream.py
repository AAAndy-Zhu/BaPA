import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
from itertools import islice
import json
import random
import torch
import torch.nn as nn

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

import requests
from PIL import Image
from io import BytesIO
import re
from tqdm import tqdm


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

def eval_downstream(args):
    temperature = 0
    top_p = None
    num_beams = 1
    max_new_tokens = 128

    model_id = args.model_path

    disable_torch_init()

    model_name = get_model_name_from_path(model_id)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_id, None, "liuhaotian/llava-v1.5-7b"
    )

    datasets = ['scienceqa', 'hallusionbench', 'crpe']


    for dataset in datasets:
        POSITION_BALANCE = args.position_balance # False or True
        # POSITION_BALANCE = False

        print("Tested Dataset:", dataset)

        model.position_balance = POSITION_BALANCE
        print("Position Balance:", model.position_balance)

        if dataset == 'scienceqa':
            eval_file_path = args.eval_file_path_scienceqa
            image_path = args.image_path_scienceqa
            if POSITION_BALANCE:
                answers_file = args.answers_file_scienceqa_balanced
            else:
                answers_file = args.answers_file_scienceqa
        elif dataset == 'hallusionbench':
            eval_file_path = args.eval_file_path_hallusionbench
            image_path = args.image_path_hallusionbench
            if POSITION_BALANCE:
                answers_file = args.answers_file_hallusionbench_balanced
            else:
                answers_file = args.answers_file_hallusionbench
        elif dataset == 'crpe':
            eval_file_path = args.eval_file_path_crpe
            image_path = args.image_path_crpe
            if POSITION_BALANCE:
                answers_file = args.answers_file_crpe_balanced
            else:
                answers_file = args.answers_file_crpe
        else:
            raise NotImplementedError('Not Implemented Dataset.')

        if dataset in ['scienceqa', 'hallusionbench']:
            test_data = json.load(open(eval_file_path))
        else:
            test_data = [json.loads(line) for line in open(eval_file_path)]

        ans_file = open(answers_file, "w")
        for data in tqdm(test_data):
            if dataset == 'scienceqa':
                if 'image' not in data:
                    continue
                qs = data['conversations'][0][
                         'value'] + "\nAnswer with the option's letter from the given choices directly."
                image_file = os.path.join(image_path, data['image'])
            elif dataset == 'crpe':
                qs = data['text']
                image_file = os.path.join(image_path, data['image'])
            elif dataset == 'hallusionbench':
                if data['visual_input'] == "0":
                    continue
                qs = data['question'] + "\nThe answer should only contain 'Yes' or 'No', without reasoning process."
                image_file = os.path.join(image_path, data['filename'])
            else:
                raise NotImplementedError('Not Implemented Dataset.')

            # print(qs)
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
            image_file = image_file.split(',')
            images = load_images(image_file)

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
                output = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    output_attentions=True, return_dict_in_generate=True
                )
            output_ids = output.sequences
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


            data['prediction'] = outputs
            ans_file.write(json.dumps(data) + "\n")
            ans_file.flush()

        ans_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/llava-v1.5-7b', help='model path')

    parser.add_argument('--eval_file_path_scienceqa', type=str, default='/path/to/scienceqa/test.json', help='evaluation file path for scienceqa')
    parser.add_argument('--image_path_scienceqa', type=str, default='/path/to/scienceqa/images', help='image file path for scienceqa')
    parser.add_argument('--answers_file_scienceqa', type=str, default='./results_downstream/scienceqa_answers.json', help='answers file for scienceqa')
    parser.add_argument('--answers_file_scienceqa_balanced', type=str, default='./results_downstream/scienceqa_answers_balanced.json', help='answers file for balanced scienceqa')

    parser.add_argument('--eval_file_path_hallusionbench', type=str, default='/path/to/hallusionbench/test.json', help='evaluation file path for hallusionbench')
    parser.add_argument('--image_path_hallusionbench', type=str, default='/path/to/hallusionbench/images', help='image file path for hallusionbench')
    parser.add_argument('--answers_file_hallusionbench', type=str, default='./results_downstream/hallusionbench_answers.json', help='answers file for hallusionbench')
    parser.add_argument('--answers_file_hallusionbench_balanced', type=str, default='./results_downstream/hallusionbench_answers_balanced.tjsonxt', help='answers file for balanced hallusionbench')

    parser.add_argument('--eval_file_path_crpe', type=str, default='/path/to/crpe/test.jsonl', help='evaluation file path for crpe')
    parser.add_argument('--image_path_crpe', type=str, default='/path/to/crpe/images', help='image file path for crpe')
    parser.add_argument('--answers_file_crpe', type=str, default='./results_downstream/crpe_answers.json', help='answers file for crpe')
    parser.add_argument('--answers_file_crpe_balanced', type=str, default='./results_downstream/crpe_answers_balanced.json', help='answers file for balanced crpe')

    parser.add_argument('--position_balance', action='store_true', help='whether to use position balance during evaluation')

    args = parser.parse_args()

    eval_downstream(args)
