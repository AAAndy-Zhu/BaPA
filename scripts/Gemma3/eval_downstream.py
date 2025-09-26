import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import json
from tqdm import tqdm
import argparse

import re
import random
torch.manual_seed(1234)


def eval_downstream(args):

    model_id = args.model_path

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    datasets = ['scienceqa', 'hallusionbench', 'crpe']


    for dataset in datasets:
        POSITION_BALANCE = args.position_balance

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
                qs = data['conversations'][0]['value'] + "\nAnswer with the option's letter from the given choices directly."
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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_file,
                        },
                        {"type": "text", "text": qs},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=128)
                generation = generation[0][input_len:]

            output_text = processor.decode(generation, skip_special_tokens=True)

            data['prediction'] = output_text

            ans_file.write(json.dumps(data) + "\n")
            ans_file.flush()
        ans_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/Gemma3', help='model path')

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

