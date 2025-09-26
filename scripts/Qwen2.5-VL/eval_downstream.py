import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
from transformers.generation import GenerationConfig
import torch
import json
from tqdm import tqdm
import re
import random
import argparse
torch.manual_seed(1234)


def eval_downstream(args):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path)

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

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

            data['prediction'] = output_text

            ans_file.write(json.dumps(data) + "\n")
            ans_file.flush()
        ans_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/Qwen2.5-VL', help='model path')

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