import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import json
from tqdm import tqdm
import re
import random
import argparse
torch.manual_seed(1234)


def eval(args):
    model_id = args.model_path

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    base_dir = args.path_to_mme

    datasets = ['existence', 'position', 'color', 'count']

    for dataset in datasets:
        POSITION_BALANCE = args.position_balance  # False or True

        print("Tested Dataset:", dataset)

        model.position_balance = POSITION_BALANCE
        print("Position Balance:", model.position_balance)

        eval_file_path = base_dir + f'/{dataset}_data.json'
        image_path = base_dir + '/MME_Benchmark'

        answers_file = base_dir + f'eval_tool/results/{args.output_dir}/{dataset}.txt'

        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        test_data = [json.loads(line) for line in open(eval_file_path)]

        ans_file = open(answers_file, "w")
        for data in tqdm(test_data):
            qs = data['question']
            image_file = os.path.join(image_path, data['image'])

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
            # print(text)
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]


            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=128)
                generation = generation[0][input_len:]

            output_text = processor.decode(generation, skip_special_tokens=True)

            # print("Output:", output_text)
            # exit()
            ans_file.write(data['image'] + "\t" + data['question'] + "\t" + data['answer'] + "\t" + output_text + "\n")
            ans_file.flush()
        ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/Qwen2.5-VL', help='path to model')
    parser.add_argument('--path_to_mme', type=str, default='/path/to/MME', help='path to MME benchmark')
    parser.add_argument('--output_dir', type=str, default='qwen2.5-vl-mme', help='output dir')
    parser.add_argument('--position_balance', action='store_true', help='whether to use position balance')
    args = parser.parse_args()

    eval(args)