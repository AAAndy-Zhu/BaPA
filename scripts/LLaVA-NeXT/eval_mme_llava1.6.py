import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import json
from tqdm import tqdm
import re
import random
from PIL import Image

torch.manual_seed(1234)

def eval(args):
    model_id  = args.model_path
    processor = LlavaNextProcessor.from_pretrained(model_id)

    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda")

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

            image = Image.open(image_file)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": qs},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            inputs = processor(
                images=image, text=text, return_tensors="pt"
            )

            inputs = inputs.to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()


            ans_file.write(data['image'] + "\t" + data['question'] + "\t" + data['answer'] + "\t" + output_text + "\n")
            ans_file.flush()
        ans_file.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/llava-v1.6-mistral-7b', help='path to model')
    parser.add_argument('--path_to_mme', type=str, default='/path/to/MME', help='path to MME benchmark')
    parser.add_argument('--output_dir', type=str, default='qwen2.5-vl-mme', help='output dir')
    parser.add_argument('--position_balance', action='store_true', help='whether to use position balance')
    args = parser.parse_args()

    eval(args)
