import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import json
from tqdm import tqdm
import re
import random
import torch.nn as nn
from PIL import Image
torch.manual_seed(1234)


def eval(args):
    POSITION_BALANCE = args.position_balance

    model_path = args.model_path

    processor = LlavaNextProcessor.from_pretrained(model_path)

    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda")

    model.position_balance = POSITION_BALANCE
    print("Position Balance:", model.position_balance)

    eval_file_path = args.eval_file
    img_path = args.img_path
    answers_file = args.answers_file
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    query = "Determine if there is a sub-image in the given image that matches the text following.\nText: "

    test_data = [json.loads(line) for line in open(eval_file_path)]
    ans_file = open(answers_file, "w")
    for data in tqdm(test_data):
        caption = data['caption']

        qs = query + caption + '\n' + "The answer should only contain 'Yes' or 'No', without reasoning process.\n"

        image_file = os.path.join(img_path, data['image'])
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
        )[0]

        data['prediction'] = output_text
        data['qs'] = qs

        ans_file.write(json.dumps(data) + '\n')
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/llava-v1.6-mistral-7b")
    parser.add_argument("--eval_file", type=str, default="../../prob_datasets/laion_bias_test_easy_10k_840_840.json")
    parser.add_argument("--img_path", type=str, default="../../prob_datasets/images")
    parser.add_argument("--answers_file", type=str, default="./results_laion_bias/llava1.6_laion_bias_easy_10k_840_840.json")
    parser.add_argument("--position_balance", action='store_true', default=False, help="whether to use position balance")

    args = parser.parse_args()

    eval(args)
