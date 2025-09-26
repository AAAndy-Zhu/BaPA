import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch
import json
from tqdm import tqdm


def eval(args):

    model_id = args.model_path
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    model.position_balance = args.position_balance
    print("Position Balance:", model.position_balance)

    eval_file_path = args.eval_file
    img_path = args.img_path
    answers_file = args.answers_file

    query = "Determine if there is a sub-image in the given image that matches the text following.\nText: "

    test_data = [json.loads(line) for line in open(eval_file_path)]
    ans_file = open(answers_file, "w")
    for data in tqdm(test_data):
        caption = data['caption']
        qs = query + caption + '\n' + "The answer should only contain 'Yes' or 'No', without reasoning process.\n"

        image_file = os.path.join(img_path, data['image'])
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

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)


        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=128)
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)

        data['prediction'] = decoded
        data['qs'] = qs
        # data['answer'] = data['position'].replace("_", " ")
        data['answer'] = "Yes"

        ans_file.write(json.dumps(data) + '\n')
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/Gemma3")
    parser.add_argument("--eval_file", type=str, default="../../prob_datasets/laion_bias_test_easy_10k_840_840.json")
    parser.add_argument("--img_path", type=str, default="../../prob_datasets/images")
    parser.add_argument("--answers_file", type=str, default="./results_laion_bias/gemma3_laion_bias_easy_10k_840_840.json")
    parser.add_argument("--position_balance", action='store_true', default=False, help="whether to use position balance")

    args = parser.parse_args()

    eval(args)