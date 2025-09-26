import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import json
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

# env: vllm
def eval(args):
    pretrained = args.model_path
    model_name = args.model_name
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map) # Add any other thing you want to pass in llava_model_args

    model.eval()
    model.tie_weights()

    eval_file_path = args.eval_file
    img_path = args.img_path
    answers_file = args.answers_file
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    query = "Determine if there is a sub-image in the given image that matches the text following.\nText: "

    test_data = [json.loads(line) for line in open(eval_file_path)]
    ans_file = open(answers_file, "w")
    for data in tqdm(test_data):
        image_path = os.path.join(img_path, data['image'])
        image = Image.open(image_path)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        caption = data['caption']
        qs = query + caption + '\n' + "The answer should only contain 'Yes' or 'No', without reasoning process.\n"

        conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]

        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=128,
            # Modalities should be the same size as the batch size
            modalities=["image"]*input_ids.shape[0]
        )
        response = tokenizer.batch_decode(cont, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        data['prediction'] = response
        data['qs'] = qs

        ans_file.write(json.dumps(data) + '\n')
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/llava-next-llama3-8b")
    parser.add_argument("--model_name", type=str, default="llava_llama3")
    parser.add_argument("--eval_file", type=str, default="../../prob_datasets/laion_bias_test_easy_10k_840_840.json")
    parser.add_argument("--img_path", type=str, default="../../prob_datasets/images")
    parser.add_argument("--answers_file", type=str, default="./results_laion_bias/llavanext_laion_bias_easy_10k_840_840.json")
    parser.add_argument("--position_balance", action='store_true', default=False, help="whether to use position balance")

    args = parser.parse_args()

    eval(args)
