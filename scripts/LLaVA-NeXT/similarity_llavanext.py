import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from PIL import Image
import requests
from tqdm import tqdm
import json
import torch
import copy

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

def main(args):
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

    test_data = [json.loads(line) for line in open(eval_file_path)]
    test_data = test_data[:9000]
    ans_file = open(answers_file, "w")
    for data in tqdm(test_data):
        caption = data['caption']
        image_file = os.path.join(img_path, data['image'])
        image = Image.open(image_file)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]


        inputs_text = tokenizer_image_token(caption, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]

        text_embeds = model.get_input_embeddings()(inputs_text)[0]
        # print(text_embeds.shape)
        image_embeds = model(
            inputs_text,
            images=image_tensor,
            image_sizes=image_sizes,
            # Modalities should be the same size as the batch size
            modalities=["image"]*inputs_text.shape[0],
            only_return_image_embeds=True
        )
        # print(image_embeds.shape)
        text_embeds_max = torch.max(text_embeds, dim=0).values  # [hidden_size]
        image_embeds_max = torch.max(image_embeds, dim=0).values  # [hidden_size]

        similarity = torch.nn.functional.cosine_similarity(
            text_embeds_max.unsqueeze(0),
            image_embeds_max.unsqueeze(0)
        )[0].item()

        # print(f"Similarity: {similarity}")
        # exit()
        data['similarity'] = similarity
        ans_file.write(json.dumps(data) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/llava-next-llama3-8b")
    parser.add_argument("--model_name", type=str, default="llava_llama3")
    parser.add_argument("--eval_file", type=str, default="../../similarity_data/laion_bias_test_exist_10k_840_840.json")
    parser.add_argument("--img_path", type=str, default="../../similarity_data/images")
    parser.add_argument("--answers_file", type=str, default="./similarity_results/llavanext_similarity.json")
    args = parser.parse_args()

    main(args)
