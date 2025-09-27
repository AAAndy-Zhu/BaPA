import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from PIL import Image
import requests
from tqdm import tqdm
import json
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse


def main(args):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path)

    model.position_balance = False

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
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_file,
                    },
                    {"type": "text", "text": caption},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        # print(image_inputs)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to('cuda')
        inputs_text = processor(text=[caption], padding=True, return_tensors="pt").to('cuda')
        text_embeds = model.get_input_embeddings()(inputs_text['input_ids'])[0]
        # print(text_embeds.shape)
        image_embeds = model(**inputs, only_return_image_embeds=True)
        # print(image_embeds.shape)
        text_embeds_mean = torch.max(text_embeds, dim=0).values
        image_embeds_mean = torch.max(image_embeds, dim=0).values

        similarity = torch.nn.functional.cosine_similarity(
            text_embeds_mean.unsqueeze(0),
            image_embeds_mean.unsqueeze(0)
        )[0].item()

        # print(f"Similarity: {similarity}")
        data['similarity'] = similarity
        ans_file.write(json.dumps(data) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/Qwen2.5-VL")
    parser.add_argument("--eval_file", type=str, default="../../similarity_data/laion_bias_test_exist_10k_840_840.json")
    parser.add_argument("--img_path", type=str, default="../../similarity_data/images")
    parser.add_argument("--answers_file", type=str, default="./similarity_results/qwen2.5vl_similarity.json")
    args = parser.parse_args()

    main(args)
