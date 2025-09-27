import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from PIL import Image
import requests
from tqdm import tqdm
import json
import torch

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration



def main(args):
    model_path = args.model_path

    processor = LlavaNextProcessor.from_pretrained(model_path)

    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda")

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
                    },
                    {"type": "text", "text": caption},
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

        inputs_text = processor(text=[caption], padding=True, return_tensors="pt").to('cuda')
        text_embeds = model.get_input_embeddings()(inputs_text['input_ids'])[0]
        # print(text_embeds.shape)
        image_embeds = model(**inputs, only_return_image_embeds=True)
        # print(image_embeds.shape)
        text_embeds_mean = torch.max(text_embeds, dim=0).values  # [hidden_size]
        image_embeds_mean = torch.max(image_embeds, dim=0).values  # [hidden_size]

        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(
            text_embeds_mean.unsqueeze(0),
            image_embeds_mean.unsqueeze(0)
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
    parser.add_argument("--model_path", type=str, default="./checkpoints/llava-v1.6-mistral-7b")
    parser.add_argument("--eval_file", type=str, default="../../similarity_data/laion_bias_test_exist_10k_840_840.json")
    parser.add_argument("--img_path", type=str, default="../../similarity_data/images")
    parser.add_argument("--answers_file", type=str, default="./similarity_results/llava1.6_similarity.json")
    args = parser.parse_args()

    main(args)
