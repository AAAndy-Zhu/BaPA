import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import json
from tqdm import tqdm
import re
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

torch.manual_seed(1234)
IMG_TOKEN_LEN = 576


def main(args):
    model_id = args.model_path

    processor = LlavaNextProcessor.from_pretrained(model_id)

    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager")
    model.to("cuda")

    datasets = ['scienceqa', 'hallusionBench', 'crpe']

    for dataset in datasets:
        attentions_all = []
        POSITION_BALANCE = args.position_balance

        print("Tested Dataset:", dataset)
        model.position_balance = POSITION_BALANCE
        print("Position Balance:", model.position_balance)

        if dataset == 'scienceqa':
            eval_file_path = args.eval_file_path_scienceqa
            image_path = args.image_path_scienceqa
        elif dataset == 'hallusionbench':
            eval_file_path = args.eval_file_path_hallusionbench
            image_path = args.image_path_hallusionbench
        elif dataset == 'crpe':
            eval_file_path = args.eval_file_path_crpe
            image_path = args.image_path_crpe
        else:
            raise NotImplementedError('Not Implemented Dataset.')

        output_dir = os.path.join(args.output_dir, dataset)
        os.makedirs(output_dir, exist_ok=True)

        if dataset in ['aokvqa', 'scienceqa', 'hallusionbench']:
            test_data = json.load(open(eval_file_path))
        else:
            test_data = [json.loads(line) for line in open(eval_file_path)]

        for data in tqdm(test_data):

            if dataset == 'scienceqa':
                if 'image' not in data:
                    continue
                question = data['conversations'][0]['value'].replace('<image>', '').strip()
                qs = question + "\nAnswer with the option's letter from the given choices directly."
                image_file = os.path.join(image_path, data['image'])
            elif dataset in ['crpe']:
                qs = data['text']
                image_file = os.path.join(image_path, data['image'])
            elif dataset == 'hallusionbench':
                if data['visual_input'] == "0":
                    continue
                qs = data['question'] + "\nThe answer should only contain 'Yes' or 'No', without reasoning process."
                image_file = os.path.join(image_path, data['filename'])
            else:
                raise NotImplementedError('Only aokvqa and scienceqa are supported.')

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
            # print(text)
            # print(image_inputs)
            inputs = processor(
                images=image, text=text, return_tensors="pt"
            )

            inputs = inputs.to("cuda")

            input_ids = inputs.input_ids
            batch_img_token_pos = torch.where(input_ids == model.config.image_token_index)[1]
            img_token_pos = batch_img_token_pos[0].item()
            # print(img_token_pos)

            outputs = model.generate(**inputs, max_new_tokens=128, output_attentions=True, return_dict_in_generate=True)
            generated_ids = outputs.sequences
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            # print(output_text)

            attentions = torch.stack(outputs.attentions[0], dim=0)
            # print(attentions.shape)
            attentions_mean = torch.mean(attentions, dim=0)
            # print(attentions_mean.shape)
            decoder_attn = torch.mean(attentions_mean, dim=1)
            # print(decoder_attn.shape)
            text_image_if = decoder_attn[0, img_token_pos + IMG_TOKEN_LEN: decoder_attn.shape[1] - 1, img_token_pos: img_token_pos + IMG_TOKEN_LEN].mean(0).cpu()
            # print(text_image_if.shape)

            del attentions, attentions_mean, decoder_attn
            torch.cuda.empty_cache()
            attentions_all.append(text_image_if)

            # print(attentions_all)
        txt_img_ifs = torch.stack(attentions_all)
        txt_img_ifs = torch.mean(txt_img_ifs, dim=0)
        txt_img_ifs = txt_img_ifs.reshape(24, 24).detach().cpu().numpy()

        txt_img_if_max = txt_img_ifs.max()
        txt_img_if_min = txt_img_ifs.min()
        norm_txt_img_if = (txt_img_ifs - txt_img_if_min) / (txt_img_if_max - txt_img_if_min)

        vmin = norm_txt_img_if.min()
        vmax = norm_txt_img_if.max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        plt.imshow(norm_txt_img_if, cmap="viridis", interpolation='nearest', norm=norm)
        plt.colorbar()
        plt.axis('off')

        plt.savefig(os.path.join(output_dir, f'{dataset}_attention.svg'), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/llava-next-1.6-7b")
    parser.add_argument("--output_dir", type=str, default="./attention_results")
    parser.add_argument("--position_balance", action='store_true', default=False, help="whether to use position balance")
    parser.add_argument("--eval_file_path_scienceqa", type=str, default="/path/to/scienceqa_test.json")
    parser.add_argument("--image_path_scienceqa", type=str, default="/path/to/scienceqa_images")
    parser.add_argument("--eval_file_path_hallusionbench", type=str, default="/path/to/hallusionbench_test.json")
    parser.add_argument("--image_path_hallusionbench", type=str, default="/path/to/hallusionbench_images")
    parser.add_argument("--eval_file_path_crpe", type=str, default="/path/to/crpe_test.json")
    parser.add_argument("--image_path_crpe", type=str, default="/path/to/crpe_images")

    args = parser.parse_args()

    main(args)

