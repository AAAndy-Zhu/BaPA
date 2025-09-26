import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from copy import deepcopy
import torch

import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor 

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageStat
import json
import random
import argparse
from itertools import islice


mask_color = 'white'  # white background
# mask_color = 'black' # black background

def mask_crop(image, crop_position, crop_size=(14, 14)):
    # get the most common color in the entire image (background)
    # Get the most common pixel color (RGB) in the image
    # pixels = list(image.getdata())
    # background_color = max(set(pixels), key=pixels.count)
    # print("background_color:", background_color)

    background_color = mask_color # white background

    masked_image = image.copy()
    draw = ImageDraw.Draw(masked_image)
    x, y = crop_position
    current_target_mask_region = (x, y, x + crop_size[0], y + crop_size[1])
    # if the pixels in the original region are all the same (so they are background already), then skip
    if len(set(masked_image.crop(current_target_mask_region).getdata())) == 1:
        return False, masked_image
    else:
        draw.rectangle([x, y, x + crop_size[0], y + crop_size[1]], fill=background_color)
        return True, masked_image


def main(args):
    seed = 1

    canvas_width = 840
    canvas_height = 840

    model_path = args.model_path
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path)

    model.position_balance = False

    mask_size = 42

    prob_results_base_dir = args.output_dir
    eval_file_path = args.eval_file
    img_path = args.img_path

    os.makedirs(prob_results_base_dir, exist_ok=True)

    query = "Determine if there is a sub-image in the given image that matches the text following.\nText: "

    # Load and group test data into chunks of 9
    test_data = [json.loads(line) for line in open(eval_file_path)]
    grouped_test_data = [list(islice(test_data, i, i + 9)) for i in range(0, len(test_data), 9)]

    # Randomly select 10 groups
    random.seed(seed)  # Set seed for reproducibility
    selected_groups = random.sample(grouped_test_data, 20)

    # Flatten the selected groups back into a single list
    test_data = [item for group in selected_groups for item in group]

    for data in tqdm(test_data):
        caption = data['caption']
        qs = query + caption + '\n' + "The answer should only contain 'Yes' or 'No', without reasoning process.\n"

        image_file = os.path.join(img_path, data['image'])
        image_id = data['image'].split('.')[0]
        prob_results_dir = os.path.join(prob_results_base_dir, image_id)
        os.makedirs(prob_results_dir, exist_ok=True)
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
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            returned = model.generate(**inputs, max_new_tokens=128, output_scores=True, output_attentions=True, return_dict_in_generate=True)

        generate_ids = returned['sequences'].detach().cpu().numpy()
        # remove input tokens
        # import pdb; pdb.set_trace()
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
        ]
        original_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if data['id'] == 0:
            print(text)
            print(data)
            print(original_response)

        scores = returned['scores']
        # print("scores:", scores)
        desired_ans_scores = []
        desired_ans_indexes = []
        for score in returned['scores'][:-1]: # each step excepts the eos tokens
            # print("score:", score)
            desired_ans_scores.append(torch.max(score).detach().cpu().numpy().tolist())
            desired_ans_indexes.append(torch.argmax(score).detach().cpu().numpy().tolist())

        # print("desired_ans_scores:", desired_ans_scores)
        # exit()
        del returned
        del inputs

        crop_size = (mask_size, mask_size)
        image_size = (canvas_width, canvas_height)
        crop_positions = [(x, y) for x in range(0, image_size[0], crop_size[0]) for y in range(0, image_size[0], crop_size[0])]

        # Store scores for each masked crop
        crop_scores = []

        image = Image.open(image_file)
        for crop_position in tqdm(crop_positions):
            masked, masked_image = mask_crop(image, crop_position, crop_size)
            if not masked:
                crop_scores.append({
                    "crop_position": crop_position,
                    "scores": desired_ans_scores,
                    "response": original_response,
                    "original": True
                })
                continue
            # print('Finish Masking:', crop_position)
            # os.makedirs(os.path.join(prob_results_dir, f'{image_id}_masked'), exist_ok=True)
            # masked_image.save(os.path.join(prob_results_dir, f'{image_id}_masked', f'{crop_position[0]}_{crop_position[1]}.png'))
            messages = [
            {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": masked_image,
                        },
                        {"type": "text", "text": qs},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # print(text)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            with torch.no_grad():
                returned = model.generate(**inputs, max_new_tokens=128, output_scores=True, output_attentions=True, return_dict_in_generate=True)

            generate_ids = returned['sequences']
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            # print(response)

            scores = returned['scores']
            crop_ans_scores = []
            # if scores are not the same length with desired_ans_indexes, then shink to the same length
            min_length = min(len(desired_ans_indexes), len(scores))

            for i in range(min_length):
                crop_ans_scores.append(scores[i][0, desired_ans_indexes[i]].detach().cpu().numpy().tolist())

            crop_scores.append({
                "original_logits": desired_ans_scores,
                "original_response": original_response,
                "crop_position": crop_position,
                "scores": crop_ans_scores,
                "response": response,
                "original": False
            })

        # attention_method = "first_token"
        attention_method = "mean"
        only_top_n = None
        min_len = None
        # min_len = 10


        # Now, use the crop_scores to draw attention maps

        attention_map = np.zeros((image_size[0]//crop_size[0], image_size[0]//crop_size[0]))

        top_scores = []
        top_locations = []
        for crop_score in crop_scores:
            x, y = crop_score['crop_position']
            # print("crop_position:", crop_score['crop_position'])
            # print("crop_score['scores']:", crop_score['scores'])

            if attention_method == "first_token":
                attention_map[y // crop_size[0], x // crop_size[0]] = abs(crop_score['scores'][0] - desired_ans_scores[0])
            elif attention_method == "mean":
                if min_len is not None:
                    min_length = min(len(crop_score['scores']), len(desired_ans_scores), min_len)
                else:
                    min_length = min(len(crop_score['scores']), len(desired_ans_scores))
                for i in range(min_length):
                    attention_map[y // crop_size[0], x // crop_size[0]] += abs(crop_score['scores'][i] - desired_ans_scores[i])
                attention_map[y // crop_size[0], x // crop_size[0]] /= min_length

        # only keep top n and remove other scores
        if only_top_n is not None:
            top_n_indexes = np.argsort(attention_map.flatten())[-only_top_n:]
            for i in range(attention_map.size):
                if i not in top_n_indexes:
                    attention_map[i // attention_map.shape[0], i % attention_map.shape[0]] = 0


        attention_map = np.where(np.isinf(attention_map), 1e5, attention_map)
        # print(attention_map)
        # print("Max attention map value:", np.max(attention_map))
        max_value = np.max(attention_map)
        attention_map = attention_map / max_value
        # print(attention_map)


        # Resize the attention map to match the original image size
        attention_map_resized = np.kron(attention_map, np.ones((crop_size[0], crop_size[0])))

        # print("Question:", qs)
        # print("Answer:", original_response)
        # save attention map
        plt.imshow(image)
        plt.imshow(attention_map_resized, cmap='YlOrRd', alpha=0.8)  # alpha controls the transparency of the heatmap
        plt.colorbar()  # Add a colorbar to show the intensity scale
        plt.axis('off')  # Turn off the axis
        plt.tight_layout()  # Adjust the layout
        plt.savefig(os.path.join(prob_results_dir, f'{image_id}_attention_map.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free up memory

        # save all the scores and attention map
        with open(os.path.join(prob_results_dir, f'{image_id}_crop_scores.json'), 'w') as f:
            json.dump(crop_scores, f, indent=4)

        # save attention_map .json
        with open(os.path.join(prob_results_dir, f'{image_id}_attention_map.json'), 'w') as f:
            json.dump(attention_map.tolist(), f, indent=4)
        # print("Done with id:", image_id)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/Qwen2.5-VL")
    parser.add_argument("--eval_file", type=str, default="../../prob_datasets/laion_bias_test_easy_10k_840_840.json")
    parser.add_argument("--img_path", type=str, default="../../prob_datasets/images")
    parser.add_argument("--output_dir", type=str, default="../qwen2.5_vl_laion_bias_prob/results")
    args = parser.parse_args()

    main(args)
