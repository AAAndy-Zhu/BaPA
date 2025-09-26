import json
import os
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    mask_size = 42

    prob_results_dir = args.prob_results_dir

    image_list = os.listdir(prob_results_dir)

    crop_size = (mask_size, mask_size)

    position_ids = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

    for position in position_ids:
        all_attention_map = np.zeros((840, 840))
        for dirs in image_list:
            if not os.path.isdir(os.path.join(prob_results_dir, dirs)):
                continue
            if not dirs.endswith(f'_{position}'):
                continue

            for file in os.listdir(os.path.join(prob_results_dir, dirs)):
                if not file.endswith('attention_map.json'):
                    continue
                with open(os.path.join(prob_results_dir, dirs, file), 'r') as f:
                    attention_map = json.load(f)
                    attention_map = np.array(attention_map)
                    attention_map_resized = np.kron(attention_map, np.ones((crop_size[0], crop_size[0])))
                    all_attention_map += attention_map_resized

        plt.imshow(all_attention_map, cmap='YlOrRd')  # alpha controls the transparency of the heatmap
        plt.colorbar()  # Add a colorbar to show the intensity scale
        plt.axis('off')  # Turn off the axis
        plt.tight_layout()  # Adjust the layout
        plt.savefig(args.output_path, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prob_results_dir', type=str, required=True, help='Directory containing perception results')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output image')
    args = parser.parse_args()

    main(args)