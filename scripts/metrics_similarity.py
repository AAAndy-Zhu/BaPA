import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


def main(args):

    position_map_acc = {
        'top_left': (0, 0),
        'top_center': (0, 1),
        'top_right': (0, 2),
        'left': (1, 0),
        'center': (1, 1),
        'right': (1, 2),
        'bottom_left': (2, 0),
        'bottom_center': (2, 1),
        'bottom_right': (2, 2),
    }

    position_sim = {}
    sim = []

    with open(args.result_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['position'] not in position_sim:
                position_sim[data['position']] = []
            position_sim[data['position']].append(data['similarity'])
            sim.append(data['similarity'])

    print('Similarity:', sum(sim) / len(sim))

    for k, v in position_sim.items():
        position_sim[k] = sum(v) / len(v)
        print(k, position_sim[k])

    print('---------------------------------------------')

    matrix = np.zeros((3, 3))

    for pos, value in position_sim.items():
        i, j = position_map_acc[pos]
        matrix[i, j] = value

    sns.heatmap(matrix, annot=True, cmap='YlOrBr', square=True, cbar=True, annot_kws={'size': 15}, vmin=0, vmax=1)
    # plt.title("Heatmap from Named Positions")
    plt.xticks([0.5, 1.5, 2.5], ['Left', 'Center', 'Right'], fontsize=15)
    plt.yticks([0.5, 1.5, 2.5], ['Top', 'Center', 'Bottom'], rotation=0, fontsize=15)
    plt.savefig(args.output_file, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, required=True, help='File containing results with similarity scores')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output heatmap image')
    args = parser.parse_args()

    main(args)