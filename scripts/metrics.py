import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


def main(args):
    acc = []
    position_acc = dict()

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

    with open(args.result_file, 'r') as f:
        for line in f:
            data = json.loads(line)

            if data['position'] not in position_acc:
                position_acc[data['position']] = []
            if data['prediction'].startswith('No'):
                data['prediction'] = 'No'
            if data['prediction'].startswith('Yes'):
                data['prediction'] = 'Yes'
            if data['prediction'].split(', ')[0].lower() == data['answer'].lower():
                position_acc[data['position']].append(1)
                acc.append(1)
            else:
                position_acc[data['position']].append(0)
                acc.append(0)

    print('acc:', sum(acc) / len(acc))
    for k, v in position_acc.items():
        position_acc[k] = sum(v) / len(v)
        print(k, position_acc[k])

    print('---------------------------------------------')

    matrix = np.zeros((3, 3))

    for pos, value in position_acc.items():
        i, j = position_map_acc[pos]
        matrix[i, j] = value

    sns.heatmap(matrix, annot=True, cmap='YlGnBu', square=True, cbar=False, annot_kws={'size': 15})
    plt.xticks([0.5, 1.5, 2.5], ['Left', 'Center', 'Right'], fontsize=15)
    plt.yticks([0.5, 1.5, 2.5], ['Top', 'Center', 'Bottom'], rotation=0, fontsize=15)
    plt.savefig(args.output_heatmap, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, required=True, help='Path to the result JSONL file')
    parser.add_argument('--output_heatmap', type=str, required=True, help='Path to save the output heatmap image')
    args = parser.parse_args()

    main(args)
