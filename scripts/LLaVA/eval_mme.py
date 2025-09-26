from llava.eval.run_llava import eval_mme
import os


def eval(args):
    base_dir = args.path_to_mme  # '/path/to/MME'

    datasets = ['existence', 'position', 'color', 'count']

    for dataset in datasets:
        POSITION_BALANCE = args.position_balance

        print("Tested Dataset:", dataset)

        eval_file_path = base_dir + f'/{dataset}_data.json'
        image_path = base_dir + '/MME_Benchmark'
        answers_file = base_dir + f'eval_tool/results/{args.output_dir}/{dataset}.txt'

        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        args = type('Args', (), {
            "model_path": args.model_path,
            "model_base": None,
            "model_name": "liuhaotian/llava-v1.5-7b",
            "query": '',
            "conv_mode": None,
            "image_file": image_path,
            "test_file": eval_file_path,
            "answers_file": answers_file,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 128,
            "position_balance": POSITION_BALANCE
        })

        eval_mme(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/llava-v1.5', help='path to model')
    parser.add_argument('--path_to_mme', type=str, default='/path/to/MME', help='path to MME benchmark')
    parser.add_argument('--output_dir', type=str, default='llava1.5-mme', help='output dir')
    parser.add_argument('--position_balance', action='store_true', help='whether to use position balance')
    args = parser.parse_args()

    eval(args)
