import argparse
import functools
import yaml
import json
import math
from pathlib import Path

import tensorflow as tf

from model import model_fn
from data import input_fn

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "predict"],
        required=True,
        help="Running mode")
    parser.add_argument(
        "--model_dir",
        default="./model_dir",
        help="Save outputs")
    parser.add_argument(
        "--data_dir",
        default="./data_dir/top21_auc/",
        help="Data file location")

    params = vars(parser.parse_args())

    return params


def main():
    params = parse_args()

    # match parameters to uno_auc_model.txt
    with open('model_params.yaml', 'r') as model_file:
        model_params = yaml.load(model_file, Loader=yaml.FullLoader)

    # overwrite params
    model_params.update(params)

    # read dataset summary and update model_params
    with open(Path(params['data_dir'], 'summary.json'), 'r') as summary_file:
        summary = json.load(summary_file)
        model_params.update(summary)

    train_steps = math.ceil(model_params['train_examples'] / model_params['batch_size'] / 2.) * 2
    epoch_steps = train_steps * model_params['epochs']
    eval_steps = model_params['test_examples'] // model_params['batch_size']

    config = tf.estimator.RunConfig(log_step_count_steps=train_steps)

    estimator = tf.estimator.Estimator(
        model_fn,
        model_dir=model_params['model_dir'],
        config=config,
        params=model_params
    )

    if params["mode"] == "train":
        print(f'train_steps: {train_steps}, eval_steps: {eval_steps}, total_training_steps: {epoch_steps}')
        estimator.train(input_fn=functools.partial(input_fn, partition='train'), max_steps=epoch_steps)
    elif params["mode"] == "eval":
        estimator.evaluate(input_fn=functools.partial(input_fn, partition='test'), steps=eval_steps)


if __name__ == '__main__':
    main()