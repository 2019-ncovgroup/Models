import argparse
import os
import functools
import yaml
import json
import math
from pathlib import Path

import tensorflow as tf

from model import model_fn
from data import input_fn

from cerebras.tf.cs_estimator import CerebrasEstimator
from cerebras.tf.run_config import CSRunConfig
from cerebras.tf.cs_slurm_cluster_resolver import CSSlurmClusterResolver

tf.logging.set_verbosity(tf.logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs_ip", default=None, help="CS-1 IP address, defaults to None")
    parser.add_argument(
        "--mode",
        choices=["validate_only", "compile_only", "train", "eval", "predict"],
        required=True,
        help=(
            "Can choose from validate_only, compile_only, train " +
            "or eval. Defaults to validate_only." +
            "  Validate only will only go up to kernel matching." +
            "  Compile only continues through and generate compiled" +
            "  executables." +
            "  Train will compile and train if on CS-1," +
            "  and just train locally (CPU/GPU) if not on CS-1." +
            "  Eval will run eval locally."
        ),
    )
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

    with open('model_params.yaml', 'r') as model_file:
        model_params = yaml.load(model_file)

    # overwrite model_params with command line arguments
    model_params.update(params)

    # read dataset summary and update model_params
    with open(Path(params['data_dir'], 'summary.json'), 'r') as summary_file:
        summary = json.load(summary_file)
        model_params.update(summary)

    train_steps = math.ceil(model_params['train_examples'] / model_params['batch_size'] / 2.) * 2
    epoch_steps = train_steps * model_params['epochs']
    eval_steps = model_params['test_examples'] // model_params['batch_size']

    # del model_params['learning_rate']
    # model_params['lr_schedule'] = [(0.001, 0), (0.0001, 277000), (0.00001, 554000), (0.000001, 831000)]
    # model_params['xla_compile'] = True
    print("model_params: ", model_params)

    if params["mode"] == "train":
        # CS1 configuration
        use_cs1 = (params["mode"] == "train" and params["cs_ip"] is not None)
        cs_ip = params["cs_ip"] + ":9000" if use_cs1 else None
        model_params["log_metrics"] = False

        slurm_cluster_resolver = CSSlurmClusterResolver(port_base=23111)
        cluster_spec = slurm_cluster_resolver.cluster_spec()
        task_type, task_id = slurm_cluster_resolver.get_task_info()
        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': cluster_spec.as_dict(),
            'task': {
                'type': task_type,
                'index': task_id
            }
        })
        config = CSRunConfig(
            cs_ip=cs_ip,
            log_step_count_steps=train_steps,
            save_checkpoints_steps=epoch_steps,
        )
        estimator = CerebrasEstimator(
            model_fn,
            model_dir=model_params['model_dir'],
            use_cs=use_cs1,
            config=config,
            params=model_params,
        )
        print(f'train_steps: {train_steps}, eval_steps: {eval_steps}, total_training_steps: {epoch_steps}')
        estimator.train(input_fn=functools.partial(input_fn, partition='train'), max_steps=epoch_steps)

    elif params["mode"] == "eval":
        estimator = CerebrasEstimator(
            model_fn,
            model_dir=model_params['model_dir'],
            params=model_params,
        )

        estimator.evaluate(input_fn=functools.partial(input_fn, partition='test'), steps=eval_steps)
    else:
        estimator = CerebrasEstimator(
            model_fn,
            model_dir=model_params['model_dir'],
            params=model_params,
        )
        estimator.compile(input_fn=functools.partial(input_fn, params=model_params, partition='test'),
                          validate_only=(params["mode"] == "validate_only"))

if __name__ == '__main__':
    main()
