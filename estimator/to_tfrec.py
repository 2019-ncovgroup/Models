import argparse
import os
from pathlib import Path
import json

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ray


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_file',
        type=str,
        default='adrp-p1.parquet',
        help='datafile to convert'
    )
    parser.add_argument(
        '--out_dir',
        default='./data_dir/',
        help='directory where TFRecords and data info will be stored; '
             'this directory will be expanded - default is ./data_dir/'
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    scaler = StandardScaler()

    _, ext = os.path.splitext(args.data_file)
    if ext == '.csv':
        df = pd.read_csv(args.data_file)
    elif ext == '.parquet':
        df = pd.read_parquet(args.data_file)
    else:
        print(f'Cannot process {args.data_file}. ext: {ext}')
        exit()

    PL = df.shape[1] - 1
    df_y = df.iloc[:, 0].astype(np.float32)
    df_x = df.iloc[:, 1:(PL + 1)].astype(np.float32)
    cols = df_x.columns
    df_x[cols] = scaler.fit_transform(df_x)

    X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=42)

    print('x_train shape:', X_train.shape, 'y_train shape:', Y_train.shape)
    print('x_test shape:', X_test.shape, 'y_test shape:', Y_test.shape)

    if not Path(args.out_dir).is_dir():
        Path(args.out_dir).mkdir(parents=True)

    ray.init()

    summary = {}
    summary.update(generate_tfrecords(args, 'train', X_train, Y_train))
    summary.update(generate_tfrecords(args, 'test', X_test, Y_test))

    with open(Path(args.out_dir, 'summary.json'), 'w') as summary_file:
        summary_file.writelines(json.dumps(summary))


def generate_tfrecords(args, partition, features, labels):

    samples_per_batch = 2 ** 13
    num_examples = len(labels)
    num_shards = (num_examples + samples_per_batch - 1) // samples_per_batch  # ceiling division

    stat = {f'{partition}_examples': num_examples}

    futures = [write_tfr.remote(
        shard=i, args=args, partition=partition,
        features=features, labels=labels, samples_per_batch=samples_per_batch,
        num_examples=num_examples) for i in range(num_shards)]
    ray.get(futures)

    return stat


@ray.remote
def write_tfr(shard, args, partition, features, labels, samples_per_batch, num_examples):
    print(f'partition: {partition}, shard: {shard}, {samples_per_batch}, {num_examples}')
    start_index = samples_per_batch * shard
    end_index = min(samples_per_batch + start_index, num_examples)
    tfr_path = str(Path(args.out_dir, f'covid.{partition}.{shard:02}.tfrecords'))

    with tf.io.TFRecordWriter(tfr_path) as writer:
        for i in range(start_index, end_index):
            feature_dict = {
                'score': _float_feature(labels.iloc[i]),
                'drug': _float_feature(features.iloc[i].to_list())
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example.SerializeToString())
    print(f'saved {tfr_path}')


def _float_feature(value):
    """
    Creates tf.Train.Feature from float value.
    """
    if value is None:
        value = []
    if isinstance(value, np.ndarray) and value.ndim > 1:
        value = value.reshape(-1)
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


if __name__ == '__main__':
    main()
