import argparse
import os
from pathlib import Path
import json
import time

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
    parser.add_argument(
        '--prefix',
        default='covid',
        help='filename prefix'
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    scaler = StandardScaler()

    start = time.time()
    _, ext = os.path.splitext(args.data_file)
    if ext == '.csv' or ext == '.gz':
        df = pd.read_csv(args.data_file, low_memory=False)
    elif ext == '.parquet':
        df = pd.read_parquet(args.data_file)
    else:
        print(f'Cannot process {args.data_file}. ext: {ext}')
        exit()
    end = time.time()
    print(f'reading datafile took {end - start} sec')

    PL = df.shape[1]
    # remove duplicated headers
    indexHeaders = df[df['ABC'] == 'ABC'].index
    df.drop(indexHeaders, inplace=True)

    # df_y = df.iloc[:, 0].astype(np.float32)
    start = time.time()
    df_y = df.iloc[:, 0:1]
    df_y.insert(loc=1, column='dock_score', value=df['dock_score'].astype(np.float32))
    end = time.time()
    print(f'df_y processed in {end - start} secs')

    start = time.time()
    df_x = df.iloc[:, 2:PL].astype(np.float32)
    cols = df_x.columns
    df_x[cols] = scaler.fit_transform(df_x)
    end = time.time()
    print(f'df_x processed in {end - start} secs')
    del df

    X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=42)
    del df_x, df_y

    print('x_train shape:', X_train.shape, 'y_train shape:', Y_train.shape)
    print('x_test shape:', X_test.shape, 'y_test shape:', Y_test.shape)

    if not Path(args.out_dir).is_dir():
        Path(args.out_dir).mkdir(parents=True)

    ray.init(object_store_memory=100 * 1024 * 1024 * 1024)

    summary = {}
    start = time.time()
    summary.update(generate_tfrecords(args, 'train', X_train, Y_train))
    end = time.time()
    print(f'train data files are processed in {end - start} secs')
    start = time.time()
    summary.update(generate_tfrecords(args, 'test', X_test, Y_test))
    end = time.time()
    print(f'test data files are processed in {end - start} secs')

    with open(Path(args.out_dir, f'{args.prefix}.summary.json'), 'w') as summary_file:
        summary_file.writelines(json.dumps(summary))


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


def generate_tfrecords(args, partition, features, labels):

    samples_per_batch = 2 ** 16
    num_examples = len(labels)
    num_shards = (num_examples + samples_per_batch - 1) // samples_per_batch  # ceiling division

    stat = {f'{partition}_examples': num_examples}

    shards = list(chunks(range(num_shards), 4))
    for sub_shards in shards:
        print(f'processing {sub_shards}')
        futures = [write_tfr.remote(
            shard=i, args=args, partition=partition,
            features=features, labels=labels, samples_per_batch=samples_per_batch,
            num_examples=num_examples) for i in sub_shards]
        start = time.time()
        ray.wait(futures)
        end = time.time()
        print(f'wait {end - start} secs')

    return stat


@ray.remote
def write_tfr(shard, args, partition, features, labels, samples_per_batch, num_examples):
    print(f'partition: {partition}, shard: {shard}, {samples_per_batch}, {num_examples}')
    start_index = samples_per_batch * shard
    end_index = min(samples_per_batch + start_index, num_examples)
    tfr_path = str(Path(args.out_dir, f'{args.prefix}.{partition}.{shard:02}.tfrecords'))

    with tf.io.TFRecordWriter(tfr_path) as writer:
        for i in range(start_index, end_index):
            feature_dict = {
                'name': _bytes_feature(labels.iloc[i][0].encode('utf-8')),
                'score': _float_feature(labels.iloc[i][1]),
                'drug': _float_feature(features.iloc[i].to_list())
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example.SerializeToString())
    print(f'saved {tfr_path}')


def _bytes_feature(value):
    """
    Creates tf.Train.Feature from bytes value.
    """
    if value is None:
        value = []
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)
        value = bytes(value)
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


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
