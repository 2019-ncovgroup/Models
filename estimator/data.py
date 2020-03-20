# import os
import tensorflow as tf
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def input_fn(params, partition):
    df = pd.read_parquet('../data/adrp-p1.parquet')
    scaler = StandardScaler()

    PL = df.shape[1] - 1
    df_y = df.iloc[:, 0].astype(np.float32)
    df_y = pd.DataFrame(data=df_y)
    df_x = df.iloc[:, 1:(PL + 1)].astype(np.float32)
    cols = df_x.columns
    df_x[cols] = scaler.fit_transform(df_x)

    X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=42)

    if partition == 'train':
        dataset = tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X_test.values, Y_test.values))

    dataset = dataset.batch(batch_size=params['batch_size'], drop_remainder=True)
    if partition == 'train':
        dataset = dataset.repeat()

    return dataset


# def input_fn(params, partition):
#     drug_input_size = params['input_sizes']
#     data_dir = params['data_dir']
#     file_pattern = os.path.join(data_dir, f'covid.{partition}.*.tfrecords')
#     filelist = tf.data.Dataset.list_files(file_pattern)
#     print(f'input files: {filelist}')

#     def _parse_record_fn(record):
#         feature_map = {
#             'score': tf.io.FixedLenFeature([1], tf.float32),
#             'drug': tf.io.FixedLenFeature([drug_input_size], tf.float32),
#         }
#         record_features = tf.io.parse_single_example(record, feature_map)
#         print(f'record_features {record_features.shape}')

#         features = record_features['drug']
#         features = tf.cast(features, dtype=tf.float16 if params.get("fp16") else tf.float32)
#         label = record_features['score']

#         print(f'score shape: {label.shape}, drug shape: {features.shape}')

#         return features, label

#     dataset = filelist.interleave(lambda x: tf.data.TFRecordDataset(x).map(_parse_record_fn, num_parallel_calls=1),
#                                   cycle_length=4, block_length=16, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset = dataset.batch(batch_size=params['batch_size'], drop_remainder=True)

#     if partition == 'train':
#         dataset = dataset.repeat()

#     return dataset
