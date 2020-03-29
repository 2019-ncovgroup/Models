import os
import tensorflow as tf


def input_fn(params, partition):
    [drug_input_size] = params['input_sizes']
    data_dir = params['data_dir']
    file_pattern = os.path.join(data_dir, f'covid.{partition}.*.tfrecords')
    filelist = tf.data.Dataset.list_files(file_pattern)

    def _parse_record_fn(record):
        feature_map = {
            'score': tf.io.FixedLenFeature([1], tf.float32),
            'drug': tf.io.FixedLenFeature([drug_input_size], tf.float32),
        }
        record_features = tf.io.parse_single_example(record, feature_map)

        features = record_features['drug']
        features = tf.cast(features, dtype=tf.float16 if params.get("fp16") else tf.float32)
        label = record_features['score']

        return features, label

    dataset = filelist.interleave(lambda x: tf.data.TFRecordDataset(x).map(_parse_record_fn, num_parallel_calls=1),
                                  cycle_length=4, block_length=16, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # batch -> prefetch -> repeat
    dataset = dataset.batch(batch_size=params['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    if partition == 'train':
        dataset = dataset.repeat()

    return dataset
