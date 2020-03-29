import argparse
from pathlib import Path

import tensorflow as tf

from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import Sequential


def build_model():
    model = Sequential()
    for i, width in enumerate([512, 256, 128, 64, 256, 128, 64, 32]):
        if i == 0:
            model.add(Dense(width, input_dim=1613, activation='relu', name=f'lin_{i}'))
        else:
            model.add(Dense(width, activation='relu', name=f'lin_{i}'))
        model.add(Dropout(0.1, name=f'dr_{i}'))
    model.add(Dense(1, name='pred'))
    return model


def read_checkpoints(args):
    with tf.Session() as sess:
        # import graph
        meta_file_path = str(Path(args.model_dir, f'{args.checkpoint}.meta'))
        saver = tf.compat.v1.train.import_meta_graph(meta_file_path)

        # load weights for graph
        checkpoint_file_path = str(Path(args.model_dir, args.checkpoint))
        saver.restore(sess, checkpoint_file_path)

        # get all global variables (including model variables)
        vars_global = tf.compat.v1.global_variables()

        # get their name and value and put them into dictionary
        sess.as_default()
        model_vars = {}
        for var in vars_global:
            try:
                model_vars[var.name] = var.eval()
            except KeyError:
                print("For var={}, an exception occurred".format(var.name))

    return model_vars


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='cs1.model.h5', help='converted model file name')
    parser.add_argument('--model_dir', default='model_dir', help='model directory')
    parser.add_argument('--checkpoint', default='model.ckpt', help='checkpoint filename')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load variables from checkpoint
    model_vars = read_checkpoints(args)

    # build model
    model = build_model()
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=0.0001, momentum=0.9),
                  metrics=['mae'])

    # set weights
    for i, layer in enumerate(model.layers):
        if layer.name.startswith('dr_'):
            print('skipping dropouts')
        else:
            prefix = f'uno/{layer.name}'
            print(f"layer: {prefix} kernel shape: {model_vars[f'{prefix}/kernel:0'].shape}, bias shape: {model_vars[f'{prefix}/bias:0'].shape}")
            weights = [model_vars[f'{prefix}/kernel:0'], model_vars[f'{prefix}/bias:0']]
            layer.set_weights(weights)

    # model.summary()
    # model_json = model.to_json()
    # with open("cs1_reg.model.json", "w") as json_file:
    #     json_file.write(model_json)
    # model.save_weights("cs1_reg.model.h5")

    model.save(args.out)


if __name__ == '__main__':
    main()
