#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Script that trains nn classifier
"""

import os
import time
import json
import keras as K
import numpy as np

from keras.utils import np_utils
from gensim.models import Doc2Vec
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from model_nn import model_A, model_B, model_C
from utils import pprint, timeit, init_directory
from train_doc2vec import num_dimensions as num_dimensions


should_aggregate_data   = False

train_arrays_outfile    = 'train_arrays.npy'
test_arrays_outfile     = 'test_arrays.npy'
train_labels_outfile    = 'train_labels.npy'
test_labels_outfile     = 'test_labels.npy'


@timeit
def aggregate_data():

    # aggregate data into train/test arrays with appropriate labels
    parsed_data_files = os.listdir('parsed_data')
    parsed_data_files.sort()
    total_examples = 0

    train_arrays = np.empty((0, num_dimensions))
    train_labels = np.empty(0)

    test_arrays = np.empty((0, num_dimensions))
    test_labels = np.empty(0)

    for file in parsed_data_files:
        with open('parsed_data/' + file, 'r') as f:
            pprint(file)
            if file[:4] == 'test':  # test file
                for idx, line in enumerate(f):
                    test_arrays = np.append(test_arrays,
                                            np.reshape(model[file.replace('-', '_').upper()[:-4] + '_' + str(idx)],
                                                       (1, num_dimensions)), axis=0)
                    test_labels = np.append(test_labels,
                                            file.replace('-', '_').upper()[5:-4]) # ignore preceding 'TEST_' and '.txt' extension
            else:  # train file
                for idx, line in enumerate(f):
                    train_arrays = np.append(train_arrays,
                                             np.reshape(model[file.replace('-', '_').upper()[:-4] + '_' + str(idx)],
                                                        (1, num_dimensions)), axis=0)
                    train_labels = np.append(train_labels,
                                             file.replace('-', '_').upper()[6:-4]) # ignore preceding 'TRAIN_' and '.txt' extension
        total_examples += idx + 1

    np.save(train_arrays_outfile, train_arrays)
    np.save(test_arrays_outfile, test_arrays)
    np.save(train_labels_outfile, train_labels)
    np.save(test_labels_outfile, test_labels)

    return (train_arrays, test_arrays, train_labels, test_labels)


if __name__ == "__main__":
    ts = time.time()

    log_dir = init_directory('./logs/{}'.format(ts))

    # load doc2vec model
    model = Doc2Vec.load('./model.d2v')

    # aggregate data into train/test data/labels
    if should_aggregate_data:
        train_arrays, \
        test_arrays,\
        train_labels,\
        test_labels = aggregate_data()
    else:
        train_arrays    = np.load(train_arrays_outfile)
        test_arrays     = np.load(test_arrays_outfile)
        train_labels    = np.load(train_labels_outfile)
        test_labels     = np.load(test_labels_outfile)

    pprint('train_arrays shape: {}'.format(train_arrays.shape))
    pprint('test_arrays shape: {}'.format(test_arrays.shape))
    pprint('train_labels shape: {}'.format(train_labels.shape))
    pprint('test_labels shape: {}'.format(test_labels.shape))

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(np.concatenate((train_labels, test_labels), axis=0))

    encoded_train_labels = encoder.transform(train_labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    one_hot_train_labels = np_utils.to_categorical(encoded_train_labels)

    encoded_test_labels = encoder.transform(test_labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    one_hot_test_labels = np_utils.to_categorical(encoded_test_labels)

    # build model
    model_nn = model_C(input_shape=(1, num_dimensions), output_shape=(1, one_hot_train_labels.shape[1]))

    pprint(model_nn.summary())

    # create tensorboard and model checkpoint callbacks
    loss_chk_pnt_weights_path = log_dir + '/' + "weights.best-loss.hdf5".format(ts)
    acc_chk_pnt_weights_path = log_dir + '/' + "weights.best-acc.hdf5".format(ts)

    loss_checkpoint_callback = ModelCheckpoint(loss_chk_pnt_weights_path,
                                               monitor='loss',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='min')
    acc_checkpoint_callback = ModelCheckpoint(acc_chk_pnt_weights_path,
                                              monitor='acc',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='max')
    tb_callback = K.callbacks.TensorBoard(log_dir=log_dir,
                                          histogram_freq=0,
                                          write_graph=True,
                                          write_images=True)

    callbacks_list = [loss_checkpoint_callback, acc_checkpoint_callback, tb_callback]

    # train model
    model_nn.fit(train_arrays,
                 one_hot_train_labels,
                 validation_data=(test_arrays, one_hot_test_labels),
                 epochs=100,
                 batch_size=256,
                 callbacks=callbacks_list,
                 verbose=1)

    # write the model architecture to file
    with open(log_dir + '/' + 'model.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        fh.write("Model Structure:\n")
        model_nn.summary(print_fn=lambda x: fh.write(x + '\n'))
        fh.write("\n\n")
        fh.write("========================================================\n")
        fh.write("\n\n")
        fh.write("Model Configuration:\n\n")
        for idx, layer in enumerate(model_nn.layers):
            fh.write("Layer {}:\n\n".format(idx))
            json.dump(layer.get_config(), fh)
            fh.write("\n\n")

    # write the compiled model to json file
    with open(log_dir + '/' + 'model.json', 'w') as fh:
        fh.write(model_nn.to_json())

    # write the compiled model/weights/optimization state to hdf5 file
    model_nn.save(log_dir + '/' + 'model.h5')