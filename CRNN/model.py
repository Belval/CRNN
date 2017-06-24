from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

def BidirectionnalRNN(inputs):
    """
        Bidirectionnal LSTM Recurrent Neural Network part
    """

    print(inputs)

    # Forward
    lstm_fw_cell = rnn.BasicLSTMCell(256, forget_bias=1.0)
    # Backward
    lstm_bw_cell = rnn.BasicLSTMCell(256, forget_bias=1.0)

    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32)

    return outputs

def CNN(inputs):
    """
        Convolutionnal Neural Network part
    """

    # 64 / 3 x 3 / 1 / 1
    conv1 = tf.layers.conv2d(inputs = inputs, filters = 64, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
    
    # 2 x 2 / 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    # 128 / 3 x 3 / 1 / 1
    conv2 = tf.layers.conv2d(inputs = pool1, filters = 128, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
    
    # 2 x 2 / 1
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    # 256 / 3 x 3 / 1 / 1
    conv3 = tf.layers.conv2d(inputs = pool2, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
    
    # 256 / 3 x 3 / 1 / 1
    conv4 = tf.layers.conv2d(inputs = conv3, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
    
    # 1 x 2 / 1
    pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[1, 2], strides=2)
    
    # 512 / 3 x 3 / 1 / 1
    conv5 = tf.layers.conv2d(inputs = pool3, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
    
    # Batch normalization layer
    bnorm1 = tf.layers.batch_normalization(conv5)

    # 512 / 3 x 3 / 1 / 1 
    conv6 = tf.layers.conv2d(inputs = bnorm1, filters = 512, kernel_size = (2, 2), padding = "same", activation=tf.nn.relu)

    #Batch normalization layer
    bnorm2 = tf.layers.batch_normalization(conv6)

    # 1 x 2 / 2
    pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[1, 2], strides=2)

    # 512 / 2 x 2 / 1 / 0     
    conv7 = tf.layers.conv2d(inputs = pool4, filters = 512, kernel_size = (2, 2), padding = "same", activation=tf.nn.relu)

    return conv7

def MapToSequence(x):
    #return tf.unstack(
    #    tf.transpose(
    #        tf.squeeze(
    #            tf.reshape(
    #                x,
    #                [-1, 512, 1, 50]
    #            )
    #        ),
    #        perm=[2, 0, 1]
    #    )
    #)
    x = tf.reshape(x, [-1, 512, 1, 50])
    print(x)
    x = tf.transpose(x, perm=[2, 3, 0, 1])
    print(x)
    x = tf.unstack(x)
    print(x)
    return x

def CRNN(x):
    """
        Feedforward function
    """

    inputs = tf.reshape(x, [-1, 32, 400, 1])
    return BidirectionnalRNN(
        MapToSequence(
            CNN(
                inputs
            )
        )
    )