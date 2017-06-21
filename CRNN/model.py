from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

def CTC(inputs):
    """
        CTC
    """

    # Probably won't work, just a placeholder
    return tf.nn.ctc_greedy_decoder(inputs, tf.shape(inputs)[0])

def BidirectionnalRNN(inputs):
    """
        Bidirectionnal LSTM Recurrent Neural Network part
    """

    # Forward
    lstm_fw_cell = rnn.BasicLSTMCell(256, forget_bias=1.0)
    # Backward
    lstm_bw_cell = rnn.BasicLSTMCell(256, forget_bias=1.0)

    outputs, _, _ = rnn.static_bidirectionnal_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32)

    # TODO: Make sure that this is what we want
    return outputs

def CNN(inputs):
    """
        Convolutionnal Neural Network part
    """

    # 64 / 3 x 3 / 1 / 1
    conv1 = tf.layers.conv2d(inputs = inputs, filters = 64, kernel_size = (3, 3), padding = 1, activation=tf.nn.relu)
    
    # 2 x 2 / 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    # 128 / 3 x 3 / 1 / 1
    conv2 = tf.layers.conv2d(inputs = pool1, filters = 128, kernel_size = (3, 3), padding = 1, activation=tf.nn.relu)
    
    # 2 x 2 / 1
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    # 256 / 3 x 3 / 1 / 1
    conv3 = tf.layers.conv2d(inputs = pool2, filters = 256, kernel_size = (3, 3), padding = 1, activation=tf.nn.relu)
    
    # 256 / 3 x 3 / 1 / 1
    conv4 = tf.layers.conv2d(inputs = conv3, filters = 256, kernel_size = (3, 3), padding = 1, activation=tf.nn.relu)
    
    # 1 x 2 / 1
    pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[1, 2], strides=2)
    
    # 512 / 3 x 3 / 1 / 1
    conv5 = tf.layers.conv2d(inputs = pool3, filters = 512, kernel_size = (3, 3), padding = 1, activation=tf.nn.relu)
    
    # Batch normalization layer
    bnorm1 = tf.layers.batch_normalization(conv5)

    # 512 / 3 x 3 / 1 / 1 
    conv6 = tf.layers.conv2d(inputs = bnorm1, filters = 512, kernel_size = (2, 2), padding = 1, activation=tf.nn.relu)

    #Batch normalization layer
    bnorm2 = tf.layers.batch_normalization(conv6)

    # 1 x 2 / 2
    pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[1, 2], strides=2)

    # 512 / 2 x 2 / 1 / 0     
    conv7 = tf.layers.conv2d(inputs = pool4, filters = 512, kernel_size = (2, 2), padding = 0, activation=tf.nn.relu)

    return conv7

def CRNN(x):
    """
        Feedforward function
    """
    
    inputs = tf.reshape(x, [-1, 32, 100, 1))
    return CTC(
        BidirectionnalRNN(
            CNN(
                inputs
            )
        )
    )