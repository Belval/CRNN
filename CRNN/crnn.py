import os
import time
import numpy as np
import tensorflow as tf
import config
from scipy.misc import imread, imresize, imsave
from tensorflow.contrib import rnn

from data_manager import DataManager
from utils import sparse_tuple_from, resize_image, label_to_array, ground_truth_to_word, levenshtein

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CRNN(object):
    def __init__(self, batch_size, model_path, examples_path, max_image_width, train_test_ratio):
        self.__data_manager = DataManager(batch_size, model_path, examples_path, max_image_width, train_test_ratio)
        self.__model_path = model_path
        self.__training_name = str(int(time.time()))
        self.__session = tf.Session()

        with self.__session.as_default():
             self.__inputs, self.__targets, self.__seq_len, self.__logits, self.__decoded, self.__optimizer, self.__acc, self.__cost, self.__loss, self.__init, self.__cnn_output, self.__crnn_model, self.__conv1, self.__pool1, self.__conv2, self.__pool2, self.__conv3, self.__conv4, self.__pool3, self.__conv5, self.__bnorm1, self.__conv6, self.__bnorm2, self.__pool4 = self.crnn()
             self.__init.run()

    def crnn(self):
        def BidirectionnalRNN(inputs, seq_len):
            """
                Bidirectionnal LSTM Recurrent Neural Network part
            """

            with tf.variable_scope(None, default_name="bidirectional-rnn-1"):
                # Forward
                lstm_fw_cell_1 = rnn.BasicLSTMCell(256)
                # Backward
                lstm_bw_cell_1 = rnn.BasicLSTMCell(256)

                inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, seq_len, dtype=tf.float32)

                inter_output = tf.concat(inter_output, 2)

            with tf.variable_scope(None, default_name="bidirectional-rnn-2"):
                # Forward
                lstm_fw_cell_2 = rnn.BasicLSTMCell(256)
                # Backward
                lstm_bw_cell_2 = rnn.BasicLSTMCell(256)

                outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, inter_output, seq_len, dtype=tf.float32)

                outputs = tf.concat(outputs, 2)


            return outputs

        def CNN(inputs):
            """
                Convolutionnal Neural Network part
            """

            # 64 / 3 x 3 / 1 / 1
            conv1 = tf.layers.conv2d(inputs=inputs, filters = 64, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 2 x 2 / 1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # 128 / 3 x 3 / 1 / 1
            conv2 = tf.layers.conv2d(inputs=pool1, filters = 128, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 2 x 2 / 1
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # 256 / 3 x 3 / 1 / 1
            conv3 = tf.layers.conv2d(inputs=pool2, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # Batch normalization layer
            bnorm1 = tf.layers.batch_normalization(conv3)

            # 256 / 3 x 3 / 1 / 1
            conv4 = tf.layers.conv2d(inputs=bnorm1, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 1 x 2 / 1
            pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[1, 2], padding="same")

            # 512 / 3 x 3 / 1 / 1
            conv5 = tf.layers.conv2d(inputs=pool3, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # Batch normalization layer
            bnorm2 = tf.layers.batch_normalization(conv5)

            # 512 / 3 x 3 / 1 / 1
            conv6 = tf.layers.conv2d(inputs=bnorm2, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 1 x 2 / 2
            pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=[1, 2], padding="same")

            # 512 / 2 x 2 / 1 / 0
            conv7 = tf.layers.conv2d(inputs=pool4, filters = 512, kernel_size = (2, 2), padding = "valid", activation=tf.nn.relu)

            return conv1, pool1, conv2, pool2, conv3, conv4, pool3, conv5, bnorm1, conv6, bnorm2, pool4, conv7

        inputs = tf.placeholder(tf.float32, [self.__data_manager.batch_size, None, 32, 1])

        # Our target output
        targets = tf.sparse_placeholder(tf.int32, name='targets')

        # The length of the sequence
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        conv1, pool1, conv2, pool2, conv3, conv4, pool3, conv5, bnorm1, conv6, bnorm2, pool4, cnn_output = CNN(inputs)

        cnn_output_2 = tf.reshape(cnn_output, [self.__data_manager.batch_size, -1, 512])

        crnn_model = BidirectionnalRNN(cnn_output_2, seq_len)

        logits = tf.reshape(crnn_model, [-1, 512])

        W = tf.Variable(tf.truncated_normal([512, config.NUM_CLASSES], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[config.NUM_CLASSES]), name="b")

        logits = tf.matmul(logits, W) + b

        logits = tf.reshape(logits, [self.__data_manager.batch_size, -1, config.NUM_CLASSES])

        # Final layer, the output of the BLSTM
        logits = tf.transpose(logits, (1, 0, 2))

        # Loss and cost calculation
        loss = tf.nn.ctc_loss(targets, logits, seq_len)

        cost = tf.reduce_mean(loss)

        # Training step
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        # The decoded answer
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)

        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)

        # The error rate
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

        init = tf.global_variables_initializer()

        return inputs, targets, seq_len, logits, dense_decoded, optimizer, acc, cost, loss, init, cnn_output, crnn_model, conv1, pool1, conv2, pool2, conv3, conv4, pool3, conv5, bnorm1, conv6, bnorm2, pool4

    def train(self, iteration_count):
        with self.__session.as_default():
            print('Training')
            for i in range(iteration_count):
                iter_loss = 0
                for batch_y, batch_sl, batch_x in self.__data_manager.get_next_train_batch():
                    data_targets = np.asarray([label_to_array(lbl, config.CHAR_VECTOR) for lbl in batch_y])
                    data_targets = sparse_tuple_from(data_targets)
                    op, decoded, loss_value, loss_array, cnn, cm, logits = self.__session.run(
                        [self.__optimizer, self.__decoded, self.__cost, self.__loss, self.__cnn_output, self.__crnn_model, self.__logits],
                        feed_dict={
                            self.__inputs: batch_x,
                            self.__seq_len: [24] * self.__data_manager.batch_size,
                            self.__targets: data_targets
                        }
                    )

                    #print(batch_x[0])
                    #print(np.shape(batch_x[0]))
                    #imsave('test.jpg', np.reshape(batch_x[0], [32, 100]))
                    #print(batch_y[0])
                    #print(batch_sl[0])
                    #print(data_targets[0][0:20])
                    #print(len(data_targets[0]))
                    #print(data_targets[1][0:20])
                    #print(len(data_targets[1]))
                    #print(data_targets[2])

                    #print(np.shape(p1))
                    #print(np.shape(p2))
                    #print(np.shape(p3))
                    #print(np.shape(p4))
                    #print(cnn.shape)
                    #print(np.shape(cm))

                    #print(np.shape(logits))

                    if i % 10 == 0:
                        for j in range(2):
                            print(batch_y[j])
                            print(ground_truth_to_word(decoded[j]))

                    iter_loss += loss_value
                print('[{}] Iteration loss: {}'.format(i, iter_loss))
        return None

    def test(self):
        with self.__session.as_default():
            print('Testing')
            total_error = 0
            example_count = 0
            for batch_y, batch_sl, batch_x in self.__data_manager.get_next_test_batch():
                data_targets = np.asarray([label_to_array(lbl, config.CHAR_VECTOR) for lbl in batch_y])
                data_targets = sparse_tuple_from(data_targets)
                decoded = self.__session.run(
                    [self.__decoded],
                    feed_dict={
                        self.__inputs: batch_x,
                        self.__seq_len: batch_sl
                    }
                )
                example_count += len(batch_y)
                total_error += np.sum(levenshtein(ground_truth_to_word(batch_y), ground_truth_to_word(decoded)))
            print('Error on test set: {}'.format(total_error, total_error / example_count))
        return None

    def save(self):
        path = os.path.join(self.__model_path, self.__training_name) if path is None else path

        with open(os.path.join(path, 'crnn.pb'), 'wb') as f:
            f.write(
                tf.graph_util.convert_variables_to_constants(
                    self.__session,
                    self.__session.graph.as_graph_def(),
                    self.__decoded
                ).SerializeToString()
            )
        return None
