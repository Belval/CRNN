import os
import sys
import time
import numpy as np
import tensorflow as tf

import config
from model import CRNN
from utils import sparse_tuple_from, to_seq_len, resize_image, label_to_array

def create_ground_truth(label):
    """
        Create our ground truth by replacing each char by its index in the CHAR_VECTOR
    """

    return [config.CHAR_VECTOR.index(l) for l in label.split('_')[0]]

def ground_truth_to_word(ground_truth):
    """
        Return the word string based on the input ground_truth
    """

    return ''.join([config.CHAR_VECTOR[i] for i in ground_truth])

def load_data(folder):
    """
        Load all the images in the folder
    """

    examples = []

    count = 0
    for f in os.listdir(folder):
        if count > 1000:
            break
        arr, initial_len = resize_image(
            os.path.join(folder, f),
            config.INPUT_WIDTH
        )
        examples.append(
            (
                arr,
                f.split('_')[0],
                initial_len
            )
        )
        count += 1
    return examples

def main(args):
    """
        Usage: train.py [iteration_count] [batch_size] [data_dir] [log_save_dir] [graph_save_dir]
    """

    # The user-defined training parameters
    iteration_count = int(args[1])
    batch_size = int(args[2])
    data_dir = args[3]
    log_save_dir = args[4]
    graph_save_dir = args[5]

    # The training data
    data = load_data(data_dir)
    train_data = data[0:int(len(data) * 0.70)]
    test_data = data[int(len(data) * 0.70):]

    graph = tf.Graph()

    with graph.as_default():
        inputs = tf.placeholder(tf.float32, [batch_size, 32, None, 1])

        # The CRNN
        crnn = CRNN(inputs)

        # Our target output
        targets = tf.sparse_placeholder(tf.int32, name='targets')

        # The length of the sequence
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        logits = tf.reshape(crnn, [-1, 512])

        W = tf.Variable(tf.truncated_normal([512, config.NUM_CLASSES], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[config.NUM_CLASSES]), name="b")

        print(logits.get_shape())

        logits = tf.matmul(logits, W) + b

        print(logits.get_shape())

        logits = tf.reshape(logits, [batch_size, -1, config.NUM_CLASSES])

        print(logits.get_shape())

        # Final layer, the output of the BLSTM
        logits = tf.transpose(logits, (1, 0, 2))

        # Loss and cost calculation
        loss = tf.nn.ctc_loss(targets, logits, seq_len)

        cost = tf.reduce_mean(loss)

        # Training step
        optimizer = tf.train.MomentumOptimizer(0.01, 0.9).minimize(cost)

        # The decoded answer
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)

        # The error rate
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        # Train

        for it in range(0, iteration_count):
            iter_avg_cost = 0
            start = time.time()
            for b in [train_data[x*batch_size:x*batch_size + batch_size] for x in range(0, int(len(train_data) / batch_size))]:
                in_data, labels, data_seq_len = zip(*b)

                print(data_seq_len)

                data_targets = np.asarray([label_to_array(lbl, config.CHAR_VECTOR) for lbl in labels])

                data_targets = sparse_tuple_from(data_targets)

                print(np.shape(data_targets[0]))
                print(np.shape(data_targets[1]))
                print(np.shape(data_targets[2]))
                print(data_targets[0])
                print(data_targets[1])
                print(data_targets[2])

                data_shape = np.shape(in_data)

                in_data = np.reshape(in_data, (data_shape[0], data_shape[1], data_shape[2], 1))

                decoded_val, cost_val = sess.run(
                    [decoded, cost],
                    {
                        inputs: in_data,
                        targets: data_targets,
                        seq_len: data_seq_len,
                    }
                )
                iter_avg_cost += (np.sum(cost_val) / batch_size) / (int(len(train_data) / batch_size))

            print('[{}] {} : {}'.format(time.time() - start, it, iter_avg_cost))

        # Test
        #in_data, labels, seq_lens = zip(*test_data)
        #decoded_val, cost_val = sess.run(
        #    [decoded, cost],
        #    {
        #        inputs: in_data,
        #        targets: sparse_tuple_from(labels, config.NUM_CLASSES),
        #        seq_len: to_seq_len(seq_lens, config.INPUT_WIDTH),
        #    }
        #)
        #print('Result: {} / {} correctly read'.format(len(filter(zip(decoded_val, labels))), len(decoded_val)))



if __name__=='__main__':
    main(sys.argv)