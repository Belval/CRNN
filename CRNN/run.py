import os
import sys
import time
import numpy as np
import tensorflow as tf

from scipy.misc import imread, imresize
from model import CRNN

# Constants
CHAR_VECTOR = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\\|:_-+=!@#$%?&*()"\' '
NUM_CLASSES = len(CHAR_VECTOR)

def resize_image(image):
    final_arr = np.zeros((32, 400))
    im_arr = imread(image, mode='L')
    r, c = np.shape(im_arr)
    ratio = float(32 / r)
    im_arr_resized = imresize(im_arr, (32, int(c * ratio)))
    final_arr[:, 0:np.shape(im_arr_resized)[1]] = im_arr_resized
    return final_arr

def create_ground_truth(label):
    """
        Create our ground truth by replacing each char by its index in the CHAR_VECTOR
    """
    return [CHAR_VECTOR.index(l) for l in label.split('_')[1]]


def load_data(folder):
    examples = []

    count = 0
    for f in os.listdir(folder):        
        if count > 1000:
            break
        examples.append(
            (
                resize_image(
                    os.path.join(folder, f)
                ),
                create_ground_truth(
                    f
                )
            )
        )
        print(examples[-1][1])
        count += 1
    return examples

def main(args):
    """
        Usage: run.py [iteration_count] [batch_size] [data_dir] [log_save_dir] [graph_save_dir]
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
        # The input placeholder. As per the paper, the shape is 32x100
        inputs = tf.placeholder(tf.float32, [None, 32, 200])
        
        # The CRNN
        crnn = CRNN(inputs)

        # Our target output
        targets = tf.sparse_placeholder(tf.int32)

        # The length of the sequence
        seq_len = tf.placeholder(tf.int32, [None])

        logits = tf.reshape(crnn, [batch_size, -1, NUM_CLASSES])

        # Final layer, the output of the BLSTM
        logits = tf.transpose(logits, (1, 0, 2))

        # Loss and cost calculation
        loss = tf.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(loss)

        # Training step
        optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)

        # The decoded answer
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

        # The error rate
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        # Train
        
        for it in iteration_count:
            iter_avg_cost = 0
            start = time.time()
            for b in [train_data[x:x*batch_size] for x in range(0, int(len(train_data) / batch_size))]:
                data, labels = b
                cost = sess.run(
                    ['cost'],
                    {
                        inputs: data,
                        words: labels,
                    }
                )
                iter_avg_cost += cost / batch_size

            print('[{}] {} : {}'.format(time.time() - start, it, iter_cost))

        # Test
        data, labels = test_data
        words, cost = sess.run(
            ['words', 'cost'],
            {
                inputs: data,
            }
        )
        print('Result: {} / {} correctly read'.format(len(filter(zip(words, labels))), len(words)))

if __name__=='__main__':
    main(sys.argv)