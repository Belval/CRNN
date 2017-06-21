import os
import sys
import time
import numpy as np
import tensorflow as tf

def load_data(dir):
    return None

def main(args):
    """
        Usage: run.py [iteration_count] [batch_size] [train_data_dir] [test_data_dir] [log_save_dir] [graph_save_dir]
    """

    iteration_count = args[1]
    batch_size = args[2]
    train_data_dir = args[3]
    test_data_dir = args[4]
    log_save_dir = args[5]
    graph_save_dir = args[6]

    train_data = load_data(train_data_dir)
    test_data = load_data(test_data_dir)

    graph = tf.Graph()

    with graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=(-1, 32, 100))
        crnn = CRNN(inputs)

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