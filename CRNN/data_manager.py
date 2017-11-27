import re
import os
import numpy as np
import config

from utils import sparse_tuple_from, to_seq_len, resize_image, label_to_array

class DataManager(object):
    def __init__(self, batch_size, model_path, examples_path, max_text_length, max_image_width, train_test_ratio):
        if train_test_ratio > 1.0 or train_test_ratio < 0:
            raise Exception('Incoherent ratio!')

        if max_text_length < 1 or max_image_width < 1:
            raise Exception('Max length and/or width should be above 0')

        print(max_image_width)

        self.train_test_ratio = train_test_ratio
        self.max_image_width = max_image_width
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.model_path = model_path
        self.current_train_offset = 0
        self.examples_path = examples_path
        self.data, self.data_len = self.__load_data()
        self.test_offset = int(train_test_ratio * self.data_len)
        self.current_test_offset = self.test_offset


    def __load_data(self):
        """
            Load all the images in the folder
        """

        print('Loading data')

        examples = []

        count = 0
        for f in os.listdir(self.examples_path):
            if count > 1000:
                break
            arr, initial_len = resize_image(
                os.path.join(self.examples_path, f),
                self.max_image_width
            )
            examples.append(
                (
                    arr,
                    f.split('_')[0],
                    initial_len
                )
            )
            count += 1

        return examples, len(examples)

    def get_next_train_batch(self):
        while not self.current_train_offset + self.batch_size > self.test_offset:
            old_offset = self.current_train_offset

            new_offset = self.current_train_offset + self.batch_size

            self.current_train_offset = new_offset

            raw_batch_x, raw_batch_y, raw_batch_sl = zip(*self.data[old_offset:new_offset])

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            batch_sl = np.reshape(
                np.array(raw_batch_sl),
                (-1)
            )

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (-1, 32, self.max_image_width, 1)
            )

            yield batch_y, batch_sl, batch_x

        self.current_train_offset = 0

    def get_next_test_batch(self):
        while not self.current_test_offset + self.batch_size > self.data_len:
            old_offset = self.current_test_offset

            new_offset = self.current_test_offset + self.batch_size

            self.current_test_offset = new_offset

            raw_batch_x, raw_batch_y, raw_batch_sl = zip(*self.data[old_offset:new_offset])

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            batch_sl = np.reshape(
                np.array(raw_batch_sl),
                (-1)
            )

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (-1, 32, self.max_image_width, 1)
            )

            yield batch_y, batch_sl, batch_x
