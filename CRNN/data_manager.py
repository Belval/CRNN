import re
import os
import numpy as np

from utils import sparse_tuple_from, resize_image, label_to_array

from scipy.misc import imsave


class DataManager(object):
    def __init__(
        self,
        batch_size,
        model_path,
        examples_path,
        max_image_width,
        train_test_ratio,
        max_char_count,
        char_vector,
    ):
        if train_test_ratio > 1.0 or train_test_ratio < 0:
            raise Exception("Incoherent ratio!")

        print(train_test_ratio)
        self.char_vector = char_vector

        self.train_test_ratio = train_test_ratio
        self.max_image_width = max_image_width
        self.batch_size = batch_size
        self.model_path = model_path
        self.current_train_offset = 0
        self.examples_path = examples_path
        self.max_char_count = max_char_count
        self.data, self.data_len = self.__load_data()
        self.test_offset = int(train_test_ratio * self.data_len)
        self.current_test_offset = self.test_offset
        self.train_batches = self.__generate_all_train_batches()
        self.test_batches = self.__generate_all_test_batches()

    def __load_data(self):
        """
            Load all the images in the folder
        """

        print("Loading data")

        examples = []

        count = 0
        skipped = 0
        for f in os.listdir(self.examples_path):
            if len(f.split("_")[0]) > self.max_char_count:
                continue
            arr, initial_len = resize_image(
                os.path.join(self.examples_path, f), self.max_image_width
            )
            examples.append(
                (
                    arr,
                    f.split("_")[0],
                    label_to_array(f.split("_")[0], self.char_vector),
                )
            )
            imsave("blah.png", arr)
            count += 1

        return examples, len(examples)

    def __generate_all_train_batches(self):
        train_batches = []
        while not self.current_train_offset + self.batch_size > self.test_offset:
            old_offset = self.current_train_offset

            new_offset = self.current_train_offset + self.batch_size

            self.current_train_offset = new_offset

            raw_batch_x, raw_batch_y, raw_batch_la = zip(
                *self.data[old_offset:new_offset]
            )

            batch_y = np.reshape(np.array(raw_batch_y), (-1))

            batch_dt = sparse_tuple_from(np.reshape(np.array(raw_batch_la), (-1)))

            raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)

            batch_x = np.reshape(
                np.array(raw_batch_x), (len(raw_batch_x), self.max_image_width, 32, 1)
            )

            train_batches.append((batch_y, batch_dt, batch_x))
        return train_batches

    def __generate_all_test_batches(self):
        test_batches = []
        while not self.current_test_offset + self.batch_size > self.data_len:
            old_offset = self.current_test_offset

            new_offset = self.current_test_offset + self.batch_size

            self.current_test_offset = new_offset

            raw_batch_x, raw_batch_y, raw_batch_la = zip(
                *self.data[old_offset:new_offset]
            )

            batch_y = np.reshape(np.array(raw_batch_y), (-1))

            batch_dt = sparse_tuple_from(np.reshape(np.array(raw_batch_la), (-1)))

            raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)

            batch_x = np.reshape(
                np.array(raw_batch_x), (len(raw_batch_x), self.max_image_width, 32, 1)
            )

            test_batches.append((batch_y, batch_dt, batch_x))
        return test_batches
