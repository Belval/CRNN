import re
import os
import numpy as np
import tensorflow as tf
from multiprocessing import Queue, Process
from utils import sparse_tuple_from, resize_image, label_to_array

from scipy.misc import imread
from trdg.generators import GeneratorFromDict


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
        use_trdg,
        language,
    ):
        if train_test_ratio > 1.0 or train_test_ratio < 0:
            raise Exception("Incoherent ratio!")

        self.char_vector = char_vector

        self.train_test_ratio = train_test_ratio
        self.max_image_width = max_image_width
        self.batch_size = batch_size
        self.model_path = model_path
        self.current_train_offset = 0
        self.examples_path = examples_path
        self.max_char_count = max_char_count
        self.use_trdg = use_trdg
        self.language = language

        if self.use_trdg:
            self.train_batches = self.multiprocess_batch_generator()
            self.test_batches = self.multiprocess_batch_generator()
        else:
            self.data, self.data_len = self.load_data()
            self.test_offset = int(train_test_ratio * self.data_len)
            self.current_test_offset = self.test_offset
            self.train_batches = self.generate_all_train_batches()
            self.test_batches = self.generate_all_test_batches()

    def batch_generator(self, queue):
        """Takes a queue and enqueue batches in it
        """

        generator = GeneratorFromDict(language=self.language)
        while True:
            batch = []
            while len(batch) < self.batch_size:
                img, lbl = generator.next()
                batch.append(
                    (
                        resize_image(np.array(img.convert("L")), self.max_image_width)[
                            0
                        ],
                        lbl,
                        label_to_array(lbl, self.char_vector),
                    )
                )

            raw_batch_x, raw_batch_y, raw_batch_la = zip(*batch)

            batch_y = np.reshape(np.array(raw_batch_y), (-1))

            batch_dt = sparse_tuple_from(np.reshape(np.array(raw_batch_la), (-1)))

            raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)

            raw_batch_x = raw_batch_x / 255.0

            batch_x = np.reshape(
                np.array(raw_batch_x), (len(raw_batch_x), self.max_image_width, 32, 1)
            )
            if queue.qsize() < 20:
                queue.put((batch_y, batch_dt, batch_x))
            else:
                pass

    def multiprocess_batch_generator(self):
        """Returns a batch generator to use in training
        """

        q = Queue()
        processes = []
        for i in range(2):
            processes.append(Process(target=self.batch_generator, args=(q,)))
            processes[-1].start()
        while True:
            yield q.get()

    def load_data(self):
        """Load all the images in the folder
        """

        print("Loading data")

        examples = []

        count = 0
        skipped = 0
        for f in os.listdir(self.examples_path):
            if len(f.split("_")[0]) > self.max_char_count:
                continue
            arr, initial_len = resize_image(
                imread(os.path.join(self.examples_path, f), mode="L"),
                self.max_image_width,
            )
            examples.append(
                (
                    arr,
                    f.split("_")[0],
                    label_to_array(f.split("_")[0], self.char_vector),
                )
            )
            count += 1

        return examples, len(examples)

    def generate_all_train_batches(self):
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

            raw_batch_x = raw_batch_x / 255.0

            batch_x = np.reshape(
                np.array(raw_batch_x), (len(raw_batch_x), self.max_image_width, 32, 1)
            )

            train_batches.append((batch_y, batch_dt, batch_x))
        return train_batches

    def generate_all_test_batches(self):
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

            raw_batch_x = raw_batch_x / 255.0

            batch_x = np.reshape(
                np.array(raw_batch_x), (len(raw_batch_x), self.max_image_width, 32, 1)
            )

            test_batches.append((batch_y, batch_dt, batch_x))
        return test_batches
