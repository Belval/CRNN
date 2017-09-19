import numpy as np

from scipy.misc import imread, imresize

def sparse_tuple_from(sequences, dtype=np.int32):
    """
        Inspired (copied) from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
    """

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        print(n)
        indices.extend(zip([n]*len(seq), [i for i in range(len(seq))]))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def resize_image(image, input_width):
    """
        Resize an image to the "good" input size
    """

    final_arr = np.zeros((32, input_width))
    im_arr = imread(image, mode='L')
    r, c = np.shape(im_arr)
    ratio = float(32 / r)
    im_arr_resized = imresize(im_arr, (32, int(c * ratio)))
    final_arr[:, 0:np.shape(im_arr_resized)[1]] = im_arr_resized
    return final_arr, c

def to_seq_len(inputs, max_len):
    return np.ones(np.shape(inputs)[0]) * max_len

def labels_to_string(labels, word_string):
    result = ""
    for l in labels:
        result += word_string[l] if l != 0 else '-'

    return result

def label_to_array(label, letters):
    return [letters.index(x) for x in label]