import numpy as np

from scipy.misc import imread, imresize

def sparse_tuple_from(sequences, num_classes, dtype=np.int32):
    """
        Create a sparse tuple from input
    """

    indices = []
    values = []

    for t_i, target in enumerate(sequences):
        for seq_i, val in enumerate(target):
            indices.append([t_i, seq_i])
            values.append(val)
    shape = [len(sequences), num_classes + 1]
    return (np.array(indices), np.array(values), np.array(shape))

def resize_image(image, input_width):
    """
        Resize an image to the "good" input size
    """
    
    #final_arr = np.zeros((32, input_width))
    im_arr = imread(image, mode='L')
    #r, c = np.shape(im_arr)
    #ratio = float(32 / r)
    #im_arr_resized = imresize(im_arr, (32, int(c * ratio)))
    #final_arr[:, 0:np.shape(im_arr_resized)[1]] = im_arr_resized
    return im_arr

def to_seq_len(inputs, batch_size):
    return np.ones(np.shape(inputs)[0]) * batch_size