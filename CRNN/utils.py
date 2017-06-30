import numpy as np

def sparse_tuple_from(sequences, num_classes, dtype=np.int32):
    indices = []
    values = []

    for t_i, target in enumerate(sequences):
        for seq_i, val in enumerate(target):
            indices.append([t_i, seq_i])
            values.append(val)
    shape = [len(sequences), num_classes + 1]
    return (np.array(indices), np.array(values), np.array(shape))

def to_seq_len(inputs):
    return np.ones(np.shape(inputs)[0]) * 50