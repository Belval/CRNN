import tensorflow as tf
import numpy as np
import argparse
from PIL import Image
from utils import resize_image, label_to_string

def arg_parse():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Test a model on data.')

    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        help="The model's path",
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        help="The image to test on",
    )

def main():
    """
        Runs the model on an picture of a line of text
    """

    args = arg_parse()

    img = resize_image(image_path, 2000)

    with tf.gfile.GFile(args.model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)

    with Session(graph=tf.Graph().as_default()) as sess:
        decoded_val = sess.run(
            ["decoded"],
            {
                inputs:
            }
        )
)

if __name__ == '__main__':
    main()