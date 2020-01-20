import argparse
from crnn import CRNN
from config import CHAR_VECTOR

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description="Train or test the CRNN model.")

    parser.add_argument(
        "-ttr",
        "--train-test-ratio",
        type=float,
        nargs="?",
        help="How the data will be split between training and testing",
        default=0.70,
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        nargs="?",
        help="Learning rate for the adam optimizer",
        default=0.0001,
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        nargs="?",
        help="The path where the pretrained model can be found or where the model will be saved",
        default="./save/",
    )
    parser.add_argument(
        "-ex",
        "--examples-path",
        type=str,
        nargs="?",
        help="The path to the file containing the examples (training samples)",
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, nargs="?", help="Size of a batch", default=64
    )
    parser.add_argument(
        "-it",
        "--iteration-count",
        type=int,
        nargs="?",
        help="How many iteration in training",
        default=10,
    )
    parser.add_argument(
        "-miw",
        "--max-image-width",
        type=int,
        nargs="?",
        help="Maximum width of an example before truncating",
        default=100,
    )
    parser.add_argument(
        "-r",
        "--restore",
        action="store_true",
        help="Define if we try to load a checkpoint file from the save folder",
    )
    parser.add_argument(
        "-cs",
        "--char-set-string",
        type=str,
        nargs="?",
        help="The charset string",
        default=CHAR_VECTOR,
    )
    parser.add_argument(
        "--use-trdg",
        action="store_true",
        help="Generate training data on the fly with TextRecognitionDataGenerator",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        nargs="?",
        help="Language to use with TRDG (Must be used with --use_trdg",
        default="en",
    )

    return parser.parse_args()

def train():
    args = parse_arguments()

    model = CRNN(
        max_width
    )

    model = model.build()

    try:
        os.mkdir(args.model_path)
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise
        pass

    early_stop = EarlyStopping(monitor="val_acc", patience=10, mode="auto")

    # Creating data_manager
    data_manager = DataManager(
        batch_size,
        model_path,
        examples_path,
        max_image_width,
        train_test_ratio,
        self.max_char_count,
        self.CHAR_VECTOR,
        use_trdg,
        language,
    )

    model.compile(
        loss=tf.nn.ctc_loss,#(targets, logits, seq_len, ignore_longer_outputs_than_inputs=True)
        optimizer=Adam(lr=args.learning_rate),
        metrics=["accuracy"],
    )

    mcp_save = ModelCheckpoint('./out/best_wts.h5', verbose=1, save_best_only=True, save_weights_only=True, monitor='val_acc', mode='max')
    tensorboard_callback = keras.callbacks.TensorBoard(f'./logs/', update_freq='epoch')

    hist = model.fit_generator(
        data_manager.train_generator,
        validation_data=data_manager.val_generator,
        steps_per_epoch=c1 // args.batch_size,
        validation_steps=c2 // args.batch_size,
        epochs=args.iteration_count,
        callbacks=[early_stop, mcp_save, tensorboard_callback],
    )

    try:
        df = pd.DataFrame.from_dict(hist.history)
        df.to_csv(os.path.join(args.model_path, "hist.csv"), encoding="utf-8", index=False)
    except Exception as ex:
        print(f"Unable to save histogram: {str(ex)}")

    model.save_weights(os.path.join(args.model_path, f"{int(time.time())}_weights.h5"))

if __name__ == "__main__":
    train()
