"""This script demonstrates how to train the model
on the SynthText90 using multiple GPUs."""
# pylint: disable=invalid-name
import datetime
import argparse
import math
import random
import string
import functools
import itertools
import os
import tarfile
import urllib.request

import numpy as np
import cv2
import imgaug
import tqdm
import tensorflow as tf

import keras_ocr


# pylint: disable=redefined-outer-name
def get_filepaths(data_path, split):
    """Get the list of filepaths for a given split (train, val, or test)."""
    with open(
        os.path.join(data_path, f"mnt/ramdisk/max/90kDICT32px/annotation_{split}.txt"),
        "r",
    ) as text_file:
        filepaths = [
            os.path.join(
                data_path, "mnt/ramdisk/max/90kDICT32px", line.split(" ")[0][2:]
            )
            for line in text_file.readlines()
        ]
    return filepaths


# pylint: disable=redefined-outer-name
def download_extract_and_process_dataset(data_path):
    """Download and extract the synthtext90 dataset."""
    archive_filepath = os.path.join(data_path, "mjsynth.tar.gz")
    extraction_directory = os.path.join(data_path, "mnt")
    if not os.path.isfile(archive_filepath) and not os.path.isdir(extraction_directory):
        print("Downloading the dataset.")
        urllib.request.urlretrieve(
            "https://thor.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz",
            archive_filepath,
        )
    if not os.path.isdir(extraction_directory):
        print("Extracting files.")
        with tarfile.open(os.path.join(data_path, "mjsynth.tar.gz")) as tfile:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tfile, data_path)


def get_image_generator(filepaths, augmenter, width, height):
    """Get an image generator for a list of SynthText90 filepaths."""
    filepaths = filepaths.copy()
    for filepath in itertools.cycle(filepaths):
        text = filepath.split(os.sep)[-1].split("_")[1].lower()
        image = cv2.imread(filepath)
        if image is None:
            print(f"An error occurred reading: {filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = keras_ocr.tools.fit(
            image,
            width=width,
            height=height,
            cval=np.random.randint(low=0, high=255, size=3).astype("uint8"),
        )
        if augmenter is not None:
            image = augmenter.augment_image(image)
        if filepath == filepaths[-1]:
            random.shuffle(filepaths)
        yield image, text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--model_id",
        default="recognizer",
        help="The name to use for saving model checkpoints.",
    )
    parser.add_argument(
        "--data_path",
        default=".",
        help="The path to the directory containing the dataset and where we will put our logs.",
    )
    parser.add_argument(
        "--logs_path",
        default="./logs",
        help=(
            "The path to where logs and checkpoints should be stored. "
            'If a checkpoint matching "model_id" is found, training will resume from that point.'
        ),
    )
    parser.add_argument(
        "--batch_size", default=16, help="The training batch size to use."
    )
    parser.add_argument(
        "--no-file-verification", dest="verify_files", action="store_false"
    )
    parser.set_defaults(verify_files=True)
    args = parser.parse_args()
    weights_path = os.path.join(args.logs_path, args.model_id + ".h5")
    csv_path = os.path.join(args.logs_path, args.model_id + ".csv")
    download_extract_and_process_dataset(args.data_path)
    with tf.distribute.MirroredStrategy().scope():
        recognizer = keras_ocr.recognition.Recognizer(
            alphabet=string.digits + string.ascii_lowercase,
            height=31,
            width=200,
            stn=False,
            optimizer=tf.keras.optimizers.RMSprop(),
            weights=None,
        )
    if os.path.isfile(weights_path):
        print("Loading saved weights and creating new version.")
        dt_string = datetime.datetime.now().isoformat()
        weights_path = os.path.join(
            args.logs_path, args.model_id + "_" + dt_string + ".h5"
        )
        csv_path = os.path.join(
            args.logs_path, args.model_id + "_" + dt_string + ".csv"
        )
        recognizer.model.load_weights(weights_path)
    augmenter = imgaug.augmenters.Sequential(
        [
            imgaug.augmenters.Multiply((0.9, 1.1)),
            imgaug.augmenters.GammaContrast(gamma=(0.5, 3.0)),
            imgaug.augmenters.Invert(0.25, per_channel=0.5),
        ]
    )
    os.makedirs(args.logs_path, exist_ok=True)

    training_filepaths, validation_filepaths = [
        get_filepaths(data_path=args.data_path, split=split)
        for split in ["train", "val"]
    ]
    if args.verify_files:
        assert all(
            os.path.isfile(filepath)
            for filepath in tqdm.tqdm(
                training_filepaths + validation_filepaths, desc="Checking filepaths."
            )
        ), "Some files appear to be missing."

    (training_image_generator, training_steps), (
        validation_image_generator,
        validation_steps,
    ) = [
        (
            get_image_generator(
                filepaths=filepaths,
                augmenter=augmenter,
                width=recognizer.model.input_shape[2],
                height=recognizer.model.input_shape[1],
            ),
            math.ceil(len(filepaths) / args.batch_size),
        )
        for filepaths, augmenter in [
            (training_filepaths, augmenter),
            (validation_filepaths, None),
        ]
    ]

    training_generator, validation_generator = [
        tf.data.Dataset.from_generator(
            functools.partial(
                recognizer.get_batch_generator,
                image_generator=image_generator,
                batch_size=args.batch_size,
            ),
            output_types=((tf.float32, tf.int64, tf.float64, tf.int64), tf.float64),
            output_shapes=(
                (
                    tf.TensorShape([None, 31, 200, 1]),
                    tf.TensorShape([None, recognizer.training_model.input_shape[1][1]]),
                    tf.TensorShape([None, 1]),
                    tf.TensorShape([None, 1]),
                ),
                tf.TensorShape([None, 1]),
            ),
        )
        for image_generator in [training_image_generator, validation_image_generator]
    ]
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=10, restore_best_weights=False
        ),
        tf.keras.callbacks.ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(csv_path),
    ]
    recognizer.training_model.fit(
        x=training_generator,
        steps_per_epoch=training_steps,
        validation_steps=validation_steps,
        validation_data=validation_generator,
        callbacks=callbacks,
        epochs=1000,
    )
