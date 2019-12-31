Fine-tuning recognizer
======================

This example shows how to fine-tune the recognizer using an existing dataset. In this case,
we will use the "Born Digital" dataset from https://rrc.cvc.uab.es/?ch=1&com=downloads

First, we download our dataset. Below we get both the training and test datasets, but
we only use the training dataset. The training dataset consists of a single folder
containing images, each of which has a single word in it. The labels are in a text
file called :code:`gt.txt`.

An interactive version of this example on Google Colab is provided `here
<https://colab.research.google.com/drive/19dGKong-LraUG3wYlJuPCquemJ13NN8R>`_.

.. code-block:: python

    import zipfile
    import random
    import string
    import math
    import itertools
    import os

    import numpy as np
    import imgaug
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import sklearn.model_selection

    import keras_ocr

    assert tf.test.is_gpu_available()

    training_dir = './training'
    test_dir = './test'
    if not os.path.isdir(training_dir):
        training_zip_path = keras_ocr.tools.download_and_verify(
            url='https://storage.googleapis.com/keras-ocr/Challenge1_Training_Task3_Images_GT.zip',
            cache_dir=data_dir,
            sha256='8ede0639f5a8031d584afd98cee893d1c5275d7f17863afc2cba24b13c932b07'
        )
        with zipfile.ZipFile(training_zip_path) as zfile:
            zfile.extractall(training_dir) 
    if not os.path.isdir(test_dir):
        test_zip_path = keras_ocr.tools.download_and_verify(
            url='https://storage.googleapis.com/keras-ocr/Challenge1_Test_Task3_Images.zip',
            cache_dir=data_dir,
            sha256='8f781b0140fd0bac3750530f0924bce5db3341fd314a2fcbe9e0b6ca409a77f0'
        )
        with zipfile.ZipFile(test_zip_path) as zfile:
            zfile.extractall(test_dir)
    test_gt_path = keras_ocr.tools.download_and_verify(
        url='https://storage.googleapis.com/keras-ocr/Challenge1_Test_Task3_GT.txt',
        cache_dir=data_dir,
        sha256='fce7f1228b7c4c26a59f13f562085148acf063d6690ce51afc395e0a1aabf8be'
    )

We next build our recognizer, using the default options to get a pretrained model.

.. code-block:: python

    recognizer = keras_ocr.recognition.Recognizer(
        width=200,
        height=31,
        optimizer='RMSprop',
        include_top=True
    )

We need to convert our dataset into the format that :code:`keras-ocr` requires. To 
do that, we have the following, which includes support for an augmenter to
generate synthetically altered samples. Note that this code is set up to skip
any characters that are not in the recognizer alphabet and that all labels
are first converted to lowercase.

.. code-block:: python

    batch_size = 8
    augmenter = imgaug.augmenters.Sequential([
        imgaug.augmenters.GammaContrast(gamma=(0.25, 3.0)),
    ])

    def read_labels_file(labels_filepath, image_folder):
        """Read a labels file and return (filepath, label) tuples.
        
        Args:
            labels_filepath: Path to labels file
            image_folder: Path to folder containing images
        """
        with open(labels_filepath, encoding='utf-8-sig') as f:
            labels = [l.strip().split(',') for l in f.readlines()]
            labels = [(os.path.join(image_folder, segments[0]), ','.join(segments[1:]).strip()[1:-1]) for segments in labels]
        return labels

    def image_generator(
        labels,
        target_height,
        target_width,
        alphabet,
        augmenter=None,
    ):
        """Generate augmented (image, text) tuples from a list
        of labels with the option of limiting to images with
        a text length below some threshold."""
        labels = labels.copy()
        for index in itertools.cycle(range(len(labels))):
            if index == 0:
                random.shuffle(labels)
            filepath, text = labels[index]
            text = ''.join([c for c in text.lower() if c in alphabet])
            if not text:
                continue
            image = keras_ocr.tools.read_and_fit(
                filepath_or_array=filepath,
                width=target_width,
                height=target_height,
                cval=np.random.randint(low=0, high=255, size=3).astype('uint8')
            )
            if augmenter:
            image = augmenter.augment_image(image)
            yield (image, text)

    test_labels = read_labels_file(
        labels_filepath=test_gt_path,
        image_folder=test_dir
    )
    train_labels = read_labels_file(
        labels_filepath=os.path.join(training_dir, 'gt.txt'),
        image_folder=training_dir
    )
    train_labels, validation_labels = sklearn.model_selection.train_test_split(train_labels, test_size=0.2, random_state=42)
    (training_image_gen, training_steps), (validation_image_gen, validation_steps) = [
        (
            image_generator(
                labels=labels,
                target_height=recognizer.model.input_shape[1],
                target_width=recognizer.model.input_shape[2],
                alphabet=recognizer.alphabet,
                augmenter=augmenter
            ),
            len(labels) // batch_size
        ) for labels, augmenter in [(train_labels, augmenter), (validation_labels, None)]     
    ]
    training_gen, validation_gen = [
        recognizer.get_batch_generator(
            image_generator=image_generator,
            batch_size=batch_size
        )
        for image_generator in [training_image_gen, validation_image_gen]
    ]

As a sanity check, we show one of the samples.

.. code-block:: python

    image, text = next(training_image_gen)
    print('text:', text)
    plt.imshow(image)

.. image:: ../_static/borndigital1.jpg
   :width: 256

Now we can run training.

.. code-block:: python

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=False),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(data_dir, 'recognizer_borndigital.h5'), monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(data_dir, 'recognizer_borndigital.csv'))
    ]
    recognizer.training_model.fit_generator(
        generator=training_gen,
        steps_per_epoch=training_steps,
        validation_steps=validation_steps,
        validation_data=validation_gen,
        callbacks=callbacks,
        epochs=1000,
    )

Finally, run inference on a test sample.

.. code-block:: python

    image_filepath, actual = test_labels[1]
    predicted = recognizer.recognize(image_filepath)
    print(f'Predicted: {predicted}, Actual: {actual}')
    _ = plt.imshow(keras_ocr.tools.read(image_filepath))

.. image:: ../_static/borndigital2.jpg
   :width: 256