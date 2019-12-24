Complete end-to-end training
============================

You may wish to train your own end-to-end OCR pipeline. Here's an example for
how you might do it. Note that the image generator has many options not
documented here (such as adding backgrounds and image augmentation). Check
the documentation for the `keras_ocr.tools.get_image_generator` function for more details.

Please note that, right now, we use a very simple training mechanism for the
text detector which seems to work but does not match the method used in the
original implementation.

Generating synthetic data
*************************

First, we define the alphabet that encompasses all characters we want our model to be able to detect and recognize. Below we designate our alphabet as the numbers 0-9, upper- and lower-case letters, and a few puncuation marks. For the recognizer, we will actually only predict lowercase letters because we know some fonts print lower- and upper-case characters with the same glyph.

In order to train on synthetic data, we require a set of fonts and backgrounds. :code:`keras-ocr` includes a set of both of these which have been downloaded from Google Fonts and Wikimedia. The code to generate both of these sets is available in the repository under :code:`scripts/create_fonts_and_backgrounds.py`.

The fonts cover different languages which may have non-overlapping characters. :code:`keras-ocr` supplies a function (:code:`font_supports_alphabet`) to verify that a font includes the characters in an alphabet. We filter to only these fonts. We also exclude any fonts that are marked as `thin` in the filename because those tend to be difficult to render in a legible manner.

The backgrounds folder contains about just over 1,000 image backgrounds.

.. code-block:: python

    import zipfile
    import datetime
    import string
    import glob
    import math
    import os

    import tqdm
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import sklearn.model_selection

    import keras_ocr

    assert tf.test.is_gpu_available(), 'No GPU is available.'

    if not os.path.isdir('fonts'):
    fonts_zip_path = keras_ocr.tools.download_and_verify(
        url='https://storage.googleapis.com/keras-ocr/fonts.zip',
        sha256='d4d90c27a9bc4bf8fff1d2c0a00cfb174c7d5d10f60ed29d5f149ef04d45b700',
        cache_dir=data_dir
    )
    with zipfile.ZipFile(fonts_zip_path) as zfile:
        zfile.extractall('./fonts')
    if not os.path.isdir('backgrounds'):
    backgrounds_zip_path = keras_ocr.tools.download_and_verify(
        url='https://storage.googleapis.com/keras-ocr/backgrounds.zip',
        sha256='f263ed0d55de303185cc0f93e9fcb0b13104d68ed71af7aaaa8e8c91389db471',
        cache_dir=data_dir
    )
    with zipfile.ZipFile(backgrounds_zip_path) as zfile:
        zfile.extractall('./backgrounds')
    
    alphabet = string.digits + string.ascii_letters + '!?. '
    recognizer_alphabet = ''.join(sorted(set(alphabet.lower())))

    fonts = [
        filepath for filepath in tqdm.tqdm(glob.glob('fonts/**/*.ttf'))
        if (
            (not any(keyword in filepath.lower() for keyword in ['thin', 'light'])) and
            keras_ocr.tools.font_supports_alphabet(filepath=filepath, alphabet=alphabet)
        )
    ]

    backgrounds = glob.glob('backgrounds/*.jpg')

With a set of fonts, backgrounds, and alphabet, we now build our data generators.

In order to create images, we need random strings. :code:`keras-ocr` has a simple method for this for English, but anything that generates strings of characters in your selected alphabet will do!

The image generator generates `(image, sentence, lines)` tuples where `image` is a HxWx3 image, `sentence` is a string using only letters from the selected alphabet, and `lines` is a list of lines of text in the image where each line is itself a list of tuples of the form :code:`((x1, y1), (x2, y2), (x3, y3), (x4, y4), c)`. `c` is the character in the line and :code:`(x1, y2), (x2, y2), (x3, y3),
(x4, y4)` define the bounding coordinates in clockwise order starting from the top left. You can replace this with your own generator, just be sure to match that function signature.

We split our generators into train, validation, and test by separating the fonts and backgrounds used in each.

.. code-block:: python

    text_generator = keras_ocr.tools.get_text_generator(alphabet=alphabet)
    print('The first generated text is:', next(text_generator))

    def get_train_val_test_split(arr):
    train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
    val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
    return train, val, test

    background_splits = get_train_val_test_split(backgrounds)
    font_splits = get_train_val_test_split(fonts)

    image_generators = [
        keras_ocr.tools.get_image_generator(
        height=640,
        width=640,
        text_generator=text_generator,
        font_groups={
            alphabet: current_fonts
        },
        backgrounds=current_backgrounds,
        font_size=(60, 120),
        margin=50,
        rotationX=(-0.05, 0.05),
        rotationY=(-0.05, 0.05),
        rotationZ=(-15, 15),
        )  for current_fonts, current_backgrounds in zip(
            font_splits,
            background_splits
        )
    ]

    # See what the first validation image looks like.
    image, text, lines = next(image_generators[1])
    print('The first generated validation image (below) contains:', text)
    plt.imshow(image)

.. image:: ../_static/generated1.jpg
   :width: 256

Build base detector and recognizer models
*****************************************

Here we build our detector and recognizer models. For both, we'll start with pretrained models. Note that for the recognizer, we freeze the weights in the backbone (all the layers except for the final classification layer).

.. code-block:: python

    detector = keras_ocr.detection.Detector(weights='clovaai_general')
    recognizer = keras_ocr.recognition.Recognizer(
        width=200,
        height=31,
        stn=True,
        alphabet=recognizer_alphabet,
        weights='kurapan',
        optimizer='RMSprop',
        include_top=False
    )
    for layer in recognizer.backbone.layers:
    layer.trainable = False

We are now ready to train our text detector. Below we use some simple defaults.

- Run training until we have no improvement on the validation set for 5 epochs.
- Save the best weights.
- For each epoch, iterate over all backgrounds one time.

The `detector` object has a `get_batch_generator` method which converts the `image_generator` (which returns images and associated annotations) into a `batch_generator` that returns `X, y` pairs for training with `fit_generator`.

If training on Colab and it assigns you a K80, you can only use batch size 1. But if you get a T4 or P100, you can use larger batch sizes.

.. code-block:: python

    detector_batch_size = 1
    detector_basepath = os.path.join(data_dir, f'detector_{datetime.datetime.now().isoformat()}')
    detection_train_generator, detection_val_generator, detection_test_generator = [
        detector.get_batch_generator(
        image_generator=image_generator,
        batch_size=detector_batch_size
        ) for image_generator in image_generators
    ]
    detector.model.fit_generator(
        generator=detection_train_generator,
        steps_per_epoch=math.ceil(len(background_splits[0]) / detector_batch_size),
        epochs=1000,
        workers=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
            tf.keras.callbacks.CSVLogger(f'{detector_basepath}.csv'),
            tf.keras.callbacks.ModelCheckpoint(filepath=f'{detector_basepath}.h5')
        ],
        validation_data=detection_val_generator,
        validation_steps=math.ceil(len(background_splits[1]) / detector_batch_size)
    )

After training the text detector, we train the recognizer. Note that the recognizer expects images to already be cropped to single lines of text. :code:`keras-ocr` provides a convenience method for converting our existing generator into a single-line generator. So we perform that conversion.

.. code-block:: python

    max_length = 10
    recognition_image_generators = [keras_ocr.tools.convert_multiline_generator_to_single_line(
        multiline_generator=image_generator,
        max_string_length=min(recognizer.training_model.input_shape[1][1], max_length),
        target_width=recognizer.model.input_shape[2],
        target_height=recognizer.model.input_shape[1],
        margin=1
    ) for image_generator in image_generators]

    # See what the first validation image for recognition training looks like.
    image, text, lines = next(recognition_image_generators[1])
    print('This image contains:', text)
    plt.imshow(image)

.. image:: ../_static/generated2.jpg
   :width: 384

Just like the :code:`detector`, the :code:`recognizer` has a method for converting the image generator into a :code:`batch_generator` that Keras' :code:`fit_generator` can use.

We use the same callbacks for early stopping and logging as before.

.. code-block:: python

    recognition_batch_size = 8
    recognizer_basepath = os.path.join(data_dir, f'recognizer_{datetime.datetime.now().isoformat()}')
    recognition_train_generator, recognition_val_generator, recogntion_test_generator = [
        recognizer.get_batch_generator(
        image_generator=image_generator,
        batch_size=recognition_batch_size,
        lowercase=True
        ) for image_generator in recognition_image_generators
    ]
    recognizer.training_model.fit_generator(
        generator=recognition_train_generator,
        epochs=1000,
        steps_per_epoch=math.ceil(len(background_splits[0]) / recognition_batch_size),
        callbacks=[
        tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=25),
        tf.keras.callbacks.CSVLogger(f'{recognizer_basepath}.csv', append=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=f'{recognizer_basepath}.h5')
        ],
        validation_data=recognition_val_generator,
        validation_steps=math.ceil(len(background_splits[1]) / recognition_batch_size),
        workers=0
    )

Once training is done, you can use recognize to extract text.

.. code-block:: python

    image, text, lines = next(image_generators[0])
    boxes = detector.detect(images=[image])[0]
    drawn = keras_ocr.detection.drawBoxes(image=image, boxes=boxes)
    predictions = recognizer.recognize_from_boxes(boxes=boxes, image=image)
    print(text, [text for text, box in predictions])
    plt.imshow(drawn)

.. image:: ../_static/predicted1.jpg
   :width: 512