Using pretrained models
=======================

The below example shows how to use the pretrained models.

.. code-block:: python

    import matplotlib.pyplot as plt

    import keras_ocr

    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    pipeline = keras_ocr.pipeline.Pipeline()

    # Get a set of three example images
    images = [
        keras_ocr.tools.read(url) for url in [
            'https://upload.wikimedia.org/wikipedia/commons/b/bd/Army_Reserves_Recruitment_Banner_MOD_45156284.jpg',
            'https://upload.wikimedia.org/wikipedia/commons/b/b4/EUBanana-500x112.jpg'
        ]
    ]

    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize(images)

    # Plot the predictions
    fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
    for ax, image, predictions in zip(axs, images, prediction_groups):
        keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)

.. image:: ../_static/readme_labeled.jpg
   :width: 768
