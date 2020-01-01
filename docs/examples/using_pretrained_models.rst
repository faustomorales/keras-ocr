Using pretrained models
=======================

The below example shows how to use the pretrained models.

.. code-block:: python

    import matplotlib.pyplot as plt

    import keras_ocr

    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    pipeline = keras_ocr.pipeline.Pipeline()

    image = keras_ocr.tools.read('tests/test_image.jpg')

    # Predictions is a list of (text, box) tuples.
    predictions = pipeline.recognize(image=image)

    # Plot the results.
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 10))
    canvas = keras_ocr.detection.drawBoxes(image, [box for text, box in predictions])
    ax1.imshow(image)
    ax2.imshow(canvas)

    for text, box in predictions:
        ax2.annotate(s=text, xy=box[0], xytext=box[0] - 50, arrowprops={'arrowstyle': '->'})

.. image:: ../../tests/test_image_labeled.jpg
   :width: 512