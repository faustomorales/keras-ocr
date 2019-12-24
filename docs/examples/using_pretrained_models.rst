Using pretrained models
=======================

The below example shows how to use the pretrained models.

.. code-block:: python

    import matplotlib.pyplot as plt

    import keras_ocr

    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    detector = keras_ocr.detection.Detector()
    recognizer = keras_ocr.recognition.Recognizer()

    image = keras_ocr.tools.read('tests/test_image.jpg')

    # Boxes will be an Nx4x2 array of box quadrangles
    # where N is the number of detected text boxes.
    # Predictions is a list of (string, box) tuples.
    boxes = detector.detect(images=[image])[0]
    predictions = recognizer.recognize_from_boxes(image=image, boxes=boxes)

    # Plot the results.
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 10))
    canvas = keras_ocr.detection.drawBoxes(image, boxes)
    ax1.imshow(image)
    ax2.imshow(canvas)

    for text, box in predictions:
        ax2.annotate(s=text, xy=box[0], xytext=box[0] - 50, arrowprops={'arrowstyle': '->'})

.. image:: ../../tests/test_image_labeled.jpg
   :width: 512