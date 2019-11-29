Complete end-to-end training
============================

You may wish to train your own end-to-end OCR pipeline. Here's an example for how you might do it. Note that the image generator has many options not documented here (such as adding backgrounds and image augmentation). Check the documentation for the `keras_ocr.tools.get_image_generator` function for more details.

Please note that, right now, we use a very simple training mechanism for the text detector which seems to work but does not match the method used in the original implementation.

This is work-in-progress: Please see the Google Colab notebook for an example.