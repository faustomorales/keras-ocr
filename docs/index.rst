keras-ocr
=========

:code:`keras-ocr` provides out-of-the-box OCR models and an end-to-end training pipeline to build new OCR models.
Please see the :doc:`examples <examples/index>` for more information.

Installation
************

:code:`keras-ocr` supports Python >= 3.6 and TensorFlow >= 2.0.0.

.. code-block:: bash

    # To install from master
    pip install git+https://github.com/faustomorales/keras-ocr.git#egg=keras-ocr

    # To install from PyPi
    pip install keras-ocr

Troubleshooting
***************

- *This package is installing* :code:`opencv-python-headless` *but I would prefer a different* :code:`opencv` *flavor.* This is due to `aleju/imgaug#473 <https://github.com/aleju/imgaug/issues/473>`_. You can uninstall the unwanted OpenCV flavor after installing :code:`keras-ocr`. We apologize for the inconvenience.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples/index
   api

