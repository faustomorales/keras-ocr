API
===

Core Detector and Recognizer
****************************

The detector and recognizer classes are the core of the
package. They provide wrappers for the underlying Keras models.

.. autoclass:: keras_ocr.detection.Detector
        :members:

.. autoclass:: keras_ocr.recognition.Recognizer
        :members:

Data Generation
***************

The :code:`data_generation` module contains the functions
for generating synthetic data.

.. automodule:: keras_ocr.data_generation
        :members:

Tools
*****

The :code:`tools` module primarily contains convenience functions for
reading images and downloading data.

.. automodule:: keras_ocr.tools
        :members:

Datasets
********

The :code:`datasets` module contains functions for using data
from public datasets. See the :doc:`fine-tuning detector <examples/fine_tuning_detector>`
and :doc:`fine-tuning recognizer <examples/fine_tuning_recognizer>` examples. 

.. automodule:: keras_ocr.datasets
        :members:

