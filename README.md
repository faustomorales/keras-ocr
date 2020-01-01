# keras-ocr
This is a slightly polished and packaged version of the [Keras CRNN implementation](https://github.com/kurapan/CRNN) and the published [CRAFT text detection model](https://github.com/clovaai/CRAFT-pytorch). It provides a high level API for training a text detection and OCR pipeline.

Please see [the documentation](https://keras-ocr.readthedocs.io/) for more examples, including for training a custom model.

## Getting Started

### Installation
```bash
# To install from master
pip install git+https://github.com/faustomorales/keras-ocr.git#egg=keras-ocr

# To install from PyPi
pip install keras-ocr
```

### Using

#### Using pretrained text detection and recognition models
The package ships with an easy-to-use implementation of the CRAFT text detection model from [this repository](https://github.com/clovaai/CRAFT-pytorch) and the CRNN recognition model from [this repository](https://github.com/kurapan/CRNN).

```python
import keras_ocr

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Predictions is a list of (string, box) tuples.
predictions = pipeline.recognize(image='tests/test_image.jpg')
```

![example of labeled image](https://raw.githubusercontent.com/faustomorales/keras-ocr/master/tests/test_image_labeled.jpg)