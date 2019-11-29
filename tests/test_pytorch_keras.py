# pylint: disable=no-member,invalid-name
import os

import numpy as np
import pytest
import torch

import keras_ocr

torch_weights_available = os.path.isfile(
    os.path.expanduser(
        os.path.join('~', '.keras-ocr',
                     'https://storage.googleapis.com/keras-ocr/craft_mlt_25k.pth')))
keras_weights_available = os.path.isfile(
    os.path.expanduser(
        os.path.join('~', '.keras-ocr',
                     'https://storage.googleapis.com/keras-ocr/craft_mlt_25k.h5')))


@pytest.mark.skipif(not keras_weights_available and torch_weights_available,
                    reason="CRAFT weights required.")
def test_pytorch_identical_output():
    weights_path_torch = keras_ocr.tools.download_and_verify(
        url='https://storage.googleapis.com/keras-ocr/craft_mlt_25k.pth',
        sha256='4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17')
    weights_path_keras = keras_ocr.tools.download_and_verify(
        url='https://storage.googleapis.com/keras-ocr/craft_mlt_25k.h5',
        sha256='7283ce2ff05a0617e9740c316175ff3bacdd7215dbdf1a726890d5099431f899')

    model_keras = keras_ocr.detection.build_keras_model(weights_path=weights_path_keras)
    model_pytorch = keras_ocr.detection.build_torch_model(weights_path=weights_path_torch)
    image = keras_ocr.tools.read('tests/test_image.jpg')
    X = keras_ocr.detection.compute_input(image)[np.newaxis, ]
    y_pred_keras = model_keras.predict(X)
    y_pred_torch = model_pytorch.forward(torch.from_numpy(X.transpose(0, 3, 1,
                                                                      2)))[0].detach().numpy()
    np.testing.assert_almost_equal(y_pred_keras, y_pred_torch, decimal=4)
