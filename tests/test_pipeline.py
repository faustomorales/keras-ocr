import keras_ocr


def test_pipeline():
    pipeline = keras_ocr.pipeline.Pipeline()

    image = keras_ocr.tools.read('tests/test_image.jpg')

    # Predictions is a list of (text, box) tuples.
    predictions = pipeline.recognize(images=[image])[0]

    assert len(predictions) == 1
    assert predictions[0][0] == 'eventdock'
