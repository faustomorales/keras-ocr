import keras_ocr


def test_iou_score():
    box1 = [(0, 0), (100, 0), (100, 100), (0, 100)]
    box2 = [(50, 50), (100, 50), (100, 100), (50, 100)]
    assert keras_ocr.evaluation.iou_score(box1, box2) == 0.25

    box2 = [(100, 100), (200, 100), (200, 200), (100, 200)]
    assert keras_ocr.evaluation.iou_score(box1, box2) == 0.0
