import numpy as np
from keras_ocr import tools


def test_fix_line():
    baseline = np.array([[10, 10], [0, 0], [0, 10], [10, 0]])
    vertical_line = [
        (baseline + [0, 0], "a"),
        (baseline + [0, 30], "d"),
        (baseline + [0, 20], "c"),
        (baseline + [0, 10], "b"),
    ]
    horizontal_line = [
        (baseline + [0, 0], "a"),
        (baseline + [30, 0], "d"),
        (baseline + [20, 0], "c"),
        (baseline + [10, 0], "b"),
    ]
    vertical_line_fixed = tools.fix_line(vertical_line)
    horizontal_line_fixed = tools.fix_line(horizontal_line)
    assert horizontal_line_fixed[1] == "horizontal"
    assert vertical_line_fixed[1] == "vertical"
    assert "".join([character for _, character in vertical_line_fixed[0]]) == "abcd"
    assert "".join([character for _, character in horizontal_line_fixed[0]]) == "abcd"
