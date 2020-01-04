# pylint: disable=too-few-public-methods
import cv2

from . import detection, recognition, tools


class Pipeline:
    """A wrapper for a combination of detector and recognizer.

    Args:
        detector: The detector to use
        recognizer: The recognizer to use
    """
    def __init__(self, detector=None, recognizer=None, scale=2):
        if detector is None:
            detector = detection.Detector()
        if recognizer is None:
            recognizer = recognition.Recognizer()
        self.scale = scale
        self.detector = detector
        self.recognizer = recognizer

    def recognize(self, image, detection_kwargs=None, recognition_kwargs=None):
        """Run the pipeline on a single image.

        Args:
            image: The image to parse (can be an actual image or a fielpath)
            detection_kwargs: Arguments to pass to the detector call
            recognition_kwargs: Arguments to pass to the recognizer call

        Returns:
            A list of (text, box) tuples.
        """
        image = tools.read(image)

        if self.scale != 1:
            image = cv2.resize(image,
                               dsize=(image.shape[1] * self.scale, image.shape[0] * self.scale))

        if detection_kwargs is None:
            detection_kwargs = {}
        if recognition_kwargs is None:
            recognition_kwargs = {}

        # Predictions is a list of (string, box) tuples.
        boxes = self.detector.detect(images=[image], **detection_kwargs)[0]
        predictions = self.recognizer.recognize_from_boxes(image=image,
                                                           boxes=boxes,
                                                           **recognition_kwargs)
        if self.scale != 1:
            predictions = tools.adjust_boxes(boxes=predictions,
                                             boxes_format='predictions',
                                             scale=1 / self.scale)
        return predictions
