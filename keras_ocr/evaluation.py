# pylint: disable=invalid-name,too-many-locals
import copy
import warnings

import editdistance
import numpy as np
import pyclipper
import cv2


# Adapted from https://github.com/andreasveit/coco-text/blob/master/coco_evaluation.py
def iou_score(box1, box2):
    """Returns the Intersection-over-Union score, defined as the area of
    the intersection divided by the intersection over the union of
    the two bounding boxes. This measure is symmetric.

    Args:
        box1: The coordinates for box 1 as a list of (x, y) coordinates
        box2: The coordinates for box 2 in same format as box1.
    """
    if len(box1) == 2:
        x1, y1 = box1[0]
        x2, y2 = box1[1]
        box1 = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if len(box2) == 2:
        x1, y1 = box2[0]
        x2, y2 = box2[1]
        box2 = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if any(cv2.contourArea(np.int32(box)[:, np.newaxis, :]) == 0 for box in [box1, box2]):
        warnings.warn('A box with zero area was detected.')
        return 0
    pc = pyclipper.Pyclipper()
    pc.AddPath(np.int32(box1), pyclipper.PT_SUBJECT, closed=True)
    pc.AddPath(np.int32(box2), pyclipper.PT_CLIP, closed=True)
    intersection_solutions = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD,
                                        pyclipper.PFT_EVENODD)
    union_solutions = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
    union = sum(cv2.contourArea(np.int32(points)[:, np.newaxis, :]) for points in union_solutions)
    intersection = sum(
        cv2.contourArea(np.int32(points)[:, np.newaxis, :]) for points in intersection_solutions)
    return intersection / union


def score(true, pred, iou_threshold=0.5, similarity_threshold=0.5, translator=None):
    """
    Args:
        true: The ground truth boxes provided as a dictionary of {image_id: annotations}
            mappings. `annotations` should be lists of dicts with a `text` and `vertices` key.
            `vertices` should be a list of (x, y) coordinates. Optionally, an "ignore" key can be
            added to indicate that detecting an annotation should neither count as a false positive
            nor should failure to detect it count as a false negative.
        pred: The predicted boxes in the same format as `true`.
        iou_threshold: The minimum IoU to qualify a box as a match.
        similarity_threshold: The minimum texg similarity required to qualify
            a text string as a match.
        translator: A translator acceptable by `str.translate`. Used to
            modify ground truth / predicted strings. For example,
            `str.maketrans(string.ascii_uppercase, string.ascii_lowercase,
            string.punctuation)` would yield a translator that changes all
            strings to lowercase and removes punctuation.

    Returns:
        A results dictionary reporting false positives, false negatives, true positives
        and near matches (IoU > iou_threshold but similarity < similarity_threshold) along
        with the compute precision and recall.
    """
    true_ids = sorted(true)
    pred_ids = sorted(pred)
    assert all(true_id == pred_id for true_id, pred_id in zip(
        true_ids, pred_ids)), 'true and pred dictionaries must have the same keys'
    results = {
        'true_positives': [],
        'false_positives': [],
        'near_true_positives': [],
        'false_negatives': []
    }
    for image_id in true_ids:
        true_anns = true[image_id]
        pred_anns = copy.deepcopy(pred[image_id])
        pred_matched = set()
        for true_index, true_ann in enumerate(true_anns):
            match = None
            for pred_index, pred_ann in enumerate(pred_anns):
                iou = iou_score(true_ann['vertices'], pred_ann['vertices'])
                if iou >= iou_threshold:
                    match = {'true_idx': true_index, 'pred_idx': pred_index, 'image_id': image_id}
                    pred_matched.add(pred_index)
                    true_text = true_ann['text']
                    pred_text = pred_ann['text']
                    if true_ann.get('ignore', False):
                        # We recorded that this prediction matched something,
                        # so it won't be a false positive. But we're also ignoring
                        # this ground truth label so we won't count it as a true
                        # positive or a near true positive.
                        continue
                    if translator is not None:
                        true_text = true_text.translate(translator)
                        pred_text = pred_text.translate(translator)
                    edit_distance_norm = max(len(true_text), len(pred_text))
                    if edit_distance_norm == 0:
                        similarity = 1
                    else:
                        similarity = 1 - (editdistance.eval(true_text, pred_text) /
                                          max(len(true_text), len(pred_text)))
                    if similarity >= similarity_threshold:
                        results['true_positives'].append(match)
                    else:
                        results['near_true_positives'].append(match)
            if match is None and not true_ann.get('ignore', False):
                results['false_negatives'].append({'image_id': image_id, 'true_idx': true_index})
        results['false_positives'].extend({
            'pred_index': pred_index,
            'image_id': image_id
        } for pred_index, _ in enumerate(pred_anns) if pred_index not in pred_matched)
    fns = len(results['false_negatives'])
    fps = len(results['false_positives'])
    tps = len(
        set((true_positive['image_id'], true_positive['true_idx'])
            for true_positive in results['true_positives']))
    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    return results, (precision, recall)
