# pylint: disable=invalid-name,too-many-branches,too-many-statements
import os
import typing
import hashlib
import urllib.request
import urllib.parse

import cv2
import numpy as np


def read(filepath_or_image: typing.Union[str, np.ndarray]):
    """Read an image from disk.

    Args:
        filepath: The path to the image

    Returns:
        The new image
    """
    if not isinstance(filepath_or_image, str):
        return filepath_or_image
    image = cv2.imread(filepath_or_image)
    if image is None:
        raise Exception(f'Could not read {filepath_or_image}.')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def warpBox(image, box, target_height, target_width, margin=0):
    """Warp a box given by a set of four points into
    a specific shape."""
    _, _, w, h = cv2.boundingRect(box)
    scale = min(target_width / w, target_height / h)
    M = cv2.getPerspectiveTransform(src=box,
                                    dst=np.array([[margin, margin], [scale * w - margin, margin],
                                                  [scale * w - margin, scale * h - margin],
                                                  [margin, scale * h - margin]]).astype('float32'))
    crop = cv2.warpPerspective(image, M, dsize=(int(scale * w), int(scale * h)))
    full = np.zeros((target_height, target_width, 3)).astype('uint8')
    full[:crop.shape[0], :crop.shape[1]] = crop
    return full


def fit(image, width: int, height: int, cval: int = 255, mode='letterbox'):
    """Obtain a new image, fit to the specified size.

    Args:
        image: The input image
        width: The new width
        height: The new height
        cval: The constant value to use to fill the remaining areas of
            the image

    Returns:
        The new image
    """
    if width == image.shape[1] and height == image.shape[0]:
        return image
    if mode == 'letterbox':
        if width / image.shape[1] <= height / image.shape[0]:
            resize_width = width
            resize_height = (width / image.shape[1]) * image.shape[0]
        else:
            resize_height = height
            resize_width = (height / image.shape[0]) * image.shape[1]
        resize_width, resize_height = map(int, [resize_width, resize_height])
        fitted = np.zeros((height, width, 3), dtype='uint8') + cval
        image = cv2.resize(image, dsize=(resize_width, resize_height))
        fitted[:image.shape[0], :image.shape[1]] = image[:height, :width]
    elif mode == 'crop':
        if width / image.shape[1] >= height / image.shape[0]:
            resize_width = width
            resize_height = (width / image.shape[1]) * image.shape[0]
        else:
            resize_height = height
            resize_width = (height / image.shape[0]) * image.shape[1]
        resize_width, resize_height = map(int, [resize_width, resize_height])
        image = cv2.resize(image, dsize=(resize_width, resize_height))
        fitted = image[:height, :width]
    else:
        raise NotImplementedError
    return fitted


def read_and_fit(filepath_or_array: typing.Union[str, np.ndarray],
                 width: int,
                 height: int,
                 cval: int = 255,
                 mode='letterbox'):
    """Read an image from disk and fit to the specified size.

    Args:
        filepath: The path to the image or numpy array of shape HxWx3
        width: The new width
        height: The new height
        cval: The constant value to use to fill the remaining areas of
            the image
        mode: The mode to pass to "fit" (crop or letterbox)

    Returns:
        The new image
    """
    image = read(filepath_or_array) if isinstance(filepath_or_array, str) else filepath_or_array
    image = fit(image=image, width=width, height=height, cval=cval, mode=mode)
    return image


def sha256sum(filename):
    """Compute the sha256 hash for a file."""
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def download_and_verify(url, sha256=None, cache_dir=None, verbose=True):
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.path.join('~', '.keras-ocr'))
    filename = os.path.basename(urllib.parse.urlparse(url).path)
    filepath = os.path.join(cache_dir, filename)
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    if verbose:
        print('Looking for ' + filepath)
    if not os.path.isfile(filepath) or (sha256 and sha256sum(filepath) != sha256):
        if verbose:
            print('Downloading ' + filepath)
        urllib.request.urlretrieve(url, filepath)
    assert sha256 is None or sha256 == sha256sum(filepath), 'Error occurred verifying sha256.'
    return filepath
