# pylint: disable=invalid-name,too-many-branches,too-many-statements
import os
import io
import typing
import hashlib
import urllib.request
import urllib.parse

import cv2
import numpy as np
import validators


def read(filepath_or_buffer: typing.Union[str, io.BytesIO]):
    """Read a file into an image object

    Args:
        filepath_or_buffer: The path to the file, a URL, or any object
            with a `read` method (such as `io.BytesIO`)
    """
    if isinstance(filepath_or_buffer, np.ndarray):
        return filepath_or_buffer
    if hasattr(filepath_or_buffer, 'read'):
        image = np.asarray(bytearray(filepath_or_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    elif isinstance(filepath_or_buffer, str):
        if validators.url(filepath_or_buffer):
            return read(urllib.request.urlopen(filepath_or_buffer))
        assert os.path.isfile(filepath_or_buffer), \
            'Could not find image at path: ' + filepath_or_buffer
        image = cv2.imread(filepath_or_buffer)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def warpBox(image, box, target_height, target_width, margin=0):
    """Warp a boxed region in an image given by a set of four points into
    a rectangle with a specified width and height. Useful for taking crops
    of distorted or rotated text.

    Args:
        image: The image from which to take the box
        box: A list of four points starting in the top left
            corner and moving clockwise.
        target_height: The height of the output rectangle
        target_width: The width of the output rectangle
    """
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


# pylint: disable=too-many-arguments
def fit(image, width: int, height: int, cval: int = 255, mode='letterbox', return_scale=False):
    """Obtain a new image, fit to the specified size.

    Args:
        image: The input image
        width: The new width
        height: The new height
        cval: The constant value to use to fill the remaining areas of
            the image
        return_scale: Whether to return the scale used for the image

    Returns:
        The new image
    """
    fitted = None
    if width == image.shape[1] and height == image.shape[0]:
        fitted = image
        scale = 1
    elif width / image.shape[1] >= height / image.shape[0]:
        scale = width / image.shape[1]
        resize_width = width
        resize_height = (width / image.shape[1]) * image.shape[0]
    else:
        scale = height / image.shape[0]
        resize_height = height
        resize_width = scale * image.shape[1]
    if fitted is None:
        resize_width, resize_height = map(int, [resize_width, resize_height])
        if mode == 'letterbox':
            fitted = np.zeros((height, width, 3), dtype='uint8') + cval
            image = cv2.resize(image, dsize=(resize_width, resize_height))
            fitted[:image.shape[0], :image.shape[1]] = image[:height, :width]
        elif mode == 'crop':
            image = cv2.resize(image, dsize=(resize_width, resize_height))
            fitted = image[:height, :width]
        else:
            raise NotImplementedError(f'Unknown mode: {mode}')
    if not return_scale:
        return fitted
    return fitted, scale


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


def download_and_verify(url, sha256=None, cache_dir=None, verbose=True, filename=None):
    """Download a file to a cache directory and verify it with a sha256
    hash.

    Args:
        url: The file to download
        sha256: The sha256 hash to check. If the file already exists and the hash
            matches, we don't download it again.
        cache_dir: The directory in which to cache the file. The default is
            `~/.keras-ocr`.
        verbose: Whether to log progress
        filename: The filename to use for the file. By default, the filename is
            derived from the URL.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.path.join('~', '.keras-ocr'))
    if filename is None:
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
