# pylint: disable=invalid-name,too-many-locals,too-many-arguments,too-many-branches,too-many-statements,stop-iteration-return
import os
import typing
import hashlib
import urllib.request
import urllib.parse

import cv2
try:
    import cairocffi
except ImportError:
    cairocffi = None
import numpy as np
import essential_generators


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


def warpBox(image, box, target_height, target_width):
    """Warp a box given by a set of four points into
    a specific shape."""
    _, _, w, h = cv2.boundingRect(box)
    scale = min(target_width / w, target_height / h)
    M = cv2.getPerspectiveTransform(src=box,
                                    dst=np.array([[0, 0], [scale * w, 0], [scale * w, scale * h],
                                                  [0, scale * h]]).astype('float32'))
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
                 cval: int = 255):
    """Read an image from disk and fit to the specified size.

    Args:
        filepath: The path to the image or numpy array of shape HxWx3
        width: The new width
        height: The new height
        cval: The constant value to use to fill the remaining areas of
            the image

    Returns:
        The new image
    """
    image = read(filepath_or_array) if isinstance(filepath_or_array, str) else filepath_or_array
    image = fit(image=image, width=width, height=height, cval=cval)
    return image


def get_text_generator(max_string_length, alphabet=None):
    """Generates a lists of tuples of the form (category, content)
    where category is always "characters."
    """
    gen = essential_generators.DocumentGenerator()
    while True:
        sentence = ''.join([s for s in gen.sentence()
                            if (alphabet is None or s in alphabet)])[:max_string_length]
        yield [('characters', sentence)]


def convert_multiline_generator_to_single_line(multiline_generator, max_string_length, target_width,
                                               target_height):
    """Convert an image generator that creates multiline images to
    a generator suitable for training an OCR model with single lines.

    Args:
        multiline_generator: A genreator for multiline images
        max_stiring_length: The maximum string length to allow
        target_width: The width to warp lines into
        target_height: The height to warp lines into
    """
    while True:
        image, sentence, lines = next(multiline_generator)
        subset = lines[np.random.randint(0, len(lines))][:max_string_length]
        points = np.concatenate([np.array(s[:4]).reshape(2, 2) for s in subset] +
                                [np.array([s[6:8], s[4:6]])
                                 for s in reversed(subset)]).astype('float32')
        rectangle = cv2.minAreaRect(points)
        box = cv2.boxPoints(rectangle)

        # Put the points in clockwise order
        box = np.array(np.roll(box, 4 - box.sum(axis=1).argmin(), 0))
        sentence = ''.join([c[-1] for c in subset])
        lines = [subset]
        image = warpBox(image=image,
                        box=box,
                        target_width=target_width,
                        target_height=target_height)
        yield image, sentence, lines


def get_image_generator(
        height,
        width,
        font_groups,
        text_generator,
        x_start: typing.Union[int, typing.List[int]] = 0,
        y_start: typing.Union[int, typing.List[int]] = 0,
        font_weights: typing.Dict[str, typing.Union[int, typing.List[int]]] = None,
        font_slants: typing.Dict[str, typing.Union[int, typing.List[int]]] = None,
        font_size: typing.Union[int, typing.Tuple[int, int]] = 18,
        backgrounds: typing.List[typing.Tuple[str, typing.Tuple[int, int, int]]] = None,
        margin: int = 0,
        rotation: typing.Union[int, typing.Tuple[int, int]] = 0,
        single_line=False,
        line_spacing=1,
        augmenter1=None,
        background_crop_mode='crop',
        augmenter2=None):
    """Create a generator for images containing text.

    Args:
        height: The height of the generated image
        width: The width of the generated image.
        font_groups: A dict mapping of { category: [font1, font2] }.
        text_generator: See get_text_generator
        x_start: Where to start writing text in the image on the
            x-axis. Can be a value or a tuple of min/max values.
        y_start: Where to start writing text in the image on the
            y-axis. Can be a value or a tuple of min/max values.
        font_weights: The list of font slants to use for each
            category, similar in structure to font_groups.
        font_slants: The list of font slants to use for each
            category, similar in structure to font_groups.
        font_size: The font size to use. Alternative, supply a tuple
            and the font size will be randomly selected between
            the two values.
        backgrounds: A list of tuples of the form (path to
            background file or image as array, text color)
        margin: Minimum margin to apply around the image.
        background_crop_mode: One of letterbox or crop, indicates
            how backgrounds will be resized to fit on the canvas.
        rotation: The text rotation to use. Alternative, supply a tuple
            and the rotation will be randomly selected between
            the two values.
        augmenter1: An image augmenter to be applied to backgrounds
        augmenter2: An image augmenter to be applied to images after text overlay
    """
    assert cairocffi is not None, 'You must install cairocffi to use the generator.'
    for elements in text_generator:
        surface = cairocffi.ImageSurface(cairocffi.FORMAT_ARGB32, width, height)
        current_font_size = np.random.randint(low=font_size[0], high=font_size[1]) if isinstance(
            font_size, tuple) else font_size
        current_rotation = np.random.uniform(low=rotation[0], high=rotation[1]) if isinstance(
            rotation, tuple) else rotation
        current_rotation *= np.pi / 180
        current_font_groups = {
            category: np.random.choice(options)
            for category, options in font_groups.items()
        }
        if font_weights is not None:
            current_font_weights = {
                category: np.random.choice(options)
                for category, options in font_weights.items()
            }
        else:
            current_font_weights = {
                category: cairocffi.FONT_WEIGHT_NORMAL
                for category in font_groups
            }
        if font_weights is not None:
            current_font_slants = {
                category: np.random.choice(options)
                for category, options in font_slants.items()
            }
        else:
            current_font_slants = {
                category: cairocffi.FONT_SLANT_NORMAL
                for category in font_groups
            }
        if backgrounds is not None:
            current_background_filepath_or_array, text_color = backgrounds[np.random.randint(
                len(backgrounds))]
            current_background = read(current_background_filepath_or_array) if isinstance(
                current_background_filepath_or_array, str) else current_background_filepath_or_array
            if augmenter1 is not None:
                current_background = augmenter1(images=[current_background])[0]
            if current_background.shape[0] != height or current_background.shape[1] != width:
                current_background = fit(current_background,
                                         width=width,
                                         height=height,
                                         mode=background_crop_mode)
        else:
            current_background = np.zeros((height, width, 3), dtype='uint8')
            text_color = (255, 255, 255)

        cos = np.cos(current_rotation)
        sin = np.sin(current_rotation)

        def context_to_canvas(x, y):
            return x * cos - y * sin, x * sin + y * cos  # pylint: disable=cell-var-from-loop

        def canvas_to_context(x, y):
            return x * cos + y * sin, -x * sin + y * cos  # pylint: disable=cell-var-from-loop

        x_start_current = np.random.randint(x_start[0], x_start[1]) if isinstance(
            x_start, tuple) else x_start
        y_start_current = np.random.randint(y_start[0], y_start[1]) if isinstance(
            y_start, tuple) else x_start

        # (x_ctx, y_ctx) refers to the bottom-left corner of the text in context space
        # (i.e. we move only along the x direction)
        x_ctx, y_ctx = canvas_to_context(x_start_current, y_start_current)
        x_ctx_start, _ = x_ctx, y_ctx

        def coordinates(x_ctx, y_ctx, text_height, x_advance):
            # All of these coordinates are in canvas space
            # x1, y1 is the top left corner of the character
            # x2, y2 is the top right corner of the character
            # x3, y3 is the bottom right corner of the character
            # x4, y4 is the bottom left corner of the character
            x1, y1 = context_to_canvas(x_ctx, y_ctx - text_height)
            x2, y2 = context_to_canvas(x_ctx + x_advance, y_ctx - text_height)
            x3, y3 = context_to_canvas(x_ctx + x_advance, y_ctx)
            x4, y4 = context_to_canvas(x_ctx, y_ctx)
            inside = not (min(x1, x2, x3, x4) < margin or max(x1, x2, x3, x4) > width - margin
                          or min(y1, y2, y3, y4) < margin or max(y1, y2, y3, y4) > height - margin)
            return x1, y1, x2, y2, x3, y3, x4, y4, inside

        # Lines contain characters
        lines = [[]]
        with cairocffi.Context(surface) as context:
            context.rotate(current_rotation)
            context.set_source_rgba(*(c / 255 for c in text_color), 1)
            complete = ''
            space_remaining = True
            for category, sentence in elements:
                if not space_remaining:
                    break
                context.select_font_face(current_font_groups[category],
                                         weight=current_font_weights[category],
                                         slant=current_font_slants[category])
                context.set_font_size(current_font_size)
                for character in sentence:
                    # We use x_advance over width because it includes whitespace
                    # We use text_height because y_advance is related to the
                    # text baseline differential which is always zero
                    # except in languages with vertical text,
                    # which are not supported.
                    _, _, _, text_height, x_advance, _ = context.text_extents(character)
                    x1, y1, x2, y2, x3, y3, x4, y4, inside = coordinates(
                        x_ctx, y_ctx, text_height, x_advance)
                    if not inside:
                        if (single_line and any(len(line) > 0 for line in lines)):  # pylint: disable=len-as-condition
                            # If we are in single line mode and this isn't the first line,
                            # we break.
                            space_remaining = False
                            break
                        # Try moving to the next line.
                        x1, y1, x2, y2, x3, y3, x4, y4, inside = coordinates(
                            x_ctx_start, y_ctx + line_spacing * current_font_size, text_height,
                            x_advance)
                        if not inside:
                            # It didn't work.
                            space_remaining = False
                            break
                        else:
                            # It worked!
                            y_ctx += line_spacing * current_font_size
                            x_ctx = x_ctx_start
                            if len(lines[-1]) > 0:  # pylint: disable=len-as-condition
                                # We only do this if there is already something
                                # recorded on the current line. If there's not,
                                # it means we are still on the first line but there
                                # wasn't enough space.
                                lines.append([])
                    lines[-1].append((x1, y1, x2, y2, x3, y3, x4, y4, character))
                    context.move_to(x_ctx, y_ctx)
                    context.show_text(character)
                    complete += character
                    x_ctx += x_advance
            transparent = np.frombuffer(surface.get_data(), np.uint8).reshape(height, width,
                                                                              4).astype('float32')
            mask = transparent[..., 3:] / 255.0
            image = (1.0 - mask) * current_background.astype(
                'float32') + mask * transparent[..., :3][..., ::-1]
            image = image.clip(0, 255).astype('uint8')
            if augmenter2 is not None:
                image = augmenter2(images=[image])[0]
        yield image, complete, lines


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def download_and_verify(url, sha256=None):
    filename = os.path.basename(urllib.parse.urlparse(url).path)
    filepath = os.path.expanduser(os.path.join('~', '.keras-ocr', filename))
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    print('Looking for ' + filepath)
    if not os.path.isfile(filepath) or (sha256 and sha256sum(filepath) != sha256):
        print('Downloading ' + filepath)
        urllib.request.urlretrieve(url, filepath)
    assert sha256 == sha256sum(filepath), 'Error occurred verifying sha256.'
    return filepath
