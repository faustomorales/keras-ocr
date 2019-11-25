# pylint: disable=invalid-name,too-many-locals,too-many-arguments,too-many-branches,too-many-statements,stop-iteration-return
import os
import typing
import hashlib
import urllib.request
import urllib.parse

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import cv2
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


def get_rotation_matrix(width, height, thetaX=0, thetaY=0, thetaZ=0):
    """Provide a rotation matrix about the center of a rectangle with
    a given width and height.

    Args:
        width: The width of the rectangle
        height: The height of the rectangle
        thetaX: Rotation about the X axis
        thetaY: Rotation about the Y axis
        thetaZ: Rotation about the Z axis

    Returns:
        A 3x3 transformation matrix
    """
    translate1 = np.array([[1, 0, width / 2], [0, 1, height / 2], [0, 0, 1]])
    rotX = np.array([[1, 0, 0], [0, np.cos(thetaX), -np.sin(thetaX)],
                     [0, np.sin(thetaX), np.cos(thetaX)]])
    rotY = np.array([[np.cos(thetaY), 0, np.sin(thetaY)], [0, 1, 0],
                     [-np.sin(thetaY), 0, np.cos(thetaY)]])
    rotZ = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0], [np.sin(thetaZ),
                                                            np.cos(thetaZ), 0], [0, 0, 1]])
    translate2 = np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]])
    M = translate1.dot(rotX).dot(rotY).dot(rotZ).dot(translate2)
    return M


def compute_transformed_contour(width, height, fontsize, M, contour, minarea=0.5):
    """Compute the permitted drawing contour
    on a padded canvas for an image of a given size.
    We assume the canvas is padded with one full image width
    and height on left and right, top and bottom respectively.

    Args:
        width: Width of image
        height: Height of image
        fontsize: Size of characters
        M: The transformation matrix
        contour: The contour to which we are limited inside
            the rectangle of size width / height
        minarea: The minimum area required for a character
            slot to qualify as being visible, expressed as
            a fraction of the untransformed fontsize x fontsize
            slot.
    """
    xslots = int(np.floor(width / fontsize))
    yslots = int(np.floor(height / fontsize))
    ys, xs = np.mgrid[:yslots, :xslots]
    basis = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1).reshape((-1, 2))
    basis *= fontsize
    slots_pretransform = np.concatenate(
        [(basis + offset)[:, np.newaxis, :]
         for offset in [[0, 0], [fontsize, 0], [fontsize, fontsize], [0, fontsize]]],
        axis=1)
    slots = cv2.perspectiveTransform(src=slots_pretransform.reshape((1, -1, 2)).astype('float32'),
                                     m=M)[0]
    inside = np.array([
        cv2.pointPolygonTest(contour=contour, pt=(x, y), measureDist=False) >= 0 for x, y in slots
    ]).reshape(-1, 4).all(axis=1)
    slots = slots.reshape(-1, 4, 2)
    areas = np.abs((slots[:, 0, 0] * slots[:, 1, 1] - slots[:, 0, 1] * slots[:, 1, 0]) +
                   (slots[:, 1, 0] * slots[:, 2, 1] - slots[:, 1, 1] * slots[:, 2, 0]) +
                   (slots[:, 2, 0] * slots[:, 3, 1] - slots[:, 2, 1] * slots[:, 3, 0]) +
                   (slots[:, 3, 0] * slots[:, 0, 1] - slots[:, 3, 1] * slots[:, 0, 0])) / 2
    slots_filtered = slots_pretransform[(areas > minarea * fontsize * fontsize) & inside]
    contour = cv2.convexHull(points=slots_filtered[:, 0, :])[:, 0, :]
    return contour


def get_maximum_uniform_contour(image, fontsize):
    """Get the largest possible contour of light or
    dark area in an image.

    Args:
        image: The image in which to find a contiguous area.
        fontsize: The fontsize for text. Will be used for blurring
            and for determining useful areas.

    Returns:
        A (contour, isDark) tuple. If no contour is found, both
        entries will be None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.blur(src=gray, ksize=(fontsize, fontsize))
    _, threshold = cv2.threshold(src=blurred, thresh=255 / 2, maxval=255, type=cv2.THRESH_BINARY)
    contoursDark, _ = cv2.findContours(255 - threshold,
                                       mode=cv2.RETR_TREE,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
    contoursLight, _ = cv2.findContours(threshold,
                                        mode=cv2.RETR_TREE,
                                        method=cv2.CHAIN_APPROX_SIMPLE)
    areasDark = list(map(cv2.contourArea, contoursDark))
    areasLight = list(map(cv2.contourArea, contoursLight))
    maxDarkArea = max(areasDark) if areasDark else 0
    maxLightArea = max(areasLight) if areasLight else 0

    if max(maxDarkArea, maxLightArea) < (4 * fontsize)**2:
        return None, None

    contour = None
    isDark = None
    if areasDark and (not areasLight or maxDarkArea >= maxLightArea):
        contour = contoursDark[np.argmax(areasDark)]
        isDark = True
    else:
        contour = contoursLight[np.argmax(areasLight)]
        isDark = False
    return contour, isDark


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


def draw_text_image(text_groups,
                    fontsize,
                    height,
                    width,
                    fonts,
                    thetaX=0,
                    thetaY=0,
                    thetaZ=0,
                    color=(0, 0, 0),
                    permitted_contour=None):
    """Get a transparent image containing text.

    Args:
        text_groups: A list of (category, text) tuples.
        fontsize: The size of text to show.
        height: The height of the output image
        width: The width of the output image
        fonts: A dictionary of {category: [paths_to_fonts]}
        thetaX: Rotation about the X axis
        thetaY: Rotation about the Y axis
        thetaZ: Rotation about the Z axis
        color: The color of drawn text
        permitted_contour: A contour defining which part of the image
            we can put text. If None, the entire canvas is permitted
            for text.

    Returns:
        An (image, sentence, lines) tuple where image is the
        transparent text image, sentence is the full text string,
        and lines is a list of lines where each line itself is a list
        of (character, box) tuples.
    """
    ligatures = {'\U0000FB01': 'fi', '\U0000FB02': 'fl'}
    for replace, search in ligatures.items():
        for index, (category, text) in enumerate(text_groups):
            text_groups[index] = (category, text.replace(search, replace))

    M = get_rotation_matrix(width=width, height=height, thetaZ=thetaZ, thetaX=thetaX, thetaY=thetaY)
    if permitted_contour is None:
        permitted_contour = np.array([[0, 0], [width, 0], [width, height],
                                      [0, height]]).astype('float32')
    transformed_contour = compute_transformed_contour(width=width,
                                                      height=height,
                                                      fontsize=fontsize,
                                                      M=M,
                                                      contour=permitted_contour)
    start_x = transformed_contour[:, 0].min()
    start_y = transformed_contour[:, 1].min()
    end_x = transformed_contour[:, 0].max()
    end_y = transformed_contour[:, 1].max()
    image = PIL.Image.new(mode='RGBA', size=(width, height), color=(255, 255, 255, 0))
    fonts = {
        category: PIL.ImageFont.truetype(font_path, size=fontsize)
        for category, font_path in fonts.items()
    }
    draw = PIL.ImageDraw.Draw(image)
    lines = [[]]
    sentence = ''
    x = start_x
    y = start_y
    out_of_space = False
    for category, text in text_groups:
        font = fonts[category]
        if out_of_space:
            break
        for character in text:
            character_width, character_height = font.getsize(character)
            if character in ligatures:
                subcharacters = ligatures[character]
                dx = character_width / len(subcharacters)
            else:
                subcharacters = character
                dx = character_width
            while cv2.pointPolygonTest(contour=transformed_contour, pt=(x, y),
                                       measureDist=False) < 0:
                if x + (dx * len(subcharacters)) > end_x:
                    if y + fontsize > end_y:
                        out_of_space = True
                        break
                    y += fontsize
                    lines.append([])
                    x = start_x
                    continue
                else:
                    x += fontsize
            if out_of_space:
                break
            draw.text(xy=(x, y), text=character, fill=color + (255, ), font=font)
            for subcharacter in subcharacters:
                lines[-1].append((subcharacter,
                                  np.array([[x, 0], [x + dx, 0], [x + dx, character_height],
                                            [0, character_height]]).astype('float32')))
                sentence += subcharacter
                x += dx
    image = cv2.warpPerspective(src=np.array(image), M=M, dsize=(width, height))
    lines = [[(character, cv2.perspectiveTransform(src=coords[np.newaxis], m=M)[0])
              for character, coords in line] for line in lines]
    return image, sentence, lines


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
        font_size: typing.Union[int, typing.Tuple[int, int]] = 18,
        backgrounds: typing.List[typing.Tuple[str, typing.Tuple[int, int, int]]] = None,
        rotationX: typing.Union[int, typing.Tuple[int, int]] = 0,
        rotationY: typing.Union[int, typing.Tuple[int, int]] = 0,
        rotationZ: typing.Union[int, typing.Tuple[int, int]] = 0,
        augmenter1=None,
        background_crop_mode='crop',
        augmenter2=None):
    """Create a generator for images containing text.

    Args:
        height: The height of the generated image
        width: The width of the generated image.
        font_groups: A dict mapping of { category: [font1, font2] }.
        text_generator: See get_text_generator
        font_weights: The list of font slants to use for each
            category, similar in structure to font_groups.
        font_slants: The list of font slants to use for each
            category, similar in structure to font_groups.
        font_size: The font size to use. Alternative, supply a tuple
            and the font size will be randomly selected between
            the two values.
        backgrounds: A list of tuples of the form (path to
            background file or image as array, text color)
        background_crop_mode: One of letterbox or crop, indicates
            how backgrounds will be resized to fit on the canvas.
        rotation: The text rotation to use. Alternative, supply a tuple
            and the rotation will be randomly selected between
            the two values.
        augmenter1: An image augmenter to be applied to backgrounds
        augmenter2: An image augmenter to be applied to images after text overlay
    """
    for elements in text_generator:
        current_font_size = np.random.randint(low=font_size[0], high=font_size[1]) if isinstance(
            font_size, tuple) else font_size
        current_rotation_X, current_rotation_Y, current_rotation_Z = [
            (np.random.uniform(low=rotation[0], high=rotation[1])
             if isinstance(rotation, tuple) else rotation) * np.pi / 180
            for rotation in [rotationX, rotationY, rotationZ]
        ]
        current_font_groups = {
            category: np.random.choice(options)
            for category, options in font_groups.items()
        }
        if backgrounds is not None:
            current_background_filepath_or_array = backgrounds[np.random.randint(len(backgrounds))]
            current_background = read(current_background_filepath_or_array) if isinstance(
                current_background_filepath_or_array, str) else current_background_filepath_or_array
            if augmenter1 is not None:
                current_background = augmenter1(images=[current_background])[0]
            if current_background.shape[0] != height or current_background.shape[1] != width:
                current_background = fit(current_background,
                                         width=width,
                                         height=height,
                                         mode=background_crop_mode)
            permitted_contour, isDark = get_maximum_uniform_contour(image=current_background,
                                                                    fontsize=current_font_size)
            text_color = (255, 255, 255) if isDark else (0, 0, 0)
        else:
            current_background = np.zeros((height, width, 3), dtype='uint8')
            permitted_contour = None
            text_color = (255, 255, 255)

        text_image, sentence, lines = draw_text_image(text_groups=elements,
                                                      width=width,
                                                      height=height,
                                                      fontsize=current_font_size,
                                                      fonts=current_font_groups,
                                                      thetaX=current_rotation_X,
                                                      thetaY=current_rotation_Y,
                                                      thetaZ=current_rotation_Z,
                                                      permitted_contour=permitted_contour,
                                                      color=text_color)
        alpha = text_image[..., -1:].astype('float32') / 255
        image = (alpha * text_image[..., :3] + (1 - alpha) * current_background).astype('uint8')
        if augmenter2 is not None:
            image = augmenter2(images=[image])[0]
        yield image, sentence, lines


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
