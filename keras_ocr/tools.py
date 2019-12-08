# pylint: disable=invalid-name,too-many-locals,too-many-arguments,too-many-branches,too-many-statements,stop-iteration-return
import os
import math
import typing
import random
import hashlib
import itertools
import urllib.request
import urllib.parse

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import numpy as np
import essential_generators
import fontTools.ttLib

LIGATURES = {'\U0000FB01': 'fi', '\U0000FB02': 'fl'}
LIGATURE_STRING = ''.join(LIGATURES.keys())


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
    spacing = math.ceil(fontsize / 2)
    xslots = int(np.floor(width / spacing))
    yslots = int(np.floor(height / spacing))
    ys, xs = np.mgrid[:yslots, :xslots]
    basis = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1).reshape((-1, 2))
    basis *= spacing
    slots_pretransform = np.concatenate(
        [(basis + offset)[:, np.newaxis, :]
         for offset in [[0, 0], [spacing, 0], [spacing, spacing], [0, spacing]]],
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
    slots_filtered = slots_pretransform[(areas > minarea * spacing * spacing) & inside]
    temporary_image = cv2.drawContours(image=np.zeros((height, width), dtype='uint8'),
                                       contours=slots_filtered,
                                       contourIdx=-1,
                                       color=255)
    temporary_image = cv2.dilate(src=temporary_image, kernel=np.ones((spacing, spacing)))
    newContours, _ = cv2.findContours(temporary_image,
                                      mode=cv2.RETR_TREE,
                                      method=cv2.CHAIN_APPROX_SIMPLE)
    x, y = slots_filtered[0][0]
    contour = newContours[next(
        index for index, contour in enumerate(newContours)
        if cv2.pointPolygonTest(contour=contour, pt=(x, y), measureDist=False) >= 0)][:, 0, :]
    return contour


def get_maximum_uniform_contour(image, fontsize, margin=0):
    """Get the largest possible contour of light or
    dark area in an image.

    Args:
        image: The image in which to find a contiguous area.
        fontsize: The fontsize for text. Will be used for blurring
            and for determining useful areas.
        margin: The minimum margin required around the image.

    Returns:
        A (contour, isDark) tuple. If no contour is found, both
        entries will be None.
    """
    if margin > 0:
        image = image[margin:-margin, margin:-margin]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.blur(src=gray, ksize=(fontsize // 2, fontsize // 2))
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
    if contour is not None:
        contour += margin
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


def draw_text_image(text,
                    fontsize,
                    height,
                    width,
                    fonts,
                    use_ligatures=False,
                    thetaX=0,
                    thetaY=0,
                    thetaZ=0,
                    color=(0, 0, 0),
                    permitted_contour=None,
                    draw_contour=False):
    """Get a transparent image containing text.

    Args:
        text: The text to draw on the image
        fontsize: The size of text to show.
        height: The height of the output image
        width: The width of the output image
        fonts: A dictionary of {subalphabet: paths_to_font}
        thetaX: Rotation about the X axis
        thetaY: Rotation about the Y axis
        thetaZ: Rotation about the Z axis
        color: The color of drawn text
        permitted_contour: A contour defining which part of the image
            we can put text. If None, the entire canvas is permitted
            for text.
        use_ligatures: Whether to render ligatures. If True,
            ligatures are always used (with an initial check for support
            which sometimes yields false positives). If False, ligatures
            are never used.

    Returns:
        An (image, sentence, lines) tuple where image is the
        transparent text image, sentence is the full text string,
        and lines is a list of lines where each line itself is a list
        of (box, character) tuples.
    """
    # pylint: disable=bad-continuation
    if not use_ligatures:
        fonts = {
            subalphabet: PIL.ImageFont.truetype(font_path, size=fontsize)
            if font_path is not None else PIL.ImageFont.load_default()
            for subalphabet, font_path in fonts.items()
        }
    if use_ligatures:
        for subalphabet, font_path in fonts.items():
            ligatures_supported = True
            font = PIL.ImageFont.truetype(
                font_path,
                size=fontsize) if font_path is not None else PIL.ImageFont.load_default()
            for ligature in LIGATURES:
                try:
                    font.getsize(ligature)
                except UnicodeEncodeError:
                    ligatures_supported = False
                    break
            if ligatures_supported:
                del fonts[subalphabet]
                subalphabet += LIGATURE_STRING
            fonts[subalphabet] = font
        for insert, search in LIGATURES.items():
            for subalphabet in fonts.keys()():
                if insert in subalphabet:
                    text = text.replace(search, insert)
    character_font_pairs = [(character,
                             next(font for subalphabet, font in fonts.items()
                                  if character in subalphabet)) for character in text]
    M = get_rotation_matrix(width=width, height=height, thetaZ=thetaZ, thetaX=thetaX, thetaY=thetaY)
    if permitted_contour is None:
        permitted_contour = np.array([[0, 0], [width, 0], [width, height],
                                      [0, height]]).astype('float32')
    character_sizes = np.array(
        [font.font.getsize(character) for character, font in character_font_pairs])
    min_character_size = character_sizes.sum(axis=1).min()
    transformed_contour = compute_transformed_contour(width=width,
                                                      height=height,
                                                      fontsize=max(min_character_size, 1),
                                                      M=M,
                                                      contour=permitted_contour)
    start_x = transformed_contour[:, 0].min()
    start_y = transformed_contour[:, 1].min()
    end_x = transformed_contour[:, 0].max()
    end_y = transformed_contour[:, 1].max()
    image = PIL.Image.new(mode='RGBA', size=(width, height), color=(255, 255, 255, 0))
    draw = PIL.ImageDraw.Draw(image)
    lines = [[]]
    sentence = ''
    x = start_x
    y = start_y
    max_y = start_y
    out_of_space = False
    for character_index, (character, font) in enumerate(character_font_pairs):
        if out_of_space:
            break
        (character_width, character_height), (offset_x, offset_y) = character_sizes[character_index]
        if character in LIGATURES:
            subcharacters = LIGATURES[character]
            dx = character_width / len(subcharacters)
        else:
            subcharacters = character
            dx = character_width
        x2, y2 = (x + character_width + offset_x, y + character_height + offset_y)
        while not all(
                cv2.pointPolygonTest(contour=transformed_contour, pt=pt, measureDist=False) >= 0
                for pt in [(x, y), (x2, y), (x2, y2), (x, y2)]):
            if x2 > end_x:
                dy = max(1, max_y - y)
                if y + dy > end_y:
                    out_of_space = True
                    break
                y += dy
                x = start_x
            else:
                x += fontsize
            if len(lines[-1]) > 0:
                # We add a new line whether we have advanced
                # in the y-direction or not because we also want to separate
                # horizontal segments of text.
                lines.append([])
            x2, y2 = (x + character_width + offset_x, y + character_height + offset_y)
        if out_of_space:
            break
        max_y = max(y + character_height + offset_y, max_y)
        draw.text(xy=(x, y), text=character, fill=color + (255, ), font=font)
        for subcharacter in subcharacters:
            lines[-1].append((np.array([[x + offset_x, y + offset_y],
                                        [x + dx + offset_x, y + offset_y], [x + dx + offset_x, y2],
                                        [x + offset_x, y2]]).astype('float32'), subcharacter))
            sentence += subcharacter
            x += dx
    image = cv2.warpPerspective(src=np.array(image), M=M, dsize=(width, height))
    if draw_contour:
        image = cv2.drawContours(image,
                                 contours=[permitted_contour.reshape((-1, 1, 2)).astype('int32')],
                                 contourIdx=0,
                                 color=(255, 0, 0, 255),
                                 thickness=int(width / 100))
    lines = strip_lines(lines)
    lines = [[(cv2.perspectiveTransform(src=coords[np.newaxis], m=M)[0], character)
              for coords, character in line] for line in lines]
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


def get_text_generator(alphabet=None, lowercase=False, max_string_length=None):
    """Generates strings of sentences using only the letters in alphabet.

    Args:
        alphabet: The alphabet of permitted characters
        lowercase: Whether to convert all strings to lowercase.
        max_string_length: The maximum length of the string
    """
    gen = essential_generators.DocumentGenerator()
    while True:
        sentence = gen.sentence()
        if lowercase:
            sentence = sentence.lower()
        sentence = ''.join([s for s in sentence if (alphabet is None or s in alphabet)])
        if max_string_length is not None:
            sentence = sentence[:max_string_length]
        yield sentence


def strip_line(line):
    """Modify a line so that spaces are excluded."""
    first_character_index = next(
        (index for index, (box, character) in enumerate(line) if not character.isspace()), None)
    if first_character_index is None:
        return []
    last_character_index = len(line) - next(
        index for index, (box, character) in enumerate(reversed(line)) if not character.isspace())
    return line[first_character_index:last_character_index]


def strip_lines(lines):
    """Modify a set of lines so that spaces are excluded."""
    lines = [line for line in lines if len(line) > 0]
    lines = [strip_line(line) for line in lines]
    lines = [line for line in lines if len(line) > 0]
    return lines


def convert_multiline_generator_to_single_line(multiline_generator,
                                               max_string_length,
                                               target_width,
                                               target_height,
                                               margin=0):
    """Convert an image generator that creates multiline images to
    a generator suitable for training an OCR model with single lines.

    Args:
        multiline_generator: A genreator for multiline images
        max_string_length: The maximum string length to allow
        target_width: The width to warp lines into
        target_height: The height to warp lines into
        margin: The margin to apply around a single line.
    """
    while True:
        image, sentence, lines = next(multiline_generator)
        if len(lines) == 0:
            continue
        subset = strip_line(lines[np.argmax(list(map(len, lines)))][:max_string_length])
        points = np.concatenate(
            [coords[:2] for coords, _ in subset] +
            [np.array([coords[3], coords[2]]) for coords, _ in reversed(subset)]).astype('float32')
        rectangle = cv2.minAreaRect(points)
        box = cv2.boxPoints(rectangle)

        # Put the points in clockwise order
        box = np.array(np.roll(box, 4 - box.sum(axis=1).argmin(), 0))
        sentence = ''.join([c[-1] for c in subset])
        lines = [subset]
        image = warpBox(image=image,
                        box=box,
                        target_width=target_width,
                        target_height=target_height,
                        margin=margin)
        yield image, sentence, lines


def get_image_generator(height,
                        width,
                        font_groups,
                        text_generator,
                        font_size: typing.Union[int, typing.Tuple[int, int]] = 18,
                        backgrounds: typing.List[typing.Union[str, np.ndarray]] = None,
                        background_crop_mode='crop',
                        rotationX: typing.Union[int, typing.Tuple[int, int]] = 0,
                        rotationY: typing.Union[int, typing.Tuple[int, int]] = 0,
                        rotationZ: typing.Union[int, typing.Tuple[int, int]] = 0,
                        margin=0,
                        use_ligatures=False,
                        augmenter=None,
                        draw_contour=False,
                        draw_contour_text=False):
    """Create a generator for images containing text.

    Args:
        height: The height of the generated image
        width: The width of the generated image.
        font_groups: A dict mapping of { subalphabet: [path_to_font1, path_to_font2] }.
        text_generator: See get_text_generator
        font_size: The font size to use. Alternative, supply a tuple
            and the font size will be randomly selected between
            the two values.
        backgrounds: A list of paths to image backgrounds or actual images
            as numpy arrays with channels in RGB order.
        background_crop_mode: One of letterbox or crop, indicates
            how backgrounds will be resized to fit on the canvas.
        rotationX: The X-axis text rotation to use. Alternative, supply a tuple
            and the rotation will be randomly selected between
            the two values.
        rotationY: The Y-axis text rotation to use. Alternative, supply a tuple
            and the rotation will be randomly selected between
            the two values.
        rotationZ: The Z-axis text rotation to use. Alternative, supply a tuple
            and the rotation will be randomly selected between
            the two values.
        margin: The minimum margin around the edge of the image.
        use_ligatures: Whether to render ligatures (see `draw_text_image`)
        augmenter: An image augmenter to be applied to backgrounds
        draw_contour: Draw the permitted contour onto images (debugging only)
        draw_contour_text: Draw the permitted contour inside the text
            drawing function.
    """
    if backgrounds is None:
        backgrounds = [np.zeros((height, width, 3), dtype='uint8')]
    alphabet = ''.join(font_groups.keys())
    assert len(set(alphabet)) == len(
        alphabet), 'Each character can appear in the subalphabet for only one font group.'
    for text, background_index, current_font_groups in zip(
            text_generator, itertools.cycle(range(len(backgrounds))),
            zip(*[
                itertools.cycle([(subalphabet, font_filepath)
                                 for font_filepath in font_group_filepaths])
                for subalphabet, font_group_filepaths in font_groups.items()
            ])):
        if background_index == 0:
            random.shuffle(backgrounds)
        current_font_groups = dict(current_font_groups)
        current_font_size = np.random.randint(low=font_size[0], high=font_size[1]) if isinstance(
            font_size, tuple) else font_size
        current_rotation_X, current_rotation_Y, current_rotation_Z = [
            (np.random.uniform(low=rotation[0], high=rotation[1])
             if isinstance(rotation, tuple) else rotation) * np.pi / 180
            for rotation in [rotationX, rotationY, rotationZ]
        ]
        current_background_filepath_or_array = backgrounds[background_index]
        current_background = read(current_background_filepath_or_array) if isinstance(
            current_background_filepath_or_array, str) else current_background_filepath_or_array
        if augmenter is not None:
            current_background = augmenter(images=[current_background])[0]
        if current_background.shape[0] != height or current_background.shape[1] != width:
            current_background = fit(current_background,
                                     width=width,
                                     height=height,
                                     mode=background_crop_mode)
        permitted_contour, isDark = get_maximum_uniform_contour(image=current_background,
                                                                fontsize=current_font_size,
                                                                margin=margin)
        if permitted_contour is None:
            # We can't draw on this background. Boo!
            continue
        random_color_values = np.random.randint(low=0, high=50, size=3)
        text_color = tuple(np.array([255, 255, 255]) -
                           random_color_values) if isDark else tuple(random_color_values)
        text_image, sentence, lines = draw_text_image(text=text,
                                                      width=width,
                                                      height=height,
                                                      fontsize=current_font_size,
                                                      fonts=current_font_groups,
                                                      thetaX=current_rotation_X,
                                                      thetaY=current_rotation_Y,
                                                      thetaZ=current_rotation_Z,
                                                      use_ligatures=use_ligatures,
                                                      permitted_contour=permitted_contour,
                                                      color=text_color,
                                                      draw_contour=draw_contour_text)
        alpha = text_image[..., -1:].astype('float32') / 255
        image = (alpha * text_image[..., :3] + (1 - alpha) * current_background).astype('uint8')
        if draw_contour:
            image = cv2.drawContours(
                image,
                contours=[permitted_contour.reshape((-1, 1, 2)).astype('int32')],
                contourIdx=0,
                color=(255, 0, 0),
                thickness=int(width / 100))
        yield image, sentence, lines


def sha256sum(filename):
    """Compute the sha256 hash for a file."""
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def font_supports_alphabet(filepath, alphabet):
    """Verify that a font contains a specific set of characters.

    Args:
        filepath: Path to fsontfile
        alphabet: A string of characters to check for.
    """
    font = fontTools.ttLib.TTFont(filepath)
    if not all(any(ord(c) in table.cmap.keys() for table in font['cmap'].tables) for c in alphabet):
        return False
    font = PIL.ImageFont.truetype(filepath)
    try:
        for character in alphabet:
            font.getsize(character)
    # pylint: disable=bare-except
    except:
        return False
    return True


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
