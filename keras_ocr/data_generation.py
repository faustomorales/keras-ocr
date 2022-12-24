# pylint: disable=invalid-name,line-too-long,too-many-locals,too-many-arguments,too-many-branches,too-many-statements,stop-iteration-return
import os
import math
import glob
import typing
import random
import zipfile
import string
import itertools

import cv2
import tqdm
import numpy as np
import essential_generators
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import fontTools.ttLib

from . import tools

LIGATURES = {"\U0000FB01": "fi", "\U0000FB02": "fl"}
LIGATURE_STRING = "".join(LIGATURES.keys())


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
    rotX = np.array(
        [
            [1, 0, 0],
            [0, np.cos(thetaX), -np.sin(thetaX)],
            [0, np.sin(thetaX), np.cos(thetaX)],
        ]
    )
    rotY = np.array(
        [
            [np.cos(thetaY), 0, np.sin(thetaY)],
            [0, 1, 0],
            [-np.sin(thetaY), 0, np.cos(thetaY)],
        ]
    )
    rotZ = np.array(
        [
            [np.cos(thetaZ), -np.sin(thetaZ), 0],
            [np.sin(thetaZ), np.cos(thetaZ), 0],
            [0, 0, 1],
        ]
    )
    translate2 = np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]])
    M = translate1.dot(rotX).dot(rotY).dot(rotZ).dot(translate2)
    return M


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
    _, threshold = cv2.threshold(
        src=blurred, thresh=255 / 2, maxval=255, type=cv2.THRESH_BINARY
    )
    contoursDark = cv2.findContours(
        255 - threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    contoursLight = cv2.findContours(
        threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    areasDark = list(map(cv2.contourArea, contoursDark))
    areasLight = list(map(cv2.contourArea, contoursLight))
    maxDarkArea = max(areasDark) if areasDark else 0
    maxLightArea = max(areasLight) if areasLight else 0

    if max(maxDarkArea, maxLightArea) < (4 * fontsize) ** 2:
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


def font_supports_alphabet(filepath, alphabet):
    """Verify that a font contains a specific set of characters.

    Args:
        filepath: Path to fsontfile
        alphabet: A string of characters to check for.
    """
    if alphabet == "":
        return True
    font = fontTools.ttLib.TTFont(filepath)
    if not all(
        any(ord(c) in table.cmap.keys() for table in font["cmap"].tables)
        for c in alphabet
    ):
        return False
    font = PIL.ImageFont.truetype(filepath)
    try:
        for character in alphabet:
            font.getsize(character)
    # pylint: disable=bare-except
    except:
        return False
    return True


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
        sentence = "".join([s for s in sentence if (alphabet is None or s in alphabet)])
        if max_string_length is not None:
            sentence = sentence[:max_string_length]
        yield sentence


def _strip_line(line):
    """Modify a line so that spaces are excluded."""
    first_character_index = next(
        (
            index
            for index, (box, character) in enumerate(line)
            if not character.isspace()
        ),
        None,
    )
    if first_character_index is None:
        return []
    last_character_index = len(line) - next(
        index
        for index, (box, character) in enumerate(reversed(line))
        if not character.isspace()
    )
    return line[first_character_index:last_character_index]


def _strip_lines(lines):
    """Modify a set of lines so that spaces are excluded."""
    lines = [line for line in lines if len(line) > 0]
    lines = [_strip_line(line) for line in lines]
    lines = [line for line in lines if len(line) > 0]
    return lines


def get_backgrounds(cache_dir=None):
    """Download a set of pre-reviewed backgrounds.

    Args:
        cache_dir: Where to save the dataset. By default, data will be
            saved to ~/.keras-ocr.

    Returns:
        A list of background filepaths.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.path.join("~", ".keras-ocr"))
    backgrounds_dir = os.path.join(cache_dir, "backgrounds")
    backgrounds_zip_path = tools.download_and_verify(
        url="https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/backgrounds.zip",
        sha256="f263ed0d55de303185cc0f93e9fcb0b13104d68ed71af7aaaa8e8c91389db471",
        filename="backgrounds.zip",
        cache_dir=cache_dir,
    )
    if len(glob.glob(os.path.join(backgrounds_dir, "*"))) != 1035:
        with zipfile.ZipFile(backgrounds_zip_path) as zfile:
            zfile.extractall(backgrounds_dir)
    return glob.glob(os.path.join(backgrounds_dir, "*.jpg"))


def get_fonts(
    cache_dir=None,
    alphabet=string.ascii_letters + string.digits,
    exclude_smallcaps=False,
):
    """Download a set of pre-reviewed fonts.

    Args:
        cache_dir: Where to save the dataset. By default, data will be
            saved to ~/.keras-ocr.
        alphabet: An alphabet which we will use to exclude fonts
            that are missing relevant characters. By default, this is
            set to `string.ascii_letters + string.digits`.
        exclude_smallcaps: If True, fonts that are known to use
            the same glyph for lowercase and uppercase characters
            are excluded.

    Returns:
        A list of font filepaths.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.path.join("~", ".keras-ocr"))
    fonts_zip_path = tools.download_and_verify(
        url="https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/fonts.zip",
        sha256="d4d90c27a9bc4bf8fff1d2c0a00cfb174c7d5d10f60ed29d5f149ef04d45b700",
        filename="fonts.zip",
        cache_dir=cache_dir,
    )
    fonts_dir = os.path.join(cache_dir, "fonts")
    if len(glob.glob(os.path.join(fonts_dir, "**/*.ttf"))) != 2746:
        print("Unzipping fonts ZIP file.")
        with zipfile.ZipFile(fonts_zip_path) as zfile:
            zfile.extractall(fonts_dir)
    font_filepaths = glob.glob(os.path.join(fonts_dir, "**/*.ttf"))
    if exclude_smallcaps:
        with open(
            tools.download_and_verify(
                url="https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/fonts_smallcaps.txt",
                sha256="6531c700523c687f02852087530d1ab3c7cc0b59891bbecc77726fbb0aabe68e",
                filename="fonts_smallcaps.txt",
                cache_dir=cache_dir,
            ),
            "r",
            encoding="utf8",
        ) as f:
            smallcaps_fonts = f.read().split("\n")
            smallcaps_fonts = [ origpath.replace('/', os.path.sep) for origpath in smallcaps_fonts ]
            font_filepaths = [
                filepath
                for filepath in font_filepaths
                if os.path.join(*filepath.split(os.sep)[-2:]) not in smallcaps_fonts
            ]
    if alphabet != "":
        font_filepaths = [
            filepath
            for filepath in tqdm.tqdm(font_filepaths, desc="Filtering fonts.")
            if font_supports_alphabet(filepath=filepath, alphabet=alphabet)
        ]
    return font_filepaths


def convert_lines_to_paragraph(lines):
    """Convert a series of lines, each consisting of
    (box, character) tuples, into a multi-line string."""
    return "\n".join(["".join([c[-1] for c in line]) for line in lines])


def convert_image_generator_to_recognizer_input(
    image_generator, max_string_length, target_width, target_height, margin=0
):
    """Convert an image generator created by get_image_generator
    to (image, sentence) tuples for training a recognizer.

    Args:
        image_generator: An image generator created by get_image_generator
        max_string_length: The maximum string length to allow
        target_width: The width to warp lines into
        target_height: The height to warp lines into
        margin: The margin to apply around a single line.
    """
    while True:
        image, lines = next(image_generator)
        if len(lines) == 0:
            continue
        for line in lines:
            line = _strip_line(line[:max_string_length])
            if not line:
                continue
            box, sentence = tools.combine_line(line)

            # remove multiple sequential spaces
            while "  " in sentence:
                sentence = sentence.replace("  ", " ")

            crop = tools.warpBox(
                image=image,
                box=box,
                target_width=target_width,
                target_height=target_height,
                margin=margin,
                skip_rotate=True,
            )
            yield crop, sentence


def draw_text_image(
    text,
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
    draw_contour=False,
):
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
        An (image, lines) tuple where image is the
        transparent text image and lines is a list of lines
        where each line itself is a list of (box, character) tuples and
        box is an array of points with shape (4, 2) providing the coordinates
        of the character box in clockwise order starting from the top left.
    """
    if not use_ligatures:
        fonts = {
            subalphabet: PIL.ImageFont.truetype(font_path, size=fontsize)
            if font_path is not None
            else PIL.ImageFont.load_default()
            for subalphabet, font_path in fonts.items()
        }
    if use_ligatures:
        for subalphabet, font_path in fonts.items():
            ligatures_supported = True
            font = (
                PIL.ImageFont.truetype(font_path, size=fontsize)
                if font_path is not None
                else PIL.ImageFont.load_default()
            )
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
    character_font_pairs = [
        (
            character,
            next(
                font for subalphabet, font in fonts.items() if character in subalphabet
            ),
        )
        for character in text
    ]
    M = get_rotation_matrix(
        width=width, height=height, thetaZ=thetaZ, thetaX=thetaX, thetaY=thetaY
    )
    if permitted_contour is None:
        permitted_contour = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]]
        ).astype("float32")
    character_sizes = np.array(
        [font.font.getsize(character) for character, font in character_font_pairs]
    )
    min_character_size = character_sizes.sum(axis=1).min()
    transformed_contour = compute_transformed_contour(
        width=width,
        height=height,
        fontsize=max(min_character_size, 1),
        M=M,
        contour=permitted_contour,
    )
    start_x = transformed_contour[:, 0].min()
    start_y = transformed_contour[:, 1].min()
    end_x = transformed_contour[:, 0].max()
    end_y = transformed_contour[:, 1].max()
    image = PIL.Image.new(mode="RGBA", size=(width, height), color=(255, 255, 255, 0))
    draw = PIL.ImageDraw.Draw(image)
    lines_raw: typing.List[typing.List[typing.Tuple[np.ndarray, str]]] = [[]]
    x = start_x
    y = start_y
    max_y = start_y
    out_of_space = False
    for character_index, (character, font) in enumerate(character_font_pairs):
        if out_of_space:
            break
        (character_width, character_height), (offset_x, offset_y) = character_sizes[
            character_index
        ]
        if character in LIGATURES:
            subcharacters = LIGATURES[character]
            dx = character_width / len(subcharacters)
        else:
            subcharacters = character
            dx = character_width
        x2, y2 = (x + character_width + offset_x, y + character_height + offset_y)
        while not all(
            cv2.pointPolygonTest(contour=transformed_contour, pt=pt, measureDist=False)
            >= 0
            for pt in [(int(x), int(y)), (int(x2), int(y)), (int(x2), int(y2)), (int(x), int(y2))]
        ):
            if x2 > end_x:
                dy = max(1, max_y - y)
                if y + dy > end_y:
                    out_of_space = True
                    break
                y += dy
                x = start_x
            else:
                x += fontsize
            if len(lines_raw[-1]) > 0:
                # We add a new line whether we have advanced
                # in the y-direction or not because we also want to separate
                # horizontal segments of text.
                lines_raw.append([])
            x2, y2 = (x + character_width + offset_x, y + character_height + offset_y)
        if out_of_space:
            break
        max_y = max(y + character_height + offset_y, max_y)
        draw.text(xy=(x, y), text=character, fill=color + (255,), font=font)
        for subcharacter in subcharacters:
            lines_raw[-1].append(
                (
                    np.array(
                        [
                            [x + offset_x, y + offset_y],
                            [x + dx + offset_x, y + offset_y],
                            [x + dx + offset_x, y2],
                            [x + offset_x, y2],
                        ]
                    ).astype("float32"),
                    subcharacter,
                )
            )
            x += dx
    image = cv2.warpPerspective(src=np.array(image), M=M, dsize=(width, height))
    if draw_contour:
        image = cv2.drawContours(
            image,
            contours=[permitted_contour.reshape((-1, 1, 2)).astype("int32")],
            contourIdx=0,
            color=(255, 0, 0, 255),
            thickness=int(width / 100),
        )
    lines_stripped = _strip_lines(lines_raw)
    lines_transformed = [
        [
            (cv2.perspectiveTransform(src=coords[np.newaxis], m=M)[0], character)
            for coords, character in line
        ]
        for line in lines_stripped
    ]
    return image, lines_transformed


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
    basis = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1).reshape(
        (-1, 2)
    )
    basis *= spacing
    slots_pretransform = np.concatenate(
        [
            (basis + offset)[:, np.newaxis, :]
            for offset in [[0, 0], [spacing, 0], [spacing, spacing], [0, spacing]]
        ],
        axis=1,
    )
    slots = cv2.perspectiveTransform(
        src=slots_pretransform.reshape((1, -1, 2)).astype("float32"), m=M
    )[0]
    inside = (
        np.array(
            [
                cv2.pointPolygonTest(contour=contour, pt=(int(x), int(y)), measureDist=False) >= 0
                for x, y in slots
            ]
        )
        .reshape(-1, 4)
        .all(axis=1)
    )
    slots = slots.reshape(-1, 4, 2)
    areas = (
        np.abs(
            (slots[:, 0, 0] * slots[:, 1, 1] - slots[:, 0, 1] * slots[:, 1, 0])
            + (slots[:, 1, 0] * slots[:, 2, 1] - slots[:, 1, 1] * slots[:, 2, 0])
            + (slots[:, 2, 0] * slots[:, 3, 1] - slots[:, 2, 1] * slots[:, 3, 0])
            + (slots[:, 3, 0] * slots[:, 0, 1] - slots[:, 3, 1] * slots[:, 0, 0])
        )
        / 2
    )
    slots_filtered = slots_pretransform[(areas > minarea * spacing * spacing) & inside]
    temporary_image = cv2.drawContours(
        image=np.zeros((height, width), dtype="uint8"),
        contours=slots_filtered,
        contourIdx=-1,
        color=255,
    )
    temporary_image = cv2.dilate(
        src=temporary_image, kernel=np.ones((spacing, spacing))
    )
    newContours, _ = cv2.findContours(
        temporary_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )
    x, y = slots_filtered[0][0]
    contour = newContours[
        next(
            index
            for index, contour in enumerate(newContours)
            if cv2.pointPolygonTest(contour=contour, pt=(int(x), int(y)), measureDist=False) >= 0
        )
    ][:, 0, :]
    return contour


def get_image_generator(
    height,
    width,
    font_groups,
    text_generator,
    font_size: typing.Union[int, typing.Tuple[int, int]] = 18,
    backgrounds: typing.List[typing.Union[str, np.ndarray]] = None,
    background_crop_mode="crop",
    rotationX: typing.Union[int, typing.Tuple[int, int]] = 0,
    rotationY: typing.Union[int, typing.Tuple[int, int]] = 0,
    rotationZ: typing.Union[int, typing.Tuple[int, int]] = 0,
    margin=0,
    use_ligatures=False,
    augmenter=None,
    draw_contour=False,
    draw_contour_text=False,
):
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

    Yields:
        Tuples of (image, lines) where image is the
        transparent text image and lines is a list of lines
        where each line itself is a list of (box, character) tuples and
        box is an array of points with shape (4, 2) providing the coordinates
        of the character box in clockwise order starting from the top left.
    """
    if backgrounds is None:
        backgrounds = [np.zeros((height, width, 3), dtype="uint8")]
    alphabet = "".join(font_groups.keys())
    assert len(set(alphabet)) == len(
        alphabet
    ), "Each character can appear in the subalphabet for only one font group."
    for text, background_index, current_font_groups in zip(
        text_generator,
        itertools.cycle(range(len(backgrounds))),
        zip(
            *[
                itertools.cycle(
                    [
                        (subalphabet, font_filepath)
                        for font_filepath in font_group_filepaths
                    ]
                )
                for subalphabet, font_group_filepaths in font_groups.items()
            ]
        ),
    ):
        if background_index == 0:
            random.shuffle(backgrounds)
        current_font_groups = dict(current_font_groups)
        current_font_size = (
            np.random.randint(low=font_size[0], high=font_size[1])
            if isinstance(font_size, tuple)
            else font_size
        )
        current_rotation_X, current_rotation_Y, current_rotation_Z = [
            (
                np.random.uniform(low=rotation[0], high=rotation[1])
                if isinstance(rotation, tuple)
                else rotation
            )
            * np.pi
            / 180
            for rotation in [rotationX, rotationY, rotationZ]
        ]
        current_background_filepath_or_array = backgrounds[background_index]
        current_background = (
            tools.read(current_background_filepath_or_array)
            if isinstance(current_background_filepath_or_array, str)
            else current_background_filepath_or_array
        )
        if augmenter is not None:
            current_background = augmenter(images=[current_background])[0]
        if (
            current_background.shape[0] != height
            or current_background.shape[1] != width
        ):
            current_background = tools.fit(
                current_background,
                width=width,
                height=height,
                mode=background_crop_mode,
            )
        permitted_contour, isDark = get_maximum_uniform_contour(
            image=current_background, fontsize=current_font_size, margin=margin
        )
        if permitted_contour is None:
            # We can't draw on this background. Boo!
            continue
        random_color_values = np.random.randint(low=0, high=50, size=3)
        text_color = (
            tuple(np.array([255, 255, 255]) - random_color_values)
            if isDark
            else tuple(random_color_values)
        )
        text_image, lines = draw_text_image(
            text=text,
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
            draw_contour=draw_contour_text,
        )
        alpha = text_image[..., -1:].astype("float32") / 255
        image = (alpha * text_image[..., :3] + (1 - alpha) * current_background).astype(
            "uint8"
        )
        if draw_contour:
            image = cv2.drawContours(
                image,
                contours=[permitted_contour.reshape((-1, 1, 2)).astype("int32")],
                contourIdx=0,
                color=(255, 0, 0),
                thickness=int(width / 100),
            )
        yield image, lines
