# pylint: disable=line-too-long,invalid-name,too-many-arguments,too-many-locals
import concurrent.futures
import itertools
import warnings
import typing
import zipfile
import random
import glob
import json
import os

import tqdm
import imgaug
import PIL.Image
import numpy as np

from . import tools


def _read_born_digital_labels_file(labels_filepath, image_folder):
    """Read a labels file and return (filepath, label) tuples.

    Args:
        labels_filepath: Path to labels file
        image_folder: Path to folder containing images
    """
    with open(labels_filepath, encoding="utf-8-sig") as f:
        labels_raw = [l.strip().split(",") for l in f.readlines()]
        labels = [
            (
                os.path.join(image_folder, segments[0]),
                None,
                ",".join(segments[1:]).strip()[1:-1],
            )
            for segments in labels_raw
        ]
    return labels


def get_cocotext_recognizer_dataset(
    split="train",
    cache_dir=None,
    limit=None,
    legible_only=False,
    english_only=False,
    return_raw_labels=False,
):
    """Get a list of (filepath, box, word) tuples from the
    COCO-Text dataset.

    Args:
        split: Which split to get (train, val, or trainval)
        limit: Limit the number of files included in the download
        cache_dir: The directory in which to cache the file. The default is
            `~/.keras-ocr`.
        return_raw_labels: Whether to return the raw labels object

    Returns:
        A recognition dataset as a list of (filepath, box, word) tuples.
        If return_raw_labels is True, you will also get a (labels, images_dir)
        tuple containing the raw COCO data and the directory in which you
        can find the images.
    """
    assert split in ["train", "val", "trainval"], f"Unsupported split: {split}"
    if cache_dir is None:
        cache_dir = tools.get_default_cache_dir()
    main_dir = os.path.join(cache_dir, "coco-text")
    images_dir = os.path.join(main_dir, "images")
    labels_zip = tools.download_and_verify(
        url="https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip",
        cache_dir=main_dir,
        sha256="1444893ce7dbcd8419b2ec9be6beb0dba9cf8a43bf36cab4293d5ba6cecb7fb1",
    )
    with zipfile.ZipFile(labels_zip) as z:
        with z.open("cocotext.v2.json") as f:
            labels = json.loads(f.read())
    selected_ids = [
        cocoid for cocoid, data in labels["imgs"].items() if data["set"] in split
    ]
    if limit:
        flatten = lambda l: [item for sublist in l for item in sublist]
        selected_ids = selected_ids[:limit]
        labels["imgToAnns"] = {
            k: v for k, v in labels["imgToAnns"].items() if k in selected_ids
        }
        labels["imgs"] = {k: v for k, v in labels["imgs"].items() if k in selected_ids}
        anns = set(flatten(list(labels.values())))
        labels["anns"] = {k: v for k, v in labels["anns"].items() if k in anns}
    selected_filenames = [
        labels["imgs"][cocoid]["file_name"] for cocoid in selected_ids
    ]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(
                [
                    executor.submit(
                        tools.download_and_verify,
                        url=f"http://images.cocodataset.org/train2014/{filename}",
                        cache_dir=images_dir,
                        verbose=False,
                    )
                    for filename in selected_filenames
                ]
            ),
            total=len(selected_filenames),
            desc="Downloading images",
        ):
            _ = future.result()
    dataset = []
    for selected_id in selected_ids:
        filepath = os.path.join(
            images_dir, selected_filenames[selected_ids.index(selected_id)]
        )
        for annIdx in labels["imgToAnns"][selected_id]:
            ann = labels["anns"][str(annIdx)]
            if english_only and ann["language"] != "english":
                continue
            if legible_only and ann["legibility"] != "legible":
                continue
            dataset.append(
                (filepath, np.array(ann["mask"]).reshape(-1, 2), ann["utf8_string"])
            )
    if return_raw_labels:
        return dataset, (labels, images_dir)
    return dataset


def get_born_digital_recognizer_dataset(split="train", cache_dir=None):
    """Get a list of (filepath, box, word) tuples from the
    BornDigital dataset. This dataset comes pre-cropped so
    `box` is always `None`.

    Args:
        split: Which split to get (train, test, or traintest)
        cache_dir: The directory in which to cache the file. The default is
            `~/.keras-ocr`.

    Returns:
        A recognition dataset as a list of (filepath, box, word) tuples
    """
    data = []
    if cache_dir is None:
        cache_dir = tools.get_default_cache_dir()
    main_dir = os.path.join(cache_dir, "borndigital")
    assert split in ["train", "traintest", "test"], f"Unsupported split: {split}"
    if split in ["train", "traintest"]:
        train_dir = os.path.join(main_dir, "train")
        training_zip_path = tools.download_and_verify(
            url="https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/Challenge1_Training_Task3_Images_GT.zip",  # pylint: disable=line-too-long
            filename="Challenge1_Training_Task3_Images_GT.zip",
            cache_dir=main_dir,
            sha256="8ede0639f5a8031d584afd98cee893d1c5275d7f17863afc2cba24b13c932b07",
        )
        if (
            len(
                glob.glob(os.path.join(train_dir, "*.png"))
                + glob.glob(os.path.join(train_dir, "*.txt"))
            )
            != 3568
        ):
            with zipfile.ZipFile(training_zip_path) as zfile:
                zfile.extractall(train_dir)
        data.extend(
            _read_born_digital_labels_file(
                labels_filepath=os.path.join(train_dir, "gt.txt"),
                image_folder=train_dir,
            )
        )
    if split in ["test", "traintest"]:
        test_dir = os.path.join(main_dir, "test")
        test_zip_path = tools.download_and_verify(
            url="https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/Challenge1_Test_Task3_Images.zip",
            filename="Challenge1_Test_Task3_Images.zip",
            cache_dir=main_dir,
            sha256="8f781b0140fd0bac3750530f0924bce5db3341fd314a2fcbe9e0b6ca409a77f0",
        )
        if len(glob.glob(os.path.join(test_dir, "*.png"))) != 1439:
            with zipfile.ZipFile(test_zip_path) as zfile:
                zfile.extractall(test_dir)
        test_gt_path = tools.download_and_verify(
            url="https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/Challenge1_Test_Task3_GT.txt",
            cache_dir=test_dir,
            filename="Challenge1_Test_Task3_GT.txt",
            sha256="fce7f1228b7c4c26a59f13f562085148acf063d6690ce51afc395e0a1aabf8be",
        )
        data.extend(
            _read_born_digital_labels_file(
                labels_filepath=test_gt_path, image_folder=test_dir
            )
        )
    return data


def get_icdar_2013_recognizer_dataset(cache_dir=None):
    """Get a list of (filepath, box, word) tuples from the
    ICDAR 2013 dataset.

    Args:
        cache_dir: The directory in which to cache the file. The default is
            `~/.keras-ocr`.

    Returns:
        A recognition dataset as a list of (filepath, box, word) tuples
    """
    dataset = []
    for image_path, lines, _ in get_icdar_2013_detector_dataset(
        cache_dir=cache_dir, skip_illegible=True
    ):
        for line in lines:
            box, text = tools.combine_line(line)
            dataset.append((image_path, box, text))
    return dataset


def get_icdar_2013_detector_dataset(cache_dir=None, skip_illegible=False):
    """Get the ICDAR 2013 text segmentation dataset for detector
    training. Only the training set has the necessary annotations.
    For the test set, only segmentation maps are provided, which
    do not provide the necessary information for affinity scores.

    Args:
        cache_dir: The directory in which to store the data.
        skip_illegible: Whether to skip illegible characters.

    Returns:
        Lists of (image_path, lines, confidence) tuples. Confidence
        is always 1 for this dataset. We record confidence to allow
        for future support for weakly supervised cases.
    """
    if cache_dir is None:
        cache_dir = tools.get_default_cache_dir()
    main_dir = os.path.join(cache_dir, "icdar2013")
    training_images_dir = os.path.join(main_dir, "Challenge2_Training_Task12_Images")
    training_zip_images_path = tools.download_and_verify(
        url="https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/Challenge2_Training_Task12_Images.zip",  # pylint: disable=line-too-long
        cache_dir=main_dir,
        filename="Challenge2_Training_Task12_Images.zip",
        sha256="7a57d1699fbb92db3ad82c930202938562edaf72e1c422ddd923860d8ace8ded",
    )
    if len(glob.glob(os.path.join(training_images_dir, "*.jpg"))) != 229:
        with zipfile.ZipFile(training_zip_images_path) as zfile:
            zfile.extractall(training_images_dir)
    training_gt_dir = os.path.join(main_dir, "Challenge2_Training_Task2_GT")
    training_zip_gt_path = tools.download_and_verify(
        url="https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/Challenge2_Training_Task2_GT.zip",  # pylint: disable=line-too-long
        cache_dir=main_dir,
        filename="Challenge2_Training_Task2_GT.zip",
        sha256="4cedd5b1e33dc4354058f5967221ac85dbdf91a99b30f3ab1ecdf42786a9d027",
    )
    if len(glob.glob(os.path.join(training_gt_dir, "*.txt"))) != 229:
        with zipfile.ZipFile(training_zip_gt_path) as zfile:
            zfile.extractall(training_gt_dir)

    dataset = []
    for gt_filepath in glob.glob(os.path.join(training_gt_dir, "*.txt")):
        image_id = os.path.split(gt_filepath)[1].split("_")[0]
        image_path = os.path.join(training_images_dir, image_id + ".jpg")
        lines = []
        with open(gt_filepath, "r", encoding="utf8") as f:
            current_line: typing.List[typing.Tuple[np.ndarray, str]] = []
            for raw_row in f.read().split("\n"):
                if raw_row == "":
                    lines.append(current_line)
                    current_line = []
                else:
                    row = raw_row.split(" ")[5:]
                    character = row[-1][1:-1]
                    if character == "" and skip_illegible:
                        continue
                    x1, y1, x2, y2 = map(int, row[:4])
                    current_line.append(
                        (np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), character)
                    )
        # Some lines only have illegible characters and if skip_illegible is True,
        # then these lines will be blank.
        lines = [line for line in lines if line]
        dataset.append((image_path, lines, 1))
    return dataset


def get_icdar_2019_semisupervised_dataset(cache_dir=None):
    """EXPERIMENTAL. Get a semisupervised labeled version
    of the ICDAR 2019 dataset. Only images with Latin-only
    scripts are available at this time.

    Args:
        cache_dir: The cache directory to use.
    """
    warnings.warn(
        "You may need to get this dataset manually in-browser by downloading "
        "https://www.mediafire.com/file/snekaezeextc3ee/ImagesPart1.zip/file "
        "and https://www.mediafire.com/file/i2snljkfm4t2ojm/ImagesPart2.zip/file "
        "and putting them in ~/.keras-ocr/icdar2019. The files are too big "
        "for GitHub Releases and we may run out of direct download  bandwidth on "
        "MediaFire where they are hosted. See "
        "https://github.com/faustomorales/keras-ocr/issues/117 for more details.",
        UserWarning,
    )
    if cache_dir is None:
        cache_dir = tools.get_default_cache_dir()
    main_dir = os.path.join(cache_dir, "icdar2019")
    training_dir_1 = os.path.join(main_dir, "ImagesPart1")
    training_dir_2 = os.path.join(main_dir, "ImagesPart2")
    if len(glob.glob(os.path.join(training_dir_1, "*"))) != 5000:
        training_zip_1 = tools.download_and_verify(
            url="https://www.mediafire.com/file/snekaezeextc3ee/ImagesPart1.zip/file",  # pylint: disable=line-too-long
            cache_dir=main_dir,
            filename="ImagesPart1.zip",
            sha256="1968894ef93b97f3ef4c97880b6dce85b1851f4d778e253f4e7265b152a4986f",
        )
        with zipfile.ZipFile(training_zip_1) as zfile:
            zfile.extractall(main_dir)
    if len(glob.glob(os.path.join(training_dir_2, "*"))) != 5000:
        training_zip_2 = tools.download_and_verify(
            url="https://www.mediafire.com/file/i2snljkfm4t2ojm/ImagesPart2.zip/file",  # pylint: disable=line-too-long
            cache_dir=main_dir,
            filename="ImagesPart2.zip",
            sha256="5651b9137e877f731bfebb2a8b75042e26baa389d2fb1cfdbb9e3da343757241",
        )
        with zipfile.ZipFile(training_zip_2) as zfile:
            zfile.extractall(main_dir)
    ground_truth = tools.download_and_verify(
        url="https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/mlt2019_dataset.json",  # pylint: disable=line-too-long
        cache_dir=main_dir,
        filename="mlt2019_dataset.json",
    )
    with open(ground_truth, "r", encoding="utf8") as f:
        character_level_dataset = json.loads(f.read())["dataset"]
    for gif_filepath in glob.glob(os.path.join(main_dir, "**", "*.gif")):
        # We need to do this because we cannot easily read GIFs.
        PIL.Image.open(gif_filepath).convert("RGB").save(
            os.path.splitext(gif_filepath)[0] + ".jpg"
        )
        os.remove(gif_filepath)
    return [
        (
            os.path.join(main_dir, entry["filepath"]),
            [
                [(np.array(box).clip(0, np.inf), None) for box in line["line"]]
                for line in entry["lines"]
                if line["line"]
            ],
            entry["percent_complete"],
        )
        for entry in character_level_dataset
        if entry["percent_complete"] > 0.5
    ]


def get_detector_image_generator(
    labels,
    width,
    height,
    augmenter=None,
    area_threshold=0.5,
    focused=False,
    min_area=None,
    shuffle=True,
):
    """Generated augmented (image, lines) tuples from a list
    of (filepath, lines, confidence) tuples. Confidence is
    not used right now but is included for a future release
    that uses semi-supervised data.

    Args:
        labels: A list of (image, lines, confience) tuples.
        augmenter: An augmenter to apply to the images.
        width: The width to use for output images
        height: The height to use for output images
        area_threshold: The area threshold to use to keep
            characters in augmented images.
        min_area: The minimum area for a character to be
            included.
        focused: Whether to pre-crop images to width/height containing
            a region containing text.
        shuffle: Whether to shuffle the data on each iteration.
    """
    labels = labels.copy()
    for index in itertools.cycle(range(len(labels))):
        if index == 0 and shuffle:
            random.shuffle(labels)
        image_filepath, lines, confidence = labels[index]
        image = tools.read(image_filepath)
        if augmenter is not None:
            image, lines = tools.augment(
                boxes=lines,
                boxes_format="lines",
                image=image,
                area_threshold=area_threshold,
                min_area=min_area,
                augmenter=augmenter,
            )
        if focused:
            boxes = [tools.combine_line(line)[0] for line in lines]
            if boxes:
                selected = np.array(boxes[np.random.choice(len(boxes))])
                left, top = selected.min(axis=0).clip(0, np.inf).astype("int")
                if left > 0:
                    left -= np.random.randint(0, min(left, width / 2))
                if top > 0:
                    top -= np.random.randint(0, min(top, height / 2))
                image, lines = tools.augment(
                    boxes=lines,
                    augmenter=imgaug.augmenters.Sequential(
                        [
                            imgaug.augmenters.Crop(px=(int(top), 0, 0, int(left))),
                            imgaug.augmenters.CropToFixedSize(
                                width=width, height=height, position="right-bottom"
                            ),
                        ]
                    ),
                    boxes_format="lines",
                    image=image,
                    min_area=min_area,
                    area_threshold=area_threshold,
                )
        image, scale = tools.fit(
            image, width=width, height=height, mode="letterbox", return_scale=True
        )
        lines = tools.adjust_boxes(boxes=lines, boxes_format="lines", scale=scale)
        yield image, lines, confidence


def get_recognizer_image_generator(
    labels, height, width, alphabet, augmenter=None, shuffle=True
):
    """Generate augmented (image, text) tuples from a list
    of (filepath, box, label) tuples.

    Args:
        labels: A list of (filepath, box, label) tuples
        height: The height of the images to return
        width: The width of the images to return
        alphabet: The alphabet which limits the characters returned
        augmenter: The augmenter to apply to images
        shuffle: Whether to shuffle the dataset on each iteration
    """
    n_with_illegal_characters = sum(
        any(c not in alphabet for c in text) for _, _, text in labels
    )
    if n_with_illegal_characters > 0:
        print(
            f"{n_with_illegal_characters} / {len(labels)} instances have illegal characters."
        )
    labels = labels.copy()
    for index in itertools.cycle(range(len(labels))):
        if index == 0 and shuffle:
            random.shuffle(labels)
        filepath, box, text = labels[index]
        cval = typing.cast(
            int, np.random.randint(low=0, high=255, size=3).astype("uint8")
        )
        if box is not None:
            image = tools.warpBox(
                image=tools.read(filepath),
                box=box.astype("float32"),
                target_height=height,
                target_width=width,
                cval=cval,
            )
        else:
            image = tools.read_and_fit(
                filepath_or_array=filepath, width=width, height=height, cval=cval
            )
        text = "".join([c for c in text if c in alphabet])
        if not text:
            continue
        if augmenter:
            image = augmenter.augment_image(image)
        yield (image, text)
