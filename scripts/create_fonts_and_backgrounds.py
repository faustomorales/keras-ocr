"""This script is what was used to generate the
backgrounds.zip and fonts.zip files.
"""
# pylint: disable=invalid-name,redefined-outer-name
import json
import urllib.request
import urllib.parse
import concurrent
import shutil
import zipfile
import glob
import os

import numpy as np
import tqdm
import cv2

import keras_ocr

if __name__ == "__main__":
    fonts_commit = "a0726002eab4639ee96056a38cd35f6188011a81"
    fonts_sha256 = "e447d23d24a5bbe8488200a058cd5b75b2acde525421c2e74dbfb90ceafce7bf"
    fonts_source_zip_filepath = keras_ocr.tools.download_and_verify(
        url=f"https://github.com/google/fonts/archive/{fonts_commit}.zip",
        cache_dir=".",
        sha256=fonts_sha256,
    )
    shutil.rmtree("fonts-raw", ignore_errors=True)
    with zipfile.ZipFile(fonts_source_zip_filepath) as zfile:
        zfile.extractall(path="fonts-raw")

    retained_fonts = []
    sha256s = []
    basenames = []
    # The blacklist includes fonts that, at least for the English alphabet, were found
    # to be illegible (e.g., thin fonts) or render in unexpected ways (e.g., mathematics
    # fonts).
    blacklist = [
        "AlmendraDisplay-Regular.ttf",
        "RedactedScript-Bold.ttf",
        "RedactedScript-Regular.ttf",
        "Sevillana-Regular.ttf",
        "Mplus1p-Thin.ttf",
        "Stalemate-Regular.ttf",
        "jsMath-cmsy10.ttf",
        "Codystar-Regular.ttf",
        "AdventPro-Thin.ttf",
        "RoundedMplus1c-Thin.ttf",
        "EncodeSans-Thin.ttf",
        "AlegreyaSans-ThinItalic.ttf",
        "AlegreyaSans-Thin.ttf",
        "FiraSans-Thin.ttf",
        "FiraSans-ThinItalic.ttf",
        "WorkSans-Thin.ttf",
        "Tomorrow-ThinItalic.ttf",
        "Tomorrow-Thin.ttf",
        "Italianno-Regular.ttf",
        "IBMPlexSansCondensed-Thin.ttf",
        "IBMPlexSansCondensed-ThinItalic.ttf",
        "Lato-ExtraLightItalic.ttf",
        "LibreBarcode128Text-Regular.ttf",
        "LibreBarcode39-Regular.ttf",
        "LibreBarcode39ExtendedText-Regular.ttf",
        "EncodeSansExpanded-ExtraLight.ttf",
        "Exo-Thin.ttf",
        "Exo-ThinItalic.ttf",
        "DrSugiyama-Regular.ttf",
        "Taviraj-ThinItalic.ttf",
        "SixCaps.ttf",
        "IBMPlexSans-Thin.ttf",
        "IBMPlexSans-ThinItalic.ttf",
        "AdobeBlank-Regular.ttf",
        "FiraSansExtraCondensed-ThinItalic.ttf",
        "HeptaSlab[wght].ttf",
        "Karla-Italic[wght].ttf",
        "Karla[wght].ttf",
        "RalewayDots-Regular.ttf",
        "FiraSansCondensed-ThinItalic.ttf",
        "jsMath-cmex10.ttf",
        "LibreBarcode39Text-Regular.ttf",
        "LibreBarcode39Extended-Regular.ttf",
        "EricaOne-Regular.ttf",
        "ArimaMadurai-Thin.ttf",
        "IBMPlexSerif-ExtraLight.ttf",
        "IBMPlexSerif-ExtraLightItalic.ttf",
        "IBMPlexSerif-ThinItalic.ttf",
        "IBMPlexSerif-Thin.ttf",
        "Exo2-Thin.ttf",
        "Exo2-ThinItalic.ttf",
        "BungeeOutline-Regular.ttf",
        "Redacted-Regular.ttf",
        "JosefinSlab-ThinItalic.ttf",
        "GothicA1-Thin.ttf",
        "Kanit-ThinItalic.ttf",
        "Kanit-Thin.ttf",
        "AlegreyaSansSC-ThinItalic.ttf",
        "AlegreyaSansSC-Thin.ttf",
        "Chathura-Thin.ttf",
        "Blinker-Thin.ttf",
        "Italiana-Regular.ttf",
        "Miama-Regular.ttf",
        "Grenze-ThinItalic.ttf",
        "LeagueScript-Regular.ttf",
        "BigShouldersDisplay-Thin.ttf",
        "YanoneKaffeesatz[wght].ttf",
        "BungeeHairline-Regular.ttf",
        "JosefinSans-Thin.ttf",
        "JosefinSans-ThinItalic.ttf",
        "Monofett.ttf",
        "Raleway-ThinItalic.ttf",
        "Raleway-Thin.ttf",
        "JosefinSansStd-Light.ttf",
        "LibreBarcode128-Regular.ttf",
    ]
    for filepath in tqdm.tqdm(
        sorted(glob.glob("fonts-raw/**/**/**/*.ttf")), desc="Filtering fonts."
    ):
        sha256 = keras_ocr.tools.sha256sum(filepath)
        basename = os.path.basename(filepath)
        # We check the sha256 and filenames because some of the fonts
        # in the repository are duplicated (see TRIVIA.md).
        if sha256 in sha256s or basename in basenames or basename in blacklist:
            continue
        sha256s.append(sha256)
        basenames.append(basename)
        retained_fonts.append(filepath)
    retained_font_families = set(
        [filepath.split(os.sep)[-2] for filepath in retained_fonts]
    )
    added = []
    with zipfile.ZipFile(file="fonts.zip", mode="w") as zfile:
        for font_family in tqdm.tqdm(retained_font_families, desc="Saving ZIP file."):
            # We want to keep all the metadata files plus
            # the retained font files. And we don't want
            # to add the same file twice.
            files = [
                input_filepath
                for input_filepath in glob.glob(f"fonts-raw/**/**/{font_family}/*")
                if input_filepath not in added
                and (
                    input_filepath in retained_fonts
                    or os.path.splitext(input_filepath)[1] != ".ttf"
                )
            ]
            added.extend(files)
            for input_filepath in files:
                zfile.write(
                    filename=input_filepath,
                    arcname=os.path.join(*input_filepath.split(os.sep)[-2:]),
                )
    print("Finished saving fonts file.")

    # pylint: disable=line-too-long
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers&gcmtype=file&format=json"
        "&gcmtitle=Category:Featured_pictures_on_Wikimedia_Commons&prop=imageinfo&gcmlimit=50&iiprop=url&iiurlwidth=1024"
    )
    gcmcontinue = None
    max_responses = 300
    responses = []
    for responseCount in tqdm.tqdm(range(max_responses)):
        current_url = url
        if gcmcontinue is not None:
            current_url += f"&continue=gcmcontinue||&gcmcontinue={gcmcontinue}"
        with urllib.request.urlopen(url=current_url) as response:
            current = json.loads(response.read())
            responses.append(current)
            gcmcontinue = (
                None
                if "continue" not in current
                else current["continue"]["gcmcontinue"]
            )
        if gcmcontinue is None:
            break
    print("Finished getting list of images.")

    # We want to avoid animated images as well as icon files.
    image_urls = []
    for response in responses:
        image_urls.extend(
            [
                page["imageinfo"][0]["thumburl"]
                for page in response["query"]["pages"].values()
            ]
        )
    image_urls = [url for url in image_urls if url.lower().endswith(".jpg")]
    shutil.rmtree("backgrounds", ignore_errors=True)
    os.makedirs("backgrounds")
    assert len(image_urls) == len(set(image_urls)), "Duplicates found!"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                keras_ocr.tools.download_and_verify,
                url=url,
                cache_dir="./backgrounds",
                verbose=False,
            )
            for url in image_urls
        ]
        for _ in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            pass
    for filepath in glob.glob("backgrounds/*.JPG"):
        os.rename(filepath, filepath.lower())

    print("Filtering images by aspect ratio and maximum contiguous contour.")
    image_paths = np.array(sorted(glob.glob("backgrounds/*.jpg")))

    def compute_metrics(filepath):
        image = keras_ocr.tools.read(filepath)
        aspect_ratio = image.shape[0] / image.shape[1]
        contour, _ = keras_ocr.tools.get_maximum_uniform_contour(image, fontsize=40)
        area = cv2.contourArea(contour) if contour is not None else 0
        return aspect_ratio, area

    metrics = np.array(
        [compute_metrics(filepath) for filepath in tqdm.tqdm(image_paths)]
    )
    filtered_paths = image_paths[
        (metrics[:, 0] < 3 / 2) & (metrics[:, 0] > 2 / 3) & (metrics[:, 1] > 1e6)
    ]
    detector = keras_ocr.detection.Detector()
    paths_with_text = [
        filepath
        for filepath in tqdm.tqdm(filtered_paths)
        if len(
            detector.detect(
                images=[keras_ocr.tools.read_and_fit(filepath, width=640, height=640)]
            )[0]
        )
        > 0
    ]
    filtered_paths = np.array(
        [path for path in filtered_paths if path not in paths_with_text]
    )
    filtered_basenames = list(map(os.path.basename, filtered_paths))
    basename_to_url = {
        os.path.basename(urllib.parse.urlparse(url).path).lower(): url
        for url in image_urls
    }
    filtered_urls = [
        basename_to_url[basename.lower()] for basename in filtered_basenames
    ]
    assert len(filtered_urls) == len(filtered_paths)
    removed_paths = [
        filepath for filepath in image_paths if filepath not in filtered_paths
    ]
    for filepath in removed_paths:
        os.remove(filepath)
    with open("backgrounds/urls.txt", "w") as f:
        f.write("\n".join(filtered_urls))
    with zipfile.ZipFile(file="backgrounds.zip", mode="w") as zfile:
        for filepath in tqdm.tqdm(
            filtered_paths.tolist() + ["backgrounds/urls.txt"], desc="Saving ZIP file."
        ):
            zfile.write(filename=filepath, arcname=os.path.basename(filepath.lower()))
