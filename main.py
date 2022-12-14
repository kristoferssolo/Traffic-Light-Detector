#!/usr/bin/env python3
import argparse
import importlib

from detector.paths import BASE_PATH, create_dirs, LOGS_PATH
from loguru import logger

log_level = "DEBUG" if BASE_PATH.joinpath("debug").exists() else "INFO"


# Set up logging
logger.add(LOGS_PATH.joinpath("detection.log"), format="{time} | {level} | {message}", level=log_level, rotation="1 MB", compression="zip")


parser = argparse.ArgumentParser(description="Traffic light detection script.")
parser.add_argument(
    "-e",
    "--extract",
    action="store_true",
    help="extracts and crops traffic light images from given images in ./assets/extraction/input/ to ./assets/extraction/cropped/",
)
parser.add_argument(
    "-t",
    "--train",
    action="store_true",
    help="trains model.",
)
parser.add_argument(
    "-i",
    "--image",
    action="store_true",
    help="detects traffic lights in images located in ./assets/detection/images_in/",
)
# parser.add_argument(
#     "-v",
#     "--video",
#     action="store_true",
#     help="detects traffic lights in videos located in ./assets/detection/videos_in/",
# )


@logger.catch
def main(args) -> None:
    create_dirs()
    if args.extract:
        module = importlib.import_module("detector.extract_traffic_lights")
        module.extract_traffic_lights()
    if args.train:
        module = importlib.import_module("detector.train_traffic_light_color")
        module.train_traffic_light_color()
    if args.image:
        module = importlib.import_module("detector.detect_traffic_light_color_image")
        module.detect_traffic_light_color_image()
    # if args.video:
    #     module = importlib.import_module("detector.detect_traffic_light_color_video")
    #     module.detect_traffic_light_color_video()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
