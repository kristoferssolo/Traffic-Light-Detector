#!/usr/bin/env python3
import argparse

from loguru import logger

from TrafficLightDetector.paths import IMAGES_IN_PATH, create_dirs
from TrafficLightDetector.traffic_light_camera import \
    TrafficLightDetectorCamera
from TrafficLightDetector.traffic_light_images import \
    TrafficLightDetectorImages


def pos_int(string: str) -> int:
    try:
        value = int(string)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected integer, got {string!r}")
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected non negative number, got {value}")
    return value


parser = argparse.ArgumentParser(description="Traffic light detection script.")
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "-i",
    "--image",
    action="store_true",
    help="Detects traffic lights in images located in ./assets/images_in/",
)
group.add_argument(
    "-c",
    "--camera",
    type=pos_int,
    nargs="?",
    const=0,
    metavar="int",
    help="Reads camera inputs to determine traffic light color. (Default: %(default)s)",
)
parser.add_argument(
    "-s",
    "--sound",
    action="store_true"
)


@logger.catch
def main(args) -> None:
    create_dirs()

    if args.image:
        for path in IMAGES_IN_PATH.iterdir():
            image = TrafficLightDetectorImages(path)
            image.draw()

    if args.camera is not None:
        camera = TrafficLightDetectorCamera(args.camera, sound=args.sound)  # Change number if webcam didn't detect
        camera.enable()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
