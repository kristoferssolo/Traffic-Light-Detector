#!/usr/bin/env python3
import argparse

from loguru import logger
from paths import create_dirs, IMAGES_IN_PATH


parser = argparse.ArgumentParser(description="Traffic light detection script.")
parser.add_argument(
    "-i",
    "--image",
    action="store_true",
    help="detects traffic lights in images located in ./assets/images_in/",
)
parser.add_argument(
    "-w",
    "--webcam",
    action="store_true",
    help="reads webcam inputs to determine traffic light color",
)
parser.add_argument(
    "-a",
    "--audio",
    action="store_true",
    help="plays audio when green light is detected",
)


@logger.catch
def main(args) -> None:
    create_dirs()

    if args.image:
        from TrafficLightDetector.traffic_light_images import TrafficLightDetectorImages
        for path in IMAGES_IN_PATH.iterdir():
            image = TrafficLightDetectorImages(path)
            image.draw()

    if args.webcam:
        from TrafficLightDetector.traffic_light_webcam import TrafficLightDetectorWebcam
        camera = TrafficLightDetectorWebcam(0)  # Change number if webcam didn't detect
        camera.enable()

    if args.audio:
        from TrafficLightDetector.traffic_light_webcam import TrafficLightDetectorWebcam
        camera = TrafficLightDetectorWebcam(0, sound=True)  # Change number if webcam didn't detect
        camera.enable()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
