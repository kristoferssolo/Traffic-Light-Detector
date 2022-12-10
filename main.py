#!/bin/python3.10
import sys

from detector.paths import create_dirs

ARGS = """
Usage: main.py [OPTION]

Options:
    -h --help       Displays this list
    -e --extract    Excracts and cropps traffic light images from given images in ./assets/exctraction/input/ to ./assets/exctraction/cropped/
    -t --train      Trains model
    -i --image      Detecs traffic lights in images located in ./assets/detection/images_in/
    -v --video      Detecs traffic lights in videos located in ./assets/detection/videos_in/
"""


def main(argv) -> None:
    create_dirs()
    for arg in argv:
        if arg in ("-h", "--help"):
            print(ARGS)
            sys.exit()
        elif arg in ("-e", "--extract"):
            from detector.extract_traffic_lights import extract_traffic_lights
            extract_traffic_lights()
        elif arg in ("-t", "--train"):
            from detector.train_traffic_light_color import train_traffic_light_color
            train_traffic_light_color()
        elif arg in ("-i", "--image"):
            from detector.detect_traffic_light_color_image import (
                detect_traffic_light_color_image,
            )
            detect_traffic_light_color_image()
        elif arg in ("-v", "--video"):
            from detector.detect_traffic_light_color_video import (
                detect_traffic_light_color_video,
            )
            detect_traffic_light_color_video()


if __name__ == "__main__":
    main(sys.argv[1:])
