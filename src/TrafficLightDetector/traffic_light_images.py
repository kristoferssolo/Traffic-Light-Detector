from pathlib import Path

import cv2

from TrafficLightDetector.paths import IMAGES_OUT_PATH
from TrafficLightDetector.traffic_light_detector import TrafficLightDetector


class TrafficLightDetectorImages(TrafficLightDetector):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._set_image(cv2.imread(str(self.path)))

    def _save_image(self) -> None:
        cv2.imwrite(str(IMAGES_OUT_PATH.joinpath(self.path.name)), self.image)

    def draw(self) -> None:
        self._outline_traffic_lights()
        self._save_image()
