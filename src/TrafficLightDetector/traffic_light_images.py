from pathlib import Path

import cv2
from paths import IMAGES_OUT_PATH
from TrafficLightDetector.traffic_light_detector import TrafficLightDetector


class TrafficLightDetectorImages(TrafficLightDetector):

    def __init__(self, path: Path) -> None:
        self.path = path
        image = cv2.imread(str(path))
        self._set_image(image)

    def _save_image(self) -> None:
        cv2.imwrite(str(IMAGES_OUT_PATH.joinpath(self.path.name)), self.image_copy)

    def draw(self) -> None:
        self._draw_circle()
        self._save_image()
