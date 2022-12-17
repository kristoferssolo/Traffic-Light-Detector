from pathlib import Path

import cv2
from paths import IMAGES_OUT_PATH
from TrafficLightDetector.traffic_light_detector import TrafficLightDetector


class TrafficLightDetectorImages(TrafficLightDetector):

    def __init__(self, path) -> None:
        self.path = path
        self.image = cv2.imread(str(path))
        super().__init__(self.image)

    def _save_image(self) -> None:
        cv2.imwrite(str(IMAGES_OUT_PATH.joinpath(self.path.name)), self.image_copy)

    def draw(self) -> None:
        self._draw_circle(self.image)
        self._save_image()
