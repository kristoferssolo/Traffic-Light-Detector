import cv2
import numpy as np
from loguru import logger
from paths import IMAGES_OUT_PATH
from pathlib import Path


@logger.catch
class Color:

    def __init__(self, name: str, lower: tuple[int, int, int], upper: tuple[int, int, int], hsv: cv2.cvtColor, minDist: int, param2: int) -> None:
        self.name: str = name
        self.lower: tuple[int, int, int] = lower
        self.upper: tuple[int, int, int] = upper
        self.mask = cv2.inRange(hsv, self.lower, self.upper)
        self.circle = cv2.HoughCircles(self.mask, cv2.HOUGH_GRADIENT, 1, minDist=minDist, param1=50, param2=param2, minRadius=0, maxRadius=30)
        if self.circle is not None:
            self.circle = np.uint16(np.around(self.circle))


@logger.catch
class TrafficLightDetector:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    RADIUS = 5
    BOUNDARY = 4 / 10

    def __init__(self, path: Path) -> None:
        self.path = path
        self.image = cv2.imread(str(self.path))
        self.image_copy = self.image
        self.size = self.image.shape
        # self.red1 = Color("RED", (0, 100, 100), (10, 255, 255))
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.red = Color("RED", (160, 100, 100), (180, 255, 255), hsv, minDist=80, param2=10)
        self.yellow = Color("YELLOW", (15, 150, 150), (35, 255, 255), hsv, minDist=60, param2=10)
        self.green = Color("GREEN", (40, 50, 50), (90, 255, 255), hsv, minDist=30, param2=5)
        self.colors = [self.red, self.yellow, self.green]

    def draw_circle(self) -> None:
        for color in self.colors:
            if color.circle is not None:

                for i in color.circle[0, :]:
                    logger.debug(f"{i = }")
                    if i[0] > self.size[1] or i[1] > self.size[0] or i[1] > self.size[0] * self.BOUNDARY:
                        continue

                    h, s = 0, 0
                    for inner_radius in range(-self.RADIUS, self.RADIUS):
                        for outter_radius in range(-self.RADIUS, self.RADIUS):
                            if (i[1] + inner_radius) >= self.size[0] or (i[0] + outter_radius) >= self.size[1]:
                                continue
                            h += color.mask[i[1] + inner_radius, i[0] + outter_radius]
                            s += 1
                    if h / s > 100:
                        cv2.circle(self.image_copy, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
                        cv2.circle(color.mask, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
                        cv2.putText(self.image_copy, color.name, (i[0], i[1]), self.FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)
        self._save_image()

    def _save_image(self) -> None:
        cv2.imwrite(str(IMAGES_OUT_PATH.joinpath(self.path.name)), self.image_copy)
