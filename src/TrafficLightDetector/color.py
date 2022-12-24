from typing import NamedTuple

import cv2
import numpy as np
from loguru import logger


class ColorRange(NamedTuple):
    color1: tuple[int, int, int]
    color2: tuple[int, int, int]


@logger.catch
class ColorAttributes:
    def __init__(self, name: str, color: tuple[int, int, int], lowers: ColorRange,
                 uppers: ColorRange, hsv: cv2.cvtColor, minDist: int, param2: int) -> None:
        self.name = name
        self.color = color

        # set mask
        masks: list[cv2.inRange] = []
        for lower, upper in zip(lowers, uppers):
            masks.append(cv2.inRange(hsv, lower, upper))
        self.mask = masks[0]
        if len(masks) > 1:
            for mask in masks:
                self.mask = cv2.add(self.mask, mask)

        # set circle
        self.circle = cv2.HoughCircles(self.mask, cv2.HOUGH_GRADIENT, 1, minDist=minDist, param1=50,
                                       param2=param2, minRadius=0, maxRadius=30)
        if self.circle is not None:
            self.circle = np.uint16(np.around(self.circle))
