import cv2
from loguru import logger
from TrafficLightDetector.color import Color


class TrafficLightDetector:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    RADIUS = 5
    BOUNDARY = 4 / 10
    # HSV values
    RED_LOWER = ((160, 100, 100), (0, 100, 100))
    RED_UPPER = ((180, 255, 255), (10, 255, 255))
    YELLOW_LOWER = ((15, 150, 150), (15, 100, 100))
    YELLOW_UPPER = ((35, 255, 255), (35, 255, 255))
    GREEN_LOWER = ((40, 50, 50), (95, 45, 38))
    GREEN_UPPER = ((90, 255, 255), (130, 60, 60))

    # BGR values
    RED = (0, 0, 200)
    YELLOW = (0, 175, 225)
    GREEN = (0, 150, 0)

    def _set_image(self, image) -> None:
        self.image = image
        self.image_copy = self.image
        self.size = self.image.shape
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.red = Color("RED", self.RED, self.RED_LOWER, self.RED_UPPER, hsv, minDist=80, param2=10)
        self.yellow = Color("YELLOW", self.YELLOW, self.YELLOW_LOWER, self.YELLOW_UPPER, hsv, minDist=60, param2=10)
        self.green = Color("GREEN", self.GREEN, self.GREEN_LOWER, self.GREEN_UPPER, hsv, minDist=30, param2=5)
        self.colors = [self.red, self.yellow, self.green]

    def _draw_circle(self):
        try:
            for color in self.colors:
                if color.circle is not None:

                    for i in color.circle[0, :]:
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
                            cv2.circle(self.image_copy, (i[0], i[1]), i[2] + 10, color.color, 2)
                            cv2.circle(color.mask, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
                            cv2.putText(self.image_copy, color.name, (i[0], i[1]), self.FONT, 1, color.color, 2, cv2.LINE_AA)
        except AttributeError:
            logger.warning("Image/frame was not specified")
