import cv2
from loguru import logger
from paths import HAAR_PATH
from TrafficLightDetector.color import Color


class TrafficLightDetector:
    CASCADE = cv2.CascadeClassifier(str(HAAR_PATH))
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    RADIUS = 5
    BOUNDARY = 2
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

    def _set_image(self, image=None) -> None:
        self.image = image
        self.image_copy = self.image
        self.size = self.image.shape
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.red = Color("RED", self.RED, self.RED_LOWER, self.RED_UPPER, hsv, minDist=80, param2=10)
        self.yellow = Color("YELLOW", self.YELLOW, self.YELLOW_LOWER, self.YELLOW_UPPER, hsv, minDist=60, param2=10)
        self.green = Color("GREEN", self.GREEN, self.GREEN_LOWER, self.GREEN_UPPER, hsv, minDist=30, param2=5)
        self.colors = [self.red, self.yellow, self.green]

    def _find_traffic_lights(self) -> None:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # draw rectangle around traffic lights
        for x, y, width, height in self.CASCADE.detectMultiScale(gray, 1.2, 5):
            cv2.rectangle(self.image, (x, y), (x + width, y + height), (255, 0, 0), self.BOUNDARY)

    def _draw_circle(self) -> None:
        for color in self.colors:
            if color.circle is not None:
                for values in color.circle[0, :]:
                    if values[0] > self.size[1] or values[1] > self.size[0] or values[1] > self.size[0] * self.BOUNDARY:
                        continue

                    h, s = 0, 0
                    for inner_radius in range(-self.RADIUS, self.RADIUS):
                        for outter_radius in range(-self.RADIUS, self.RADIUS):
                            if (values[1] + inner_radius) >= self.size[0] or (values[0] + outter_radius) >= self.size[1]:
                                continue
                            h += color.mask[values[1] + inner_radius, values[0] + outter_radius]
                            s += 1
                    if h / s > 100:
                        cv2.circle(self.image_copy, (values[0], values[1]), values[2] + 10, color.color, 2)
                        cv2.circle(color.mask, (values[0], values[1]), values[2] + 30, (255, 255, 255), 2)
                        cv2.putText(self.image_copy, color.name, (values[0], values[1]), self.FONT, 1, color.color, 2, cv2.LINE_AA)
                        self.signal = color.name

    def get_signal(self) -> str:
        return self.signal
