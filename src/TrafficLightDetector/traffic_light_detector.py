import cv2
from loguru import logger
from paths import HAAR_PATH
from TrafficLightDetector.color import Color


class TrafficLightDetector:
    CASCADE = cv2.CascadeClassifier(str(HAAR_PATH))
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    RADIUS = 5
    BOUNDARY = 2
    TEXT = False
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

    def _set_image(self, image=None, roi=None, detectTrafficLights=True) -> None:
        self.image = image
        self.roi = self.image if roi is None else roi
        self.size = self.image.shape if roi is None else self.roi.shape
        hsv = cv2.cvtColor(self.image if roi is None else self.roi, cv2.COLOR_BGR2HSV)
        self.red = Color("RED", self.RED, self.RED_LOWER, self.RED_UPPER, hsv, minDist=80, param2=10)
        self.yellow = Color("YELLOW", self.YELLOW, self.YELLOW_LOWER, self.YELLOW_UPPER, hsv, minDist=60, param2=10)
        self.green = Color("GREEN", self.GREEN, self.GREEN_LOWER, self.GREEN_UPPER, hsv, minDist=30, param2=5)
        self.colors = [self.red, self.yellow, self.green]
        self.detect_traffic_lights = detectTrafficLights

    def _outline_traffic_lights(self) -> None:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # draw rectangle around traffic lights
        for x, y, width, height in self.CASCADE.detectMultiScale(gray, 1.2, 5):
            cv2.rectangle(self.image, (x, y), (x + width, y + height), (255, 0, 0), self.BOUNDARY)
            roi = self.image[y:y + height, x:x + width]
            self._set_image(self.image, roi)
            self._draw_circle()

    def _draw_circle(self) -> None:
        for color in self.colors:
            if color.circle is not None:
                for value in color.circle[0, :]:
                    if value[0] > self.size[1] or value[1] > self.size[0] or value[1] > self.size[0] * self.BOUNDARY:
                        continue

                    h, s = 0, 0
                    for i in range(-self.RADIUS, self.RADIUS):
                        for j in range(-self.RADIUS, self.RADIUS):
                            if (value[1] + i) >= self.size[0] or (value[0] + j) >= self.size[1]:
                                continue
                            h += color.mask[value[1] + i, value[0] + j]
                            s += 1
                    if h / s > 100:
                        cv2.circle(self.roi if self.detect_traffic_lights else self.image, (value[0], value[1]), value[2] + 10, color.color, 2)  # draws circle
                        cv2.circle(color.mask, (value[0], value[1]), value[2] + 30, (255, 255, 255), 2)
                        if self.TEXT:
                            cv2.putText(self.roi if self.detect_traffic_lights else self.image, color.name,
                                        (value[0], value[1]), self.FONT, 1, color.color, 2, cv2.LINE_AA)  # draws text
                        self.signal = color.name

    def get_signal(self) -> str:
        return self.signal
