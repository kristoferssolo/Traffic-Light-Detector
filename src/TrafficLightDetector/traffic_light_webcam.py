import cv2
from loguru import logger
from paths import HAAR_PATH
from TrafficLightDetector.traffic_light_detector import TrafficLightDetector


class TrafficLightDetectorWebcam(TrafficLightDetector):

    def __init__(self) -> None:
        self.video_capture = cv2.VideoCapture(0)  # Change number if webcam didn't detect
        self.lights_cascade = cv2.CascadeClassifier(str(HAAR_PATH))

    def enable(self) -> None:
        while True:
            _, frame = self.video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            lights = self.lights_cascade.detectMultiScale(gray, 1.2, 5)
            for x, y, w, h in lights:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)

            # self._set_image(frame)
            # self._draw_circle()
            # cv2.imshow("Video", self.image_copy)
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.video_capture.release()
        cv2.destroyAllWindows()
