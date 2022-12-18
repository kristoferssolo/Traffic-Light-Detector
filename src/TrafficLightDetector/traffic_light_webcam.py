import cv2
from loguru import logger
from TrafficLightDetector.traffic_light_detector import TrafficLightDetector


class TrafficLightDetectorWebcam(TrafficLightDetector):

    def __init__(self) -> None:
        self.video_capture = cv2.VideoCapture(0)  # Change number if webcam didn't detect

    def enable(self) -> None:
        while True:
            _, frame = self.video_capture.read()
            self._set_image(frame)
            self._outline_traffic_lights()
            cv2.imshow("Video", self.image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.video_capture.release()
        cv2.destroyAllWindows()
