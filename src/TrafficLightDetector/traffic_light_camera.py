import cv2
from magicsound import magicsound

from TrafficLightDetector.paths import SOUND_PATH
from TrafficLightDetector.traffic_light_detector import TrafficLightDetector


class TrafficLightDetectorCamera(TrafficLightDetector):
    def __init__(self, source: int, sound: bool = False) -> None:
        self.video_capture = cv2.VideoCapture(source)
        self.sound = sound

    def enable(self) -> None:
        while True:
            self._get_video()
            if self.sound:
                self._make_sound()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.video_capture.release()
        cv2.destroyAllWindows()

    def _get_video(self) -> None:
        _, frame = self.video_capture.read()
        self._set_image(frame)
        self._outline_traffic_lights()
        cv2.imshow("Video", self.image)

    def _make_sound(self) -> None:
        if self.signal_color == "GREEN":
            magicsound(str(SOUND_PATH.joinpath("move.mp3")))
