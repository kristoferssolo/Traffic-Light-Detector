"""This program uses a trained neural network to detect the color of a traffic light in images."""

from detector.object_detection import load_ssd_coco, perform_object_detection
from detector.paths import IMAGES_IN_PATH, MODEL_PATH
from loguru import logger
from tensorflow import keras


@logger.catch
def detect_traffic_light_color_image() -> None:
    model_traffic_lights_nn = keras.models.load_model(str(MODEL_PATH))
    # Load the SSD neural network that is trained on the COCO data set
    model_ssd = load_ssd_coco()

    # Go through all image files, and detect the traffic light color.
    for file in IMAGES_IN_PATH.iterdir():
        image, out, file_name = perform_object_detection(model=model_ssd, file_name=file, save_annotated=True, model_traffic_lights=model_traffic_lights_nn)
        logger.info(f"Performed object detection on {file}")
