"""This program uses a trained neural network to detect the color of a traffic light in images."""

from pathlib import Path

from detector.object_detection import load_ssd_coco, perform_object_detection
from detector.paths import IMAGES_IN_PATH, MODEL_PATH
from loguru import logger
from tensorflow import keras


@logger.catch
def detect_traffic_light_color_image() -> None:
    model_traffic_lights_nn = keras.models.load_model(str(MODEL_PATH))

    # Go through all image files, and detect the traffic light color.
    for file in Path.iterdir(IMAGES_IN_PATH):
        image, out, file_name = perform_object_detection(load_ssd_coco(), file, save_annotated=True, model_traffic_lights=model_traffic_lights_nn)
        logger.info(f"{file} {out}")
