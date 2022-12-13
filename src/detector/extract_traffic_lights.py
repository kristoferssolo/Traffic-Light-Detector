"""This program extracts traffic lights from images."""

from pathlib import Path

import cv2
from detector.object_detection import (
    LABEL_TRAFFIC_LIGHT,
    load_ssd_coco,
    perform_object_detection,
)
from detector.paths import CROPPED_IMAGES_PATH, INPUT_PATH
from loguru import logger


@logger.catch
def extract_traffic_lights() -> None:
    files = INPUT_PATH.iterdir()

    # Load the object detection model
    this_model = load_ssd_coco()

    # Keep track of the number of traffic lights found
    traffic_light_count = 0

    # Keep track of the number of image files that were processed
    file_count = 0

    # Display a count of the number of images we need to process
    # logger.info(f"Number of Images: {len(files)}")

    # Go through each image file, one at a time
    for file in files:

        # Detect objects in the image
        # img_rgb is the original image in RGB format
        # out is a dictionary containing the results of object detection
        # file_name is the name of the file
        img_rgb, out, file_name = perform_object_detection(model=this_model, file_name=file, save_annotated=None, model_traffic_lights=None)

        # Every 10 files that are processed
        if file_count % 10 == 0:

            # Display a count of the number of files that have been processed
            logger.info(f"Images processed: {file_count}")

            # Display the total number of traffic lights that have been identified so far
            logger.info(f"Number of Traffic lights identified: {traffic_light_count}")

        # Increment the number of files by 1
        file_count += 1

        # For each traffic light (i.e. bounding box) that was detected
        for idx, _ in enumerate(out["boxes"]):

            # Extract the type of object that was detected
            obj_class = out["detection_classes"][idx]

            # If the object that was detected is a traffic light
            if obj_class == LABEL_TRAFFIC_LIGHT:

                # Extract the coordinates of the bounding box
                box = out["boxes"][idx]

                # Extract (i.e. crop) the traffic light from the image
                traffic_light = img_rgb[box["y"]:box["y2"], box["x"]:box["x2"]]

                # Convert the traffic light from RGB format into BGR format
                traffic_light = cv2.cvtColor(traffic_light, cv2.COLOR_RGB2BGR)

                # Store the cropped image in a folder named 'traffic_light_cropped'
                cv2.imwrite(str(CROPPED_IMAGES_PATH.joinpath(f"{traffic_light_count}.jpg")), traffic_light)

                # Increment the number of traffic lights by 1
                traffic_light_count += 1

    # Display the total number of traffic lights identified
    logger.info(f"Number of Traffic lights identified: {traffic_light_count}")
