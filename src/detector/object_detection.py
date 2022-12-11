"""This program helps detect objects (e.g. traffic lights) in images."""
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from detector.paths import IMAGES_OUT_PATH
from loguru import logger

# Inception V3 model for Keras
from tensorflow.keras.applications.inception_v3 import preprocess_input


# COCO labels
LABEL_PERSON = 1
LABEL_CAR = 3
LABEL_BUS = 6
LABEL_TRUCK = 8
LABEL_TRAFFIC_LIGHT = 10
LABEL_STOP_SIGN = 13

# Create a dictionary that maps object class labels to their corresponding colors and text labels
LABELS = {
    LABEL_PERSON: (0, 255, 255),
    LABEL_CAR: (255, 255, 0),
    LABEL_BUS: (255, 255, 0),
    LABEL_TRUCK: (255, 255, 0),
    LABEL_TRAFFIC_LIGHT: (255, 255, 255),
    LABEL_STOP_SIGN: (128, 0, 0),
}

LABEL_TEXT = {
    LABEL_PERSON: "Person",
    LABEL_CAR: "Car",
    LABEL_BUS: "Bus",
    LABEL_TRUCK: "Truck",
    LABEL_TRAFFIC_LIGHT: "Traffic Light",
    LABEL_STOP_SIGN: "Stop Sign",
}


@logger.catch
def accept_box(boxes: list[dict[str, float]] | None, box_index: int, tolerance: int) -> bool:
    """Eliminate duplicate bounding boxes."""
    if boxes is not None:
        box = boxes[box_index]

        for idx in range(box_index):
            other_box = boxes[idx]
            if abs(center(other_box, "x") - center(box, "x")) < tolerance and abs(center(other_box, "y") - center(box, "y")) < tolerance:
                return False

        return True
    return False


@logger.catch
def load_model(model_name: str) -> tf.saved_model.LoadOptions:
    """Download a pretrained object detection model, and save it to your hard drive."""
    url = f"http://download.tensorflow.org/models/object_detection/tf2/20200711/{model_name}.tar.gz"

    # Download a file from a URL that is not already in the cache
    model_dir = tf.keras.utils.get_file(fname=model_name, untar=True, origin=url)

    logger.info(f"Model path: {model_dir}")

    return tf.saved_model.load(f"{model_dir}/saved_model")


@logger.catch
def load_rgb_images(files, shape: tuple[int, int] | None = None):
    """Loads the images in RGB format."""

    # For each image in the directory, convert it from BGR format to RGB format
    images = [cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB) for file in files]

    # Resize the image if the desired shape is provided
    return [cv2.resize(img, shape) for img in images] if shape else images


@logger.catch
def load_ssd_coco() -> tf.saved_model.LoadOptions:
    """Load the neural network that has the SSD architecture, trained on the COCO data set."""
    return load_model("ssd_resnet50_v1_fpn_640x640_coco17_tpu-8")


@logger.catch
def save_image_annotated(image_rgb, file_name: Path, output, model_traffic_lights=None) -> None:
    """Annotate the image with the object types, and generate cropped images of traffic lights."""
    output_file = Path.joinpath(IMAGES_OUT_PATH, file_name.name)

    # For each bounding box that was detected
    for idx, (box, object_class) in enumerate(zip(output["boxes"], output["detection_classes"])):

        color = LABELS.get(object_class, (255, 255, 255))
        # How confident the object detection model is on the object's type
        score: int = object_class * 100

        # Extract the bounding box
        box = output["boxes"][idx]

        label_text = f"{object_class} {score}"
        if object_class == LABEL_TRAFFIC_LIGHT:
            if model_traffic_lights is not None:

                # Annotate the image and save it
                image_traffic_light = image_rgb[box["y"]:box["y2"], box["x"]:box["x2"]]
                image_inception = cv2.resize(image_traffic_light, (299, 299))

                # Uncomment this if you want to save a cropped image of the traffic light
                image_inception = np.array([preprocess_input(image_inception)])

                prediction = model_traffic_lights.predict(image_inception)
                label = np.argmax(prediction)
                score_light = int(np.max(prediction) * 100)

                if label == 0:
                    label_text = f"Green {score_light}"
                elif label == 1:
                    label_text = f"Yellow {score_light}"
                elif label == 2:
                    label_text = f"Red {score_light}"
                else:
                    label_text = "NO-LIGHT"

            # Draw the bounding box and object class label on the image, if the confidence score is above 50 and the box is not a duplicate
            if color and label_text and accept_box(output["boxes"], idx, 5) and score > 50:
                cv2.rectangle(image_rgb, (box["x"], box["y"]), (box["x2"], box["y2"]), color, 2)
                cv2.putText(image_rgb, label_text, (box["x"], box["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(str(output_file), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    logger.info(output_file)


@logger.catch
def center(box: dict[str, float], coord_type: str) -> float:
    """Get center of the bounding box."""
    return (box[coord_type] + box[coord_type + "2"]) / 2


@logger.catch
def perform_object_detection(model, file_name, save_annotated=False, model_traffic_lights=None):
    """Perform object detection on an image using the predefined neural network."""
    # Store the image
    image_bgr = cv2.imread(str(file_name))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_rgb)  # Input needs to be a tensor
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run the model
    output = model(input_tensor)

    logger.info(f"Number detections: {output['num_detections']} {int(output['num_detections'])}")

    # Convert the tensors to a NumPy array
    num_detections = int(output.pop("num_detections"))
    output = {key: value[0, :num_detections].numpy() for key, value in output.items()}
    output["num_detections"] = num_detections

    logger.info(f"Detection classes: {output['detection_classes']}")
    logger.info(f"Detection Boxes: {output['detection_boxes']}")

    # The detected classes need to be integers.
    output["detection_classes"] = output["detection_classes"].astype(np.int64)
    output["boxes"] = [{"y": int(box[0] * image_rgb.shape[0]),
                        "x": int(box[1] * image_rgb.shape[1]),
                        "y2": int(box[2] * image_rgb.shape[0]),
                        "x2": int(box[3] * image_rgb.shape[1])}
                       for box in output["detection_boxes"]]

    if save_annotated:
        save_image_annotated(image_rgb, file_name, output, model_traffic_lights)

    return image_rgb, output, file_name


@logger.catch
def perform_object_detection_video(video_frame, model, model_traffic_lights):
    """Perform object detection on a video using the predefined neural network."""

    # Store the image
    image_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_rgb)  # Input needs to be a tensor
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run the model
    output = model(input_tensor)

    # Convert the tensors to a NumPy array
    number_detections = int(output.pop("num_detections"))
    output = {key: value[0, :number_detections].numpy() for key, value in output.items()}
    output["num_detections"] = number_detections

    # The detected classes need to be integers.
    output["detection_classes"] = output["detection_classes"].astype(np.int64)
    output["boxes"] = [{"y": int(box[0] * image_rgb.shape[0]),
                        "x": int(box[1] * image_rgb.shape[1]),
                        "y2": int(box[2] * image_rgb.shape[0]),
                        "x2": int(box[3] * image_rgb.shape[1])}
                       for box in output["detection_boxes"]]

    # For each bounding box that was detected
    for idx, (box, object_class) in enumerate(zip(output.get("boxes"), output.get("detection_classes"))):
        color = LABELS.get(object_class, None)
        # How confident the object detection model is on the object's type
        score: int = object_class * 100
        label_text = f"{LABEL_TEXT.get(object_class)} {score}"

        if object_class == LABEL_TRAFFIC_LIGHT:
            # Annotate the image and save it
            image_traffic_light = image_rgb[box.get("y"):box.get("y2"), box.get("x"):box.get("x2")]
            image_inception = cv2.resize(image_traffic_light, (299, 299))

            image_inception = np.array([preprocess_input(image_inception)])

            prediction = model_traffic_lights.predict(image_inception)
            label = np.argmax(prediction)
            score_light = int(np.max(prediction) * 100)

            if label == 0:
                label_text = f"Green {score_light}"
            elif label == 1:
                label_text = f"Yellow {score_light}"
            elif label == 2:
                label_text = f"Red {score_light}"
            else:
                label_text = "NO-LIGHT"

        # Use the score variable to indicate how confident we are it is a traffic light (in % terms)
        # On the actual video frame, we display the confidence that the light is either  green, yellow,
        # red or not a valid traffic light.
        if accept_box(output.get("boxes"), idx, 5) and score > 20:
            cv2.rectangle(image_rgb, (box.get("x"), box.get("y")), (box.get("x2"), box.get("y2")), color, 2)
            cv2.putText(image_rgb, label_text, (box.get("x"), box.get("y")), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


@logger.catch
def double_shuffle(images: list[str], labels: list[int]) -> tuple[list[str], list[int]]:
    """Shuffle the images to add some randomness."""
    indexes = np.random.permutation(len(images))

    return [images[idx] for idx in indexes], [labels[idx] for idx in indexes]


@logger.catch
def reverse_preprocess_inception(image_preprocessed):
    """Reverse the preprocessing process for an image that has been input to the Inception V3 model."""
    image = image_preprocessed + 1 * 127.5
    return image.astype(np.uint8)
