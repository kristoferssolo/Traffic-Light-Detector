import tensorflow as tf
import numpy as np
import cv2
import logging
from detector.paths import LOGS_PATH, IMAGES_OUT_PATH
from pathlib import Path

# Inception V3 model for Keras
from tensorflow.keras.applications.inception_v3 import preprocess_input


# Set up logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler(str(Path.joinpath(LOGS_PATH, f"{__name__}.log")))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# COCO labels
LABEL_PERSON = 1
LABEL_CAR = 3
LABEL_BUS = 6
LABEL_TRUCK = 8
LABEL_TRAFFIC_LIGHT = 10
LABEL_STOP_SIGN = 13


def accept_box(boxes, box_index, tolerance) -> bool:
    """Eliminate duplicate bounding boxes."""
    box = boxes[box_index]

    for idx in range(box_index):
        other_box = boxes[idx]
        if abs(center(other_box, "x") - center(box, "x")) < tolerance and abs(center(other_box, "y") - center(box, "y")) < tolerance:
            return False

    return True


def load_model(model_name):
    """Download a pretrained object detection model, and save it to your hard drive."""
    url = f"http://download.tensorflow.org/models/object_detection/tf2/20200711/{model_name}.tar.gz"

    # Download a file from a URL that is not already in the cache
    model_dir = tf.keras.utils.get_file(fname=model_name, untar=True, origin=url)

    logger.info(f"Model path: {model_dir}")

    return tf.saved_model.load(f"{model_dir}/saved_model")


def load_rgb_images(files, shape=None):
    """Loads the images in RGB format."""

    # For each image in the directory, convert it from BGR format to RGB format
    images = [cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB) for file in files]

    # Resize the image if the desired shape is provided
    return [cv2.resize(img, shape) for img in images] if shape else images


def load_ssd_coco():
    """Load the neural network that has the SSD architecture, trained on the COCO data set."""
    return load_model("ssd_resnet50_v1_fpn_640x640_coco17_tpu-8")


def save_image_annotated(img_rgb, file_name: Path, output, model_traffic_lights=None) -> None:
    """Annotate the image with the object types, and generate cropped images of traffic lights."""
    output_file = Path.joinpath(IMAGES_OUT_PATH, file_name.name)

    # For each bounding box that was detected
    for idx, _ in enumerate(output["boxes"]):

        # Extract the type of the object that was detected
        obj_class = output["detection_classes"][idx]

        # How confident the object detection model is on the object's type
        score = int(output["detection_scores"][idx] * 100)

        # Extract the bounding box
        box = output["boxes"][idx]

        color = None
        label_text = ""

        # if obj_class == LABEL_PERSON:
        #     color = (0, 255, 255)
        #     label_text = f"Person {score}"
        # if obj_class == LABEL_CAR:
        #     color = (255, 255, 0)
        #     label_text = f"Car {score}"
        # if obj_class == LABEL_BUS:
        #     label_text = f"Bus {score}"
        #     color = (255, 255, 0)
        # if obj_class == LABEL_TRUCK:
        #     color = (255, 255, 0)
        #     label_text = f"Truck {score}"
        # if obj_class == LABEL_STOP_SIGN:
        #     color = (128, 0, 0)
        #     label_text = f"Stop Sign {score}"
        if obj_class == LABEL_TRAFFIC_LIGHT:
            color = (255, 255, 255)
            label_text = f"Traffic Light {score}"

            if model_traffic_lights:

                # Annotate the image and save it
                img_traffic_light = img_rgb[box["y"]:box["y2"], box["x"]:box["x2"]]
                img_inception = cv2.resize(img_traffic_light, (299, 299))

                # Uncomment this if you want to save a cropped image of the traffic light
                # cv2.imwrite(output_file.replace('.jpg', '_crop.jpg'), cv2.cvtColor(img_inception, cv2.COLOR_RGB2BGR))
                img_inception = np.array([preprocess_input(img_inception)])

                prediction = model_traffic_lights.predict(img_inception)
                label = np.argmax(prediction)
                score_light = int(np.max(prediction) * 100)

                match label:
                    case 0: label_text = f"Green {score_light}"
                    case 1: label_text = f"Yellow {score_light}"
                    case 2: label_text = f"Red {score_light}"
                    case _: label_text = "NO-LIGHT"

        if color and label_text and accept_box(output["boxes"], idx, 5) and score > 50:
            cv2.rectangle(img_rgb, (box["x"], box["y"]), (box["x2"], box["y2"]), color, 2)
            cv2.putText(img_rgb, label_text, (box["x"], box["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(str(output_file), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    logger.info(output_file)


def center(box, coord_type):
    """Get center of the bounding box."""
    return (box[coord_type] + box[coord_type + "2"]) / 2


def perform_object_detection(model, file_name, save_annotated=False, model_traffic_lights=None):
    """Perform object detection on an image using the predefined neural network."""
    # Store the image
    img_bgr = cv2.imread(str(file_name))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img_rgb)  # Input needs to be a tensor
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run the model
    output = model(input_tensor)

    logger.info(f"Number detections: {output['num_detections']} {int(output['num_detections'])}")

    # Convert the tensors to a NumPy array
    num_detections = int(output.pop("num_detections"))
    output = {key: value[0, :num_detections].numpy()
              for key, value in output.items()}
    output["num_detections"] = num_detections

    logger.info(f"Detection classes: {output['detection_classes']}")
    logger.info(f"Detection Boxes: {output['detection_boxes']}")

    # The detected classes need to be integers.
    output["detection_classes"] = output["detection_classes"].astype(np.int64)
    output["boxes"] = [
        {"y": int(box[0] * img_rgb.shape[0]), "x": int(box[1] * img_rgb.shape[1]), "y2": int(box[2] * img_rgb.shape[0]),
         "x2": int(box[3] * img_rgb.shape[1])} for box in output["detection_boxes"]]

    if save_annotated:
        save_image_annotated(img_rgb, file_name, output, model_traffic_lights)

    return img_rgb, output, file_name


def perform_object_detection_video(model, video_frame, model_traffic_lights=None):
    """Perform object detection on a video using the predefined neural network."""
    # Store the image
    img_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img_rgb)  # Input needs to be a tensor
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run the model
    output = model(input_tensor)

    # Convert the tensors to a NumPy array
    num_detections = int(output.pop("num_detections"))
    output = {key: value[0, :num_detections].numpy()
              for key, value in output.items()}
    output["num_detections"] = num_detections

    # The detected classes need to be integers.
    output["detection_classes"] = output["detection_classes"].astype(np.int64)
    output["boxes"] = [
        {"y": int(box[0] * img_rgb.shape[0]), "x": int(box[1] * img_rgb.shape[1]), "y2": int(box[2] * img_rgb.shape[0]),
         "x2": int(box[3] * img_rgb.shape[1])} for box in output["detection_boxes"]]

    # For each bounding box that was detected
    for idx, _ in enumerate(output["boxes"]):

        # Extract the type of the object that was detected
        obj_class = output["detection_classes"][idx]

        # How confident the object detection model is on the object's type
        score = int(output["detection_scores"][idx] * 100)

        # Extract the bounding box
        box = output["boxes"][idx]

        color = None
        label_text = ""

        # if obj_class == LABEL_PERSON:
        #     color = (0, 255, 255)
        #     label_text = "Person " + str(score)
        # if obj_class == LABEL_CAR:
        #     color = (255, 255, 0)
        #     label_text = "Car " + str(score)
        # if obj_class == LABEL_BUS:
        #     color = (255, 255, 0)
        #     label_text = "Bus " + str(score)
        # if obj_class == LABEL_TRUCK:
        #     color = (255, 255, 0)
        #     label_text = "Truck " + str(score)
        # if obj_class == LABEL_STOP_SIGN:
        #     color = (128, 0, 0)
        #     label_text = f"Stop Sign {score}"
        if obj_class == LABEL_TRAFFIC_LIGHT:
            color = (255, 255, 255)
            label_text = f"Traffic Light {score}"

            if model_traffic_lights:

                # Annotate the image and save it
                img_traffic_light = img_rgb[box["y"]:box["y2"], box["x"]:box["x2"]]
                img_inception = cv2.resize(img_traffic_light, (299, 299))

                img_inception = np.array([preprocess_input(img_inception)])

                prediction = model_traffic_lights.predict(img_inception)
                label = np.argmax(prediction)
                score_light = int(np.max(prediction) * 100)

                match label:
                    case 0: label_text = f"Green {score_light}"
                    case 1: label_text = f"Yellow {score_light}"
                    case 2: label_text = f"Red {score_light}"
                    case _: label_text = "NO-LIGHT"  # This is not a traffic light

        # Use the score variable to indicate how confident we are it is a traffic light (in % terms)
        # On the actual video frame, we display the confidence that the light is either red, green,
        # yellow, or not a valid traffic light.
        if color and label_text and accept_box(output["boxes"], idx, 5) and score > 20:
            cv2.rectangle(img_rgb, (box["x"], box["y"]), (box["x2"], box["y2"]), color, 2)
            cv2.putText(img_rgb, label_text, (box["x"], box["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def double_shuffle(images, labels):
    """Shuffle the images to add some randomness."""
    indexes = np.random.permutation(len(images))

    return [images[idx] for idx in indexes], [labels[idx] for idx in indexes]


def reverse_preprocess_inception(image_preprocessed):
    """Reverse the preprocessing process."""
    image = image_preprocessed + 1 * 127.5
    return image.astype(np.uint8)
