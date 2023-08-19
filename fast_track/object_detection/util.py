""" Utilities file for object detection. """

import logging
from typing import List, Tuple

from .object_detector import ObjectDetector
from .third_party.yolo_nas.yolo_nas import YOLONAS
from .third_party.yolov8.yolov8 import YOLOv8
from .third_party.yolov7.yolov7 import YOLOv7

MODELS = {
    "YOLO-NAS L": "yolo_nas_l",
    "YOLO-NAS M": "yolo_nas_m",
    "YOLO-NAS S": "yolo_nas_s",
    "YOLOv8 N": "yolov8_n",
    "YOLOv8 S": "yolov8_s",
    "YOLOv8 M": "yolov8_m",
    "YOLOv8 X": "yolov8_x",
}

def get_detector(detector_name: str, names: List[str], image_shape: Tuple[int, int], detector_params: dict) -> ObjectDetector:
    """ Detector selector.

    Args:
        detector_name: name of the detector to setup.
        names: labels associated with model.
        image_shape: used for input shape for model.
        detector_params: key value parameter pairs.

    Returns:
        An object detector.
    """
    try:
        detector_name = MODELS[detector_name]
    except KeyError:
        logging.info("Attempting to load model from provided weights")
    if detector_name.startswith("yolo_nas"):
        return YOLONAS(weights_path=detector_name, names=names, image_shape=image_shape, **detector_params)
    elif detector_name.startswith("yolov8"):
        return YOLOv8(weights_path=detector_name, names=names, image_shape=image_shape, **detector_params)
    elif detector_name.startswith("yolov7"):
        return YOLOv7(weights_path=detector_name, names=names, image_shape=image_shape, **detector_params)
    else:
        raise ValueError("Detector name not found.")
