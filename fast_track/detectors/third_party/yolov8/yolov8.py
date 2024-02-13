""" YOLOv8 detector wrapper """

import os
import tempfile
from typing import Tuple, List, Optional
import urllib.request

import numpy as np
from ultralytics import YOLO

from ...object_detector import ObjectDetector


MODELS = {
    "yolov8_n": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    "yolov8_s": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
    "yolov8_m": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
    "yolov8_l": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
    "yolov8_x": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
}


class YOLOv8(ObjectDetector):
    """ YOLOv8 Wrapper class.

    Attributes:
        model: super_gradients model.
    """

    def __init__(self, weights_path: str,
                 names: List[str],
                 image_shape: Tuple[int, int],
                 visualize: Optional[bool] = False):
        """ Initializes a YOLOv8 object.

        Args:
            weights_path: str path to model weights.
            names: list of class names.
            image_shape: tuple of height and width of input images.
            visualize: boolean value to visualize outputs.
        """
        # if pretrained model provided, download from ultralytics
        if MODELS.get(weights_path):
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, f"{weights_path}.pt")
            urllib.request.urlretrieve(MODELS[weights_path], file_path)
            weights_path = file_path
        super().__init__(weights_path=weights_path, names=names, image_shape=image_shape, visualize=visualize)
        self.model = YOLO(self.weights_path)

    def detect(self, image: np.ndarray) -> Tuple[List[int], List[float], List[np.ndarray]]:
        """ Runs inference over an input image.

        Args:
            image: input image

        Returns:
            Tuple[List[int], List[float], List[np.ndarray]]: Postprocessed output (class_ids, scores, boxes).
        """
        detections = self.model(image)[0]
        boxes = detections.boxes.xyxy.tolist()
        scores = detections.boxes.conf.tolist()
        class_ids = detections.boxes.cls.to(int).tolist()
        return class_ids, scores, boxes
