""" ObjectDetector base class """

import logging
from typing import Tuple, List, Optional
from abc import ABCMeta, abstractmethod

import numpy as np
import cv2


class ObjectDetector(metaclass = ABCMeta):

    def __init__(self, names: List[str], image_shape: Tuple[int, int]):
        self.names = names
        self.image_shape = image_shape

        # Generate class colors for detection visualization
        rng = np.random.default_rng()
        self.class_colors = [rng.integers(low=0, high=255, size=3).tolist() for _ in self.names]

    @property
    def image_width(self) -> int:
        return self.image_shape[0]

    @property
    def image_height(self) -> int:
        return self.image_shape[1]

    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[list, list, list]:
        pass

    def visualize_detections(self, frame: np.ndarray, class_ids: list, scores: list, boxes: list, thickness: Optional[int] = 2) -> None:
        for id, score, box in zip(class_ids, scores, boxes):
            logging.info(id, score, box)
            x1, y1, x2, y2 = box.astype(int)
            cv2.putText(frame, self.names[id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, self.class_colors[id], thickness, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.class_colors[id], thickness)
