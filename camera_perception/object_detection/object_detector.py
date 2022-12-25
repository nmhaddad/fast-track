""" ObjectDetector base class """

import logging
from typing import Tuple
from abc import ABCMeta, abstractmethod

import numpy as np
import cv2


class ObjectDetector(metaclass = ABCMeta):

    def __init__(self, image_shape):

        self.detector = None
        self.names = None
        self.image_shape = image_shape

    @property
    def image_width(self) -> int:
        return self.image_shape[0]

    @property
    def image_height(self)-> int:
        return self.image_shape[1]

    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[list, list, list]:
        pass

    def visualize_detections(self, frame: np.ndarray, class_ids: list, scores: list, boxes: list) -> None:
        for id, score, box in zip(class_ids, scores, boxes):
            logging.info(id, score, box)
            x1, y1, x2, y2 = box.astype(int)
            cv2.putText(frame, self.names[id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
