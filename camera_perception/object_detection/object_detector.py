from typing import Tuple
from abc import ABCMeta, abstractmethod

import numpy as np


class ObjectDetector(metaclass = ABCMeta):

    def __init__(self, image_shape):

        self.detector = None
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
