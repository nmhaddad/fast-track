""" ObjectDetector base class """

import logging
from typing import Tuple, List
from abc import ABCMeta, abstractmethod

import numpy as np
import cv2


class ObjectDetector(metaclass=ABCMeta):
    """ Base class for object detectors.

    Attributes:
        names: list of class names.
        image_shape: shape of input images.
        visualize: boolean value to visualize outputs.
    """

    def __init__(self, weights_path: str, names: List[str], image_shape: Tuple[int, int], visualize: bool):
        """ Init ObjectDetector objects.

        Args:
            weights_path: str path to model weights.
            names: list of class names.
            image_shape: tuple of height and width of input images.
            visualize: boolean value to visualize outputs.
        """
        self.weights_path = weights_path
        self.names = names
        self.image_shape = image_shape
        self.visualize = visualize

        # Generate class colors for detection visualization
        rng = np.random.default_rng()
        self.class_colors = [rng.integers(low=0, high=255, size=3, dtype=np.uint8).tolist() for _ in self.names]

    @property
    def image_width(self) -> int:
        """ Image width attribute.

        Returns:
            Image width.
        """
        return self.image_shape[0]

    @property
    def image_height(self) -> int:
        """ Image height attribute.

        Returns:
            Image height.
        """
        return self.image_shape[1]

    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[list, list, list]:
        """ Runs inference over an input image.

        Args:
            image: input image

        Returns:
            Postprocessed output.
        """
        raise NotImplementedError

    def visualize_detections(self, frame: np.ndarray, class_ids: list, scores: list, boxes: list,
                             thickness: int = 2) -> None:
        """ Visualizes output.

        Args:
            frame: input image.
            class_ids: list of ids detected.
            scores: list of scores of detected objects.
            boxes: list of detected boxes.
            thickness: int associated with thickness of text and box lines.
        """
        for cid, score, box in zip(class_ids, scores, boxes):
            x1, y1, x2, y2 = box.astype(int)
            cv2.putText(frame, self.names[cid], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, self.class_colors[id],
                        thickness, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.class_colors[cid], thickness)

    def __call__(self, image: np.ndarray) -> Tuple[list, list, list]:
        """ Runs inference over an input image.

        Args:
            image: input image

        Returns:
            Postprocessed output.
        """
        class_ids, scores, boxes = self.detect(image)
        if self.visualize:
            self.visualize_detections(image, class_ids, scores, boxes)
        return class_ids, scores, boxes
