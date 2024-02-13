""" ObjectDetectorONNX class """

from typing import Tuple, List
from abc import ABCMeta, abstractmethod

import onnxruntime as ort
import numpy as np
import cv2

from .object_detector import ObjectDetector


class ObjectDetectorONNX(ObjectDetector, metaclass=ABCMeta):
    """ Class inherited by ONNX object detectors.

    Attributes:
        weights_path: str to weights to load model.
        names: list of class names.
        image_shape: shape of input images.
        visualize: boolean value to visualize outputs.
    """

    def __init__(self, weights_path: str, names: List[str], image_shape: Tuple[int, int], visualize: bool):
        """ Init ObjectDetector objects.

        Args:
            weights_path: str to weights to load model.
            names: list of class names.
            image_shape: tuple of height and width of input images.
            visualize: boolean value to visualize outputs.
        """
        super().__init__(weights_path, names, image_shape, visualize)
        self.providers = [
            'TensorrtExecutionProvider',
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        self.session = None
        self.input_shape = None
        self.input_names = None
        self.output_names = None

        self.initialize_model()

    @abstractmethod
    def postprocess(self, tensor: np.ndarray) -> Tuple[list, list, list]:
        """ Postprocesses output.

        Args:
            tensor: output tensor from ONNX session.

        Returns:
            Postprocessed output as a tuple of class_ids, scores, and boxes.
        """

    def initialize_model(self) -> None:
        """ Creates a YOLOvX detector from an ONNX file """
        self.session = ort.InferenceSession(self.weights_path, providers = self.providers)

        model_inputs = self.session.get_inputs()
        self.input_names = [i.name for i in model_inputs]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        model_outputs = self.session.get_outputs()
        self.output_names = [o.name for o in model_outputs]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """ Preprocesses input for ONNX inference.

        Args:
            image: an input image.

        Returns:
            An input tensor
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        input_tensor = image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def rescale_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """ Rescales bounding boxes.

        Args:
            boxes: an array of bounding boxes.

        Returns:
            Rescaled bounding boxes.
        """
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        return boxes

    def detect(self, image: np.ndarray) -> Tuple[list, list, list]:
        """ Runs inference over an input image.

        Args:
            image: input image

        Returns:
            Postprocessed output.
        """
        input_tensor = self.preprocess(image)
        output_tensor = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        detections = self.postprocess(output_tensor)
        return detections
