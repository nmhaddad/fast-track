"""YOLOv7ONNX detector wrapper"""

from typing import Tuple, List

import numpy as np

from ...object_detector_onnx import ObjectDetectorONNX


class YOLOv7ONNX(ObjectDetectorONNX):
    """YOLOv7ONNX detector.

    Attributes:
        weights_path: path to pretrained weights.
        providers: flags for CUDA execution.
        sessions: ORT session.
        input_names: model input names.
        input_shape: input shape (B,C,H,W)
        input_height: input height.
        input_width: input width.
        output_names: model output names.
    """

    def __init__(self, weights_path: str, names: List[str], image_shape: Tuple[int, int], visualize: bool = False):
        """Init YOLOv7 objects with given parameters.

        Args:
            weights_path. path to pretrained weights.
            names: a list of names for classes.
            image_shape: shape of input images.
            visualize: bool to visualize output or not.
        """
        super().__init__(weights_path, names, image_shape, visualize)

    def postprocess(self, tensor: np.ndarray) -> Tuple[list, list, list]:
        """Postprocesses output.

        Args:
            tensor: output tensor from ONNX session.

        Returns:
            Postprocessed output as a tuple of class_ids, scores, and boxes.
        """
        scores = tensor[0][:, -1]
        predictions = tensor[0][:, [0, 5, 1, 2, 3, 4]]
        class_ids = predictions[:, 1].astype(int)
        boxes = predictions[:, 2:]
        boxes = self.rescale_boxes(boxes)
        if len(scores) == 0:
            return [], [], []
        return class_ids, scores, boxes
