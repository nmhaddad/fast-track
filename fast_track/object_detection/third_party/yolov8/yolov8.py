""" YOLOv8 ONNX detector wrapper """

from typing import Tuple, List, Optional

import numpy as np
from ultralytics.yolo.utils import ops
import torch

from ...object_detector_onnx import ObjectDetectorONNX


class YOLOv8(ObjectDetectorONNX):
    """ YOLOv8 ONNX detector.

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

    def __init__(self, weights_path: str, names: List[str], image_shape: Tuple[int, int], visualize: Optional[bool] = True):
        """ Init YOLOv8 objects with given parameters.

        Args:
            weights_path. path to pretrained weights.
            names: a list of names for classes.
            image_shape: shape of input images.
            visualize: bool to visualize output or not.
        """
        super().__init__(weights_path, names, image_shape, visualize)

    def postprocess(self, tensor: np.ndarray) -> Tuple[list, list, list]:
        """ Postprocesses output.

        Args:
            tensor: output tensor from ONNX session.

        Returns:
            Postprocessed output as a tuple of class_ids, scores, and boxes.
        """
        predictions = ops.non_max_suppression(torch.tensor(tensor[0]), conf_thres=0.25, iou_thres=0.45, classes=None,
                                              agnostic=False, multi_label=False, labels=(), max_det=300, nm=0)
        boxes = predictions[0][:, :4].int().numpy()
        class_ids = predictions[0][:, 5:6].int().flatten().tolist()
        scores = predictions[0][:, 4:5].flatten().tolist()
        if len(scores) == 0:
            return [], [], []
        boxes = self.rescale_boxes(boxes)
        return class_ids, scores, boxes
