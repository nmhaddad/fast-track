""" YOLOv8 detector wrappers """

import os
import tempfile
from typing import Tuple, List, Optional, Union
import urllib.request

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ops

from ...object_detector import ObjectDetector
from ...object_detector_onnx import ObjectDetectorONNX


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


class YOLOv8ONNX(ObjectDetectorONNX):
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
        conf_thresh: The confidence threshold below which boxes will be filtered out. Valid values are
                     between 0.0 and 1.0.
        iou_thresh: The IoU threshold below which boxes will be filtered out during NMS. Valid values are
                    between 0.0 and 1.0.
        agnostic:If True, the model is agnostic to the number of classes, and all classes will be considered as one.
        multi_label:If True, each box may have multiple labels.
        labels: A list of lists, where each inner list contains the apriori labels for a given image. The list should
                be in the format output by a dataloader, with each label being a tuple of
                (class_index, x1, y1, x2, y2).
        max_det: The maximum number of boxes to keep after NMS.
        nm: The number of masks output by the model.
    """

    def __init__(self,
                 weights_path: str,
                 names: List[str],
                 image_shape: Tuple[int, int],
                 visualize: Optional[bool] = False,
                 conf_thres: Optional[float] = 0.25,
                 iou_thres: Optional[float] = 0.45,
                 classes: Optional[List[int]] = None,
                 agnostic: Optional[bool] = False,
                 multi_label: Optional[bool] = False,
                 labels: Optional[List[List[Union[int, float, torch.Tensor]]]] = (),
                 max_det: Optional[int] = 300):
        """ Init YOLOv8 objects with given parameters.

        Args:
            weights_path. path to pretrained weights.
            names: a list of names for classes.
            image_shape: shape of input images.
            visualize: bool to visualize output or not.
            conf_thresh: The confidence threshold below which boxes will be filtered out. Valid values are
                         between 0.0 and 1.0.
            iou_thresh: The IoU threshold below which boxes will be filtered out during NMS. Valid values are
                        between 0.0 and 1.0.
            agnostic: If True, the model is agnostic to the number of classes, and all classes will be considered
                      as one.
            multi_label:If True, each box may have multiple labels.
            labels: A list of lists, where each inner list contains the apriori labels for a given image. The list
                    should be in the format output by a dataloader, with each label being a tuple of
                    (class_index, x1, y1, x2, y2).
            max_det: The maximum number of boxes to keep after NMS.
        """
        super().__init__(weights_path, names, image_shape, visualize)
        self.conf_thresh = conf_thres
        self.iou_thresh = iou_thres
        self.classes = classes
        self.agnostic = agnostic
        self.multi_label = multi_label
        self.labels = labels
        self.max_det = max_det

    def postprocess(self, tensor: np.ndarray) -> Tuple[list, list, list]:
        """ Postprocesses output.

        Args:
            tensor: output tensor from ONNX session.

        Returns:
            Postprocessed output as a tuple of class_ids, scores, and boxes.
        """
        predictions = ops.non_max_suppression(torch.tensor(tensor[0]),
                                              conf_thres=self.conf_thresh,
                                              iou_thres=self.iou_thresh,
                                              classes=self.classes,
                                              agnostic=self.agnostic,
                                              multi_label=self.multi_label,
                                              labels=self.labels,
                                              max_det=self.max_det)
        boxes = predictions[0][:, :4].int().numpy()
        class_ids = predictions[0][:, 5:6].int().flatten().tolist()
        scores = predictions[0][:, 4:5].flatten().tolist()
        if len(scores) == 0:
            return [], [], []
        boxes = self.rescale_boxes(boxes)
        return class_ids, scores, boxes
