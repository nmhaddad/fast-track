""" YOLOv8ONNX detector wrapper """

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from ultralytics.utils import ops

from ...object_detector_onnx import ObjectDetectorONNX


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
                 visualize: bool = False,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 classes: Optional[List[int]] = None,
                 agnostic: bool = False,
                 multi_label: bool = False,
                 labels: List[List[Union[int, float, torch.Tensor]]] = (),
                 max_det: int = 300):
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
                                              classes=len(self.classes),
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
