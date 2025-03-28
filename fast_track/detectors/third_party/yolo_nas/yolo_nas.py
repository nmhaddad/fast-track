"""YOLONAS Wrapper implementation.

See licenses:
fast_track/object_detection/third_party/yolo_nas/LICENSE.md
fast_track/object_detection/third_party/yolo_nas/LICENSE.YOLONAS.md
"""

from typing import List, Tuple

import numpy as np
import torch

from ....detectors import ObjectDetector


class YOLONAS(ObjectDetector):
    """YOLONAS Wrapper class.

    Attributes:
        model: super_gradients model.
    """

    def __init__(
        self,
        weights_path: str,
        names: List[str],
        image_shape: Tuple[int, int],
        visualize: bool = False,
        pretrained="coco",
    ):
        """Initializes a YOLONAS object.

        Args:
            weights_path: str path to model weights.
            names: list of class names.
            image_shape: tuple of height and width of input images.
            visualize: boolean value to visualize outputs.
        """
        super().__init__(weights_path=weights_path, names=names, image_shape=image_shape, visualize=visualize)
        import super_gradients

        self.model = super_gradients.training.models.get(self.weights_path, pretrained_weights=pretrained)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def detect(self, image: np.ndarray) -> Tuple[List[int], List[float], List[np.ndarray]]:
        """Runs inference over an input image.

        Args:
            image: input image

        Returns:
            Tuple[List[int], List[float], List[np.ndarray]]: Postprocessed output (class_ids, scores, boxes).
        """
        detections = self.model.predict(image)
        boxes = detections.prediction.bboxes_xyxy.tolist()
        class_ids = detections.prediction.labels.astype(int).tolist()
        scores = detections.prediction.confidence.tolist()
        return class_ids, scores, boxes
