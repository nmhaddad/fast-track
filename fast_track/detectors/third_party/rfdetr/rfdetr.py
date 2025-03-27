""" RF-DETR detector wrapper """

from typing import Optional, Tuple

import numpy as np
from rfdetr import RFDETRBase, RFDETRLarge

from ...object_detector import ObjectDetector


class RFDETR(ObjectDetector):
    """ RF-DETR Wrapper class.

    Attributes:
        model: rfdetr model.
    """

    def __init__(self,
                 names: list,
                 resolution: Optional[int] = 560,
                 model_name: Optional[str] = 'rfdetr_base',
                 weights_path: Optional[str] = None,
                 visualize: bool = False,
                 threshold: float = 0.5):
        """ Initializes a RF-DETR object.

        Args:
            weights_path: str path to model weights.
            names: list of class names.
            image_shape: tuple of height and width of input images. Must be divisible by 56.
            visualize: boolean value to visualize outputs.
            threshold: float value to filter out low confidence detections.
        """
        # weights_path = weights_path if weights_path else 'rfdetr_base'
        super().__init__(weights_path=weights_path, names=names, image_shape=resolution, visualize=visualize)
        self.threshold = threshold

        if model_name == 'rfdetr_base':
            self.model = RFDETRBase(resolution=resolution)
        elif model_name == 'rfdetr_large':
            self.model = RFDETRLarge(resolution=resolution, device="cuda")
        else:
            raise NotImplementedError(f"Custom weight loading currently not supported for RF-DETR.")


    def detect(self, image: np.ndarray) -> Tuple[list, list, list]:
        """ Runs inference over an input image.

        Args:
            image: input image

        Returns:
            Tuple[list, list, list]: Postprocessed output (class_ids, scores, boxes).
        """
        detections = self.model.predict(image, threshold=self.threshold)
        return detections.class_id, detections.confidence, detections.xyxy
