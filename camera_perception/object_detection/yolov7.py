from typing import Tuple

import onnxruntime as ort
import cv2
import numpy as np

from .object_detector import ObjectDetector


class YOLOv7(ObjectDetector):

    def __init__(self, weights_path, image_shape: Tuple[int, int]):
        super().__init__(image_shape)
        self.model = None
        self.weights_path = weights_path
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.initialize_model()

    def initialize_model(self):
        """ Creates a YOLOv7 detector from an ONNX file """
        self.session = ort.InferenceSession(self.weights_path, providers = self.providers)
        
        model_inputs = self.session.get_inputs()
        self.input_names = [i.name for i in model_inputs]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        model_outputs = self.session.get_outputs()
        self.output_names = [o.name for o in model_outputs]

    def preprocess_input(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        input_tensor = image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def postprocess_output(self, tensor):
        scores = tensor[0][:, -1]
        predictions = tensor[0][:, [0,5,1,2,3,4]]
        class_ids = predictions[:, 1].astype(int)
        boxes = predictions[:, 2:]
        boxes = self.rescale_boxes(boxes)
        if len(scores) == 0:
            return [], [], []
        return class_ids, scores, boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
    
        return boxes

    def detect(self, image):
        input_tensor = self.preprocess_input(image)
        output_tensor = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        detections = self.postprocess_output(output_tensor)
        return detections
