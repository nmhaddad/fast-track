
import onnxruntime as ort

from .object_detector import ObjectDetector

class YOLOv7(ObjectDetector):

    def __init__(self, weights_path):

        self.model = None
        self.weights_path = weights_path
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.initialize_model()

    def initialize_model(self):
        """ Creates a YOLOv7 detector from an ONNX file """
        self.session = ort.InferenceSession(self.weights_path, providers = self.providers)
        self.get_input_details()
        self.get_output_details()

    def preprocess_input(self, image):
        pass

    def postprocess_output(self, tensor):
        pass

    def detect(self, image):
        return image
