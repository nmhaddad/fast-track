""" Initializes the object_detection module """

from .object_detector import ObjectDetector
from .object_detector_onnx import ObjectDetectorONNX
from .third_party.yolov7.yolov7 import YOLOv7
from .third_party.yolov8 import YOLOv8, YOLOv8ONNX
from .third_party.yolo_nas.yolo_nas import YOLONAS
from .util import get_detector
