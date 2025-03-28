"""Initializes the object_detection module"""

from .object_detector import ObjectDetector
from .object_detector_onnx import ObjectDetectorONNX
from .third_party.yolov7 import YOLOv7ONNX
from .third_party.yolov8 import YOLOv8, YOLOv8ONNX
from .third_party.yolo_nas import YOLONAS
from .third_party.yolov9 import YOLOv9ONNX
from .third_party.rfdetr import RFDETR
from .util import get_detector

__all__ = [
    "ObjectDetector",
    "ObjectDetectorONNX",
    "YOLOv7ONNX",
    "YOLOv8",
    "YOLOv8ONNX",
    "YOLONAS",
    "YOLOv9ONNX",
    "RFDETR",
    "get_detector",
]
