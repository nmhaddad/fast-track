"""Initializes the yolov8 module."""

from .yolov8 import YOLOv8
from .yolov8_onnx import YOLOv8ONNX

__all__ = ["YOLOv8", "YOLOv8ONNX"]
