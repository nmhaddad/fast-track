""" Processing pipeline with detection and tracking """

from __future__ import annotations

import logging
from typing import Optional
from types import TracebackType

import cv2

from .object_detection import ObjectDetector
from .object_tracking import ObjectTracker
from .utils import save_video


class Pipeline:
    """ Class that represents a camera-based detection and tracking pipeline

    Attributes:
        camera: opencv-python camera for reading video data.
        detector: object detector.
        tracker: object tracker.
        frames: list containing processed frames.
        outfile: path to write processed frames to.
    """

    def __init__(self,
                 camera: cv2.VideoCapture,
                 detector: ObjectDetector,
                 tracker: ObjectTracker,
                 outfile: Optional[str] = 'video.avi'):
        """ Inits Pipeline class with a given object detector and tracker.

        Args:
            data_path: path to camera file or stream.
            detector: object detector.
            tracker: object tracker.
            outfile: path to write processed frames to.
        """
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.frames = []
        self.outfile = outfile

    def __enter__(self):
        """ Context manager enter. """
        return self

    def __exit__(self,
                 type: Optional[type[BaseException]] = None,
                 value: Optional[BaseException] = None,
                 traceback: Optional[TracebackType] = None) -> None:
        """ Context manager exit. """
        if type or value or traceback:
            logging.info(type)
            logging.info(value)
            logging.info(traceback)
        self.camera.release()
        cv2.destroyAllWindows()

    def run(self) -> None:
        """ Runs object tracking pipeline. """
        while True:
            ret, frame = self.camera.read()

            if not ret:
                break

            # detection
            class_ids, scores, boxes = self.detector.detect(frame)
            if self.detector.visualize:
                self.detector.visualize_detections(frame, class_ids, scores, boxes)

            # tracking
            self.tracker.update(boxes, scores, class_ids)
            self.tracker.visualize_tracks(frame)

            self.frames.append(frame)

        if self.frames:
            logging.info(f"saving output to {self.outfile}")
            save_video(self.frames, self.outfile, fps=self.camera.get(cv2.CAP_PROP_FPS))
