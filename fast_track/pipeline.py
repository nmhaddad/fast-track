""" Processing pipeline with detection and tracking """

from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional
from types import TracebackType

import cv2

from .detectors import ObjectDetector
from .trackers import ObjectTracker


class Pipeline:
    """ Class that represents a camera-based detection and tracking pipeline

    Attributes:
        camera: opencv-python camera for reading video data.
        detector: object detector.
        tracker: object tracker.
        frames: list containing processed frames.
        outfile: cv2.VideoWriter object to write processed frames to.
    """

    def __init__(self,
                 camera: cv2.VideoCapture,
                 detector: ObjectDetector,
                 tracker: Optional[ObjectTracker] = None,
                 outfile: str = 'video.mp4'):
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
        # Output settings
        fourcc = cv2.VideoWriter_fourcc(*'h264')
        fps = int(self.camera.get(cv2.CAP_PROP_FPS))
        w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_dir = tempfile.mkdtemp()
        self.outfile_path = os.path.join(temp_dir, outfile)
        self.outfile = cv2.VideoWriter(self.outfile_path, fourcc, fps, (w, h))

    def __enter__(self):
        """ Context manager enter. """
        return self

    def __exit__(self,
                 exc_type: Optional[type[BaseException]] = None,
                 exc_value: Optional[BaseException] = None,
                 exc_traceback: Optional[TracebackType] = None) -> None:
        """ Context manager exit. """
        if exc_type or exc_value or exc_traceback:
            logging.info(type)
            logging.info(exc_value)
            logging.info(exc_traceback)
        self.camera.release()
        self.outfile.release()
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
            if self.tracker:
                self.tracker.update(boxes, scores, class_ids, frame)
                self.tracker.visualize_tracks(frame)

            # write processed frame to output file
            self.outfile.write(frame)
        return self.outfile_path
