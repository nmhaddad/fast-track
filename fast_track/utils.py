""" Utilities file """

from typing import List

import cv2
import numpy as np


COCO_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
    "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def save_video(frames: List[np.ndarray], outfile_path: str, fps: int = 30) -> None:
    """ Saves a list of frames to a video file.
    Args:
        frames: list
            List of frames to process into a video
        outfile_path: str
            Path to write the video to
    """
    h, w, *_ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    outfile = cv2.VideoWriter(outfile_path, fourcc, fps, (w, h))
    for frame in frames:
        outfile.write(frame)
    outfile.release()


def read_names_file(names_file: str) -> List[str]:
    """ Read a list of class names into a Python list.

    Args:
        names_file: path to a names file

    Returns:
        A list of names
    """
    with open(names_file, "r", encoding="utf8") as file:
        names = [line.rstrip() for line in file]
    return names
