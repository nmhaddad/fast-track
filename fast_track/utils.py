""" Utilities file """

from typing import Optional, List

import cv2
import numpy as np


def save_video(frames: List[np.ndarray], outfile_path: str, fps: Optional[int] = 30) -> None:
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
