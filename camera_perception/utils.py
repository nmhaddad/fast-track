""" Utilities file """

import cv2


def save_video(frames: list, outfile_path: str) -> None:
    """ Saves a list of frames to a video file.
    Args:
        frames: list
            List of frames to process into a video
        outfile_path: str
            Path to write the video to
    """
    h, w, *_ = frames[0].shape
    outfile = cv2.VideoWriter(outfile_path, -1, 30, (w, h))
    for frame in frames:
        outfile.write(frame)
    outfile.release()
