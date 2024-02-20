""" Utilities file for object tracking. """

from typing import List

from .object_tracker import ObjectTracker
from .third_party.bytetrack.byte_tracker import BYTETracker


def get_tracker(tracker_name: str, names: List[str], **kwargs) -> ObjectTracker:
    """ Tracker selector.

    Args:
        tracker_name: name of the tracker to setup.
        names: list of label names.
        tracker_params: parameters for tracker.

    Returns:
        An object tracker.
    """
    if tracker_name == "ByteTrack":
        return BYTETracker(names=names, **kwargs)
    else:
        raise ValueError("Unsupported object tracker provided as argument.")
