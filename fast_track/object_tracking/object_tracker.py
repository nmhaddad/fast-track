""" ObjectTracker base class """

from typing import List
from abc import ABCMeta

import numpy as np


class ObjectTracker(metaclass = ABCMeta):
    """ Object tracking base class.

    Attributes:
        visualize: bool to visualize tracks.
        names: names of classes/labels.
        class_colors: colors associates with classes/labels.
    """

    def __init__(self, names: List[str], visualize: bool):
        """ Initializes base object trackers.

        Args:
            names: list of classes/labels.
            visualize: bool to visualize tracks.
        """
        self.visualize = visualize

        # Generate class colors for detection visualization
        self.names = names
        rng = np.random.default_rng()
        self.class_colors = [rng.integers(low=0, high=255, size=3, dtype=np.uint8).tolist() for _ in self.names]
