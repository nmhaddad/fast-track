""" ObjectTracker base class """

from typing import List
from abc import ABCMeta, abstractmethod

import numpy as np


class ObjectTracker(metaclass = ABCMeta):

    def __init__(self, names: List[str], visualize: bool):
        self.visualize = visualize

        # Generate class colors for detection visualization
        self.names = names
        rng = np.random.default_rng()
        self.class_colors = [rng.integers(low=0, high=255, size=3).tolist() for _ in self.names]
