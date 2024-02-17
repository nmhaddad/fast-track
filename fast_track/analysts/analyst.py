""" Analyst base class. """

from abc import ABCMeta, abstractmethod
from typing import Any, Dict


class Analyst(metaclass=ABCMeta):
    """ Base Analyst class. """

    def __init__(self):
        """ Initializes Analyst objects. """
        pass

    @abstractmethod
    def update(self, track_messages: Dict[str, Any]) -> None:
        """ Updates data store for Analyst objects.

        Args:
            track_messages: dicitonary of track information.
        """
