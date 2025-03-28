"""This module contains the schemas for the database models."""

from .base import Base
from .detection import Detection
from .track import Track
from .job import Job
from .frame import Frame

__all__ = ["Base", "Detection", "Track", "Job", "Frame"]
