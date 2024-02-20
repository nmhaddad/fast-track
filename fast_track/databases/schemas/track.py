""" Track schema """

from typing import List

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship, Mapped

from .base import Base
from .job import Job
from .detection import Detection


class Track(Base):
    """ Track schema """
    __tablename__ = "tracks"
    track_id = Column(Integer, primary_key=True)
    count = Column(Integer, nullable=False)
    is_activated = Column(Boolean, nullable=False)
    state = Column(String, nullable=False)
    score = Column(Integer, nullable=False)
    start_frame_number = Column(Integer, nullable=False)
    curr_frame_number = Column(Integer, nullable=False)
    time_since_update = Column(Integer, nullable=False)
    location = Column(String, nullable=False)
    detections: Mapped[List["Detection"]] = relationship(
        back_populates="track", cascade="all, delete, delete-orphan"
    )
    class_name = Column(String, nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.job_id"))
    job: Mapped["Job"] = relationship(back_populates="tracks")
