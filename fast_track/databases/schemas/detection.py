""" Detection schemas """

from typing import TYPE_CHECKING

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, Mapped

from .base import Base

if TYPE_CHECKING:
    from .track import Track
else:
    Track = "Track"


class Detection(Base):
    """ Detection schema """
    __tablename__ = "detections"
    detection_id = Column(Integer, primary_key=True)
    frame_number = Column(Integer, nullable=False)
    image_base64 = Column(String, nullable=False)
    track_id = Column(Integer, ForeignKey("tracks.track_id"))
    track: Mapped["Track"] = relationship(back_populates="detections")
