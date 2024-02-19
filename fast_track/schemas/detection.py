""" Detection schemas """

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, Mapped

from .base import Base
# from .track import Track


class Detection(Base):
    """ Detection schema """
    __tablename__ = "detections"
    detection_id = Column(Integer, primary_key=True)
    frame_number = Column(Integer, nullable=False)
    image_base64 = Column(String, nullable=False)
    track_id = Column(Integer, ForeignKey("tracks.track_id"))
    track: Mapped["Track"] = relationship(back_populates="detections")
