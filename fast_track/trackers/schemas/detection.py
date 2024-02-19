""" Detection schema """

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, Mapped

from .base import Base
# from .track import Track


class Detection(Base):
    """ Detection schema """
    __tablename__ = "detections"
    image_id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, nullable=False)
    cropped_image = Column(String, nullable=False)
    track_id = Column(Integer, ForeignKey("tracks.track_id"))
    track: Mapped["Track"] = relationship(back_populates="detections")
