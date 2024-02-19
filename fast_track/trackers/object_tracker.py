""" ObjectTracker base class """

from typing import Any, Dict, List, Optional
from abc import ABCMeta, abstractmethod
from datetime import datetime

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .schemas import Base, Detection, Job, Track


class ObjectTracker(metaclass=ABCMeta):
    """ Object tracking base class.

    Attributes:
        visualize: bool to visualize tracks.
        names: names of classes/labels.
        class_colors: colors associates with classes/labels.
        db_uri: database uri.
    """

    def __init__(self,
                 names: List[str],
                 visualize: bool,
                 db_uri: Optional[str] = None):
        """ Initializes base object trackers.

        Args:
            names: list of classes/labels.
            visualize: bool to visualize tracks.
            db_uri: database uri.
        """
        self.visualize = visualize

        # Generate class colors for detection visualization
        self.names = names
        rng = np.random.default_rng()
        self.class_colors = [
            rng.integers(low=0, high=255, size=3, dtype=np.uint8).tolist()
            for _ in self.names
        ]

        self.db_uri = db_uri
        self.db = None
        self.job_id = None

        if self.db_uri:
            self._connect_db()

    def _connect_db(self) -> None:
        engine = create_engine(self.db_uri)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        self.db = session
        job = Job(job_name=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.db.add(job)
        self.db.commit()
        self.job_id = job.job_id

    def update_db(self) -> None:
        track_messages = self._get_track_messages()
        for track_message in track_messages:
            looks = track_message.pop("looks")
            track_message["class_name"] = self.names[track_message.pop("class_id")]
            # check to see if track_id already exists, if so, update it, else add it
            existing_track = self.db.query(Track).filter(Track.track_id == track_message["track_id"]).first()
            if existing_track:
                existing_track.count = track_message["count"]
                existing_track.is_activated = track_message["is_activated"]
                existing_track.state = track_message["state"]
                existing_track.score = track_message["score"]
                existing_track.frame_id = track_message["frame_id"]
                existing_track.time_since_update = track_message["time_since_update"]
                existing_track.location = track_message["location"]
                existing_track.class_name = track_message["class_name"]

            else:
                self.db.add(Track(
                    **track_message,
                    job_id=self.job_id
                ))

            # check to see if track has same number of images, if not, add the images
            existing_images = self.db.query(Detection).filter(Detection.track_id == track_message["track_id"]).all()
            if len(existing_images) != len(looks):
                image = Detection(
                    frame_id=track_message["frame_id"],
                    cropped_image="image",  # replace with image_to_text(looks[-1])
                    track_id=track_message["track_id"]
                )
                self.db.add(image)
        self.db.commit()

    @abstractmethod
    def update(self) -> List[Any]:
        """ Updates track states.

        Returns:
            A list of active tracks.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_track_messages() -> Dict[str, Any]:
        """ Gets a dictionary of track attributes.

        Returns:
            A dictionary of track attributes.
        """
        raise NotImplementedError
