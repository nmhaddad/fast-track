""" ObjectTracker base class """

from abc import ABCMeta, abstractmethod
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..schemas import Base, Detection, Job, Track, Frame
from ..utils import generate_frame_caption, encode_image


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
                 db_uri: Optional[str] = None,
                 use_gpt4v_captions: bool = False):
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

        # database
        self.db_uri = db_uri
        self.use_gpt4v_captions = use_gpt4v_captions
        self.db = None
        self.job_id = None
        if self.db_uri:
            self._connect_db()

    def _connect_db(self) -> None:
        """ Connects to the database and creates tables if they don't exist. """
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
        """ Updates the database with tracks and detections. """
        track_messages = self._get_track_messages()
        for track_message in track_messages:
            crops = track_message.pop("crops")
            track_message["class_name"] = self.names[track_message.pop("class_id")]
            # check to see if track_id already exists, if so, update it, else add it
            existing_track = self.db.query(Track).filter(Track.track_id == track_message["track_id"]).first()
            if existing_track:
                existing_track.count = track_message["count"]
                existing_track.is_activated = track_message["is_activated"]
                existing_track.state = track_message["state"]
                existing_track.score = track_message["score"]
                existing_track.curr_frame_number = track_message["curr_frame_number"]
                existing_track.time_since_update = track_message["time_since_update"]
                existing_track.location = track_message["location"]
                existing_track.class_name = track_message["class_name"]
            else:
                self.db.add(Track(**track_message, job_id=self.job_id))

            # check to see if track has same number of images, if not, add the images
            existing_images = self.db.query(Detection).filter(Detection.track_id == track_message["track_id"]).all()
            if len(existing_images) != len(crops):
                image = Detection(
                    frame_number=track_message["curr_frame_number"],
                    image_base64=base64.b64encode(crops[-1]).decode("utf-8"),
                    track_id=track_message["track_id"]
                )
                self.db.add(image)
        self.db.commit()

    def add_frame(self, frame: np.ndarray, frame_number: int) -> None:
        """ Adds a frame to the tracker.

        Args:
            frame: frame to add.
        """
        frame_base64 = encode_image(frame)
        self.db.add(Frame(
            frame_number=frame_number,
            frame_base64=frame_base64,
            time_created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            gpt4v_caption=generate_frame_caption(frame_base64) if self.use_gpt4v_captions else None,
            job_id=self.job_id
        ))
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
