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

        self.count = 0

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
                def image_to_text(image):
                    from transformers import AutoProcessor, Blip2ForConditionalGeneration
                    import torch
                    from PIL import Image
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"DEVICE: {device}")
                    image = Image.fromarray(image)
                    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
                    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto", offload_folder="offload")#, load_in_8bit=True)
                    inputs = processor(image, return_tensors="pt").to(device, torch.float16)

                    generated_ids = model.generate(**inputs, max_new_tokens=20)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                    return generated_text
                self.count +=1
                image = Detection(
                    frame_id=track_message["frame_id"],
                    cropped_image=image_to_text(looks[-1]),
                    track_id=track_message["track_id"]
                )
                self.db.add(image)
                print(self.count)
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
