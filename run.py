""" Run file for fast_track """

import cv2
import yaml
from dotenv import load_dotenv

from fast_track import Pipeline
from fast_track.detectors import YOLOv9ONNX
from fast_track.trackers import BYTETracker
from fast_track.databases import SQLDatabase

load_dotenv()


with open('config/coco.yml', 'r') as f:
    config = yaml.safe_load(f)

camera = cv2.VideoCapture(config['data_path'])
detector = YOLOv9ONNX(**config['detector'], names=config['names'], image_shape=(camera.get(3), camera.get(4)))
tracker = BYTETracker(**config['tracker'], names=config['names'])
database = SQLDatabase(**config["db"], class_names=config['names'])

with Pipeline(
    camera=camera,
    detector=detector,
    tracker=tracker,
    database=database,
    outfile=config['outfile']
) as pipeline:
    pipeline.run()
