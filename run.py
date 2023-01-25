""" Run file for camera_perception """

import cv2
import yaml

from fast_track import Pipeline
from fast_track.object_detection import YOLOv8
from fast_track.object_tracking import BYTETracker


if __name__ == '__main__':
    with open('config/coco.yml', 'r') as f:
        config = yaml.safe_load(f)

    camera = cv2.VideoCapture(config['data_path'])
    detector = YOLOv8(**config['detector'], names=config['names'], image_shape=(camera.get(3), camera.get(4)))
    tracker = BYTETracker(**config['tracker'], names=config['names'])

    with Pipeline(camera=camera, detector=detector, tracker=tracker, outfile=config['outfile']) as p:
        p.run()
