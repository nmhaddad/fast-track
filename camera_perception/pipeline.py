""" Processing pipeline with detection and tracking """

import cv2


from .object_detection import YOLOv7
from .utils import save_video


class Pipeline:

    def __init__(self):

        self.detector = YOLOv7('models/yolov7-tiny.onnx')
        data_path = 'data/first_test.mp4'

        self.camera = cv2.VideoCapture(data_path)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.camera.release()
        cv2.destroyAllWindows()

    def run(self):
        frames = []
        while True:

            ret, frame = self.camera.read()

            if not ret:
                break


            # detection

            # tracking




            frames.append(frame)
        if frames:
            save_video(frames, 'video.mp4')

