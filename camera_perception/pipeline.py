""" Processing pipeline with detection and tracking """

import cv2

from .object_detection import YOLOv7
from .utils import save_video


class Pipeline:

    def __init__(self, config: dict):
        self.config = config
        data_path = 'data/first_test.mp4'

        self.camera = cv2.VideoCapture(data_path)
        self.detector = YOLOv7('models/yolov7-tiny.onnx', config['names'], (self.camera.get(3), self.camera.get(4)))
        # self.tracker - BYTETracker()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.camera.release()
        cv2.destroyAllWindows()

    def run(self) -> None:
        frames = []
        while True:

            ret, frame = self.camera.read()

            if not ret:
                break

            # detection
            class_ids, scores, boxes = self.detector.detect(frame)

            for id, score, box in zip(class_ids, scores, boxes):
                x1, y1, x2, y2 = box.astype(int)

                print(id, score, box)
                cv2.putText(frame, self.detector.names[id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # tracking
            # online_targets = tracker.update(dets, info_imgs, img_size)


            frames.append(frame)
            # if len(frames) ==30:
            #     break
        if frames:
            save_video(frames, 'video.avi')

