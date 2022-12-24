""" Processing pipeline with detection and tracking """

import cv2
import numpy as np

from .object_detection import YOLOv7
from .object_tracking import BYTETracker
from .utils import save_video


class Pipeline:

    def __init__(self, config: dict):
        self.config = config
        data_path = 'data/first_test.mp4'

        self.camera = cv2.VideoCapture(data_path)
        self.detector = YOLOv7('models/yolov7-tiny.onnx', config['names'], (self.camera.get(3), self.camera.get(4)))
        self.tracker = BYTETracker()
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.camera.release()
        cv2.destroyAllWindows()

    def run(self) -> None:
        frames = []
        results = []
        counter = 0
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # tracking (image_info = [h, w])
            online_targets = self.tracker.update(np.array(boxes), np.array(scores), frame.shape, (self.camera.get(3), self.camera.get(4)))

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    tx1, ty1, tw, th = tlwh.astype(int)
                    track_str = f"{counter},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    cv2.putText(frame, str(tid), (tx1 + tw, ty1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    # print(f"x1 {tx1}, y1 {ty1}, w{tw}, h {th} - - - imgshape {frame.shape}")
                    
                    # sub_img = frame[ty1:ty1+th, tx1:tx1 + tw]
                    # white_rect = np.ones(sub_img.shape, dtype = np.uint8) * 255
                    # cv2.rectangle(frame, (tx1, ty1), (tx1 + tw, ty1 + th), (255, 0, 0), 1)
                    cv2.circle(frame, (np.mean([tx1, tx1 + tw]).astype(int), np.mean([ty1, ty1 + th]).astype(int)), 3, (0, 0, 255), cv2.FILLED)
                    # weighted = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                    # frame[ty1:ty1+th, tx1:tx1 + tw] = weighted

                    results.append(track_str)
                    print(track_str)
            frames.append(frame)
            # if len(frames) ==30:
            #     break
            counter += 1
        if frames:
            save_video(frames, 'video.avi')

