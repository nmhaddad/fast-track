# Fast-Track 🚀 Real-Time Object Tracking Pipelines

Installable Python package for object tracking pipelines with YOLO-NAS, YOLOv8, and YOLOv7 object detectors and BYTETracker object tracking.

UPDATES: Now supporting YOLO-NAS! [Try it out now with Gradio.](#run-the-demo)

![Try out the Gradio Demo!](media/gradio_demo.png)

## Installation:

Package is installable with Python 3.8, 3.9, and 3.10

1. `git clone <repo> && cd <repo>`
1. `pip install .`
1. To use ByteTrack object tracking, run: `pip install .[bytetrack]`

## Running:

1. Example usage:
    ```
    import cv2
    import yaml

    from fast_track import Pipeline
    from fast_track.object_detection import YOLONAS
    from fast_track.object_tracking import BYTETracker


    if __name__ == '__main__':
        with open('config/coco.yml', 'r') as f:
            config = yaml.safe_load(f)

        camera = cv2.VideoCapture(config['data_path'])
        detector = YOLONAS(**config['detector'], names=config['names'], image_shape=(camera.get(3), camera.get(4)))
        tracker = BYTETracker(**config['tracker'], names=config['names'])

        with Pipeline(camera=camera, detector=detector, tracker=tracker, outfile=config['outfile']) as p:
            p.run()
    ```

## Run the Demo

Follow the installation instructions above, then install Gradio with `pip install .[gradio]`

Finally, launch the app with `python app.py`

## Contact:
Author: Nate Haddad - nhaddad2112[at]gmail[dot]com

## License:
[See LICENSE.txt](LICENSE.txt)

[See YOLO-NAS LICENSE.md](fast_track/object_detection/third_party/yolo_nas/LICENSE.md)

[See YOLO-NAS LICENSE.YOLONAS.md](LICENSE.YOLONAS.md)

## References:
[1] Jocher, Glenn; "YOLOv8 in PyTorch > ONNX > CoreML > TFLite"; https://github.com/ultralytics/; 2023; [Online]. Available: https://github.com/ultralytics/ultralytics 

[2] Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark; "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"; https://github.com/WongKinYiu/; 2022; [Online]. Available: https://github.com/WongKinYiu/yolov7

[3] Gorordo, Ibai; "ONNX YOLOv7 Object Detection"; https://github.com/ibaiGorordo/; 2022; [Online]. Available: https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection

[4] Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang; "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"; https://github.com/ifzhang; 2022; [Online]. Available: https://github.com/ifzhang/ByteTrack

[5] Aharon, Shay and Louis-Dupont and Ofri Masad and Yurkova, Kate and Lotem Fridman and Lkdci and Khvedchenya, Eugene and Rubin, Ran and Bagrov, Natan and Tymchenko, Borys and Keren, Tomer and Zhilko, Alexander and Eran-Deci; "Super-Gradients"; https://github.com/Deci-AI/super-gradients; 2023; [Online]. Available: https://github.com/Deci-AI/super-gradients
