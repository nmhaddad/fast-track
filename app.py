""" Gradio application for fast_track. """

from typing import Optional

import cv2
import gradio as gr
import yaml

from fast_track import Pipeline
from fast_track.utils import COCO_NAMES, read_names_file
from fast_track.object_detection import get_detector
from fast_track.object_tracking import get_tracker


def run_fast_track(input_video: str,
                   detector_type: str,
                   tracker_name: str,
                   custom_detector_type: Optional[str] = None,
                   custom_detector_weights: Optional[str] = None,
                   custom_detector_names: Optional[str] = None) -> str:
    """ Runs a fast_track pipeline with a selected detector and tracker.

    Args:
        input_video: path to processing video.
        detector_name: name of detector to use.
        tracker_name: name of tracker to use.

    Returns:
        A path to an output video.
    """
    camera = cv2.VideoCapture(input_video)
    weights_path = detector_type
    names = COCO_NAMES

    # Load custom detector if all information is provided
    if custom_detector_type and custom_detector_weights and custom_detector_names:
        weights_path = custom_detector_weights.name
        detector_type = custom_detector_type
        names = read_names_file(custom_detector_names.name)

    print(weights_path)
    print(detector_type)
    print(names)

    detector = get_detector(weights_path=weights_path,
                            detector_type=detector_type,
                            names=names,
                            image_shape=(camera.get(3), camera.get(4)))
    tracker = get_tracker(tracker_name=tracker_name,
                          names=names)
    with Pipeline(camera=camera, detector=detector, tracker=tracker) as p:
        outfile = p.run()
    return outfile


with gr.Blocks() as demo:
    gr.Markdown("""
                # Fast-Track Demo 🚀
                ### Upload a video and process it with a Fast-Track pipeline!
                """)
    with gr.Row():
        # Input Column
        with gr.Column():

            input_video = gr.PlayableVideo(label="Input Video", interactive=True)

            # Object Detector
            with gr.Group():
                detector_type = gr.Dropdown(
                    label="Select a Pretrained Model",
                    info="Models are pretrained on MS-COCO",
                    choices=[
                        "YOLO-NAS L",
                        "YOLO-NAS M",
                        "YOLO-NAS S",
                        "YOLOv8 N",
                        "YOLOv8 S",
                        "YOLOv8 M",
                        "YOLOv8 X",
                    ],
                    value="YOLO-NAS M"
                )

                with gr.Accordion(label="Load a Custom Model (Overrides Pretrained Model)", open=False):
                    custom_detector_type = gr.Dropdown(
                        label="Object Detector Type",
                        info="Select a supported object detector type. ONNX models are accepted for YOLOv8 and YOLOv7",
                        choices=[
                            "YOLO-NAS",
                            "YOLOv8",
                            "YOLOv7",
                        ],
                    )
                    custom_detector_weights = gr.File(
                        label="Custom Model Weights",
                        info="Upload a weights file corresponding to the model type provided above",
                        show_label=True,
                        file_types=[".pt", ".onnx"]
                    )
                    custom_detector_names = gr.File(
                        label="Custom Model Names File",
                        info="Upload a list of class names in a txt file",
                        file_types=[".txt"],
                        show_label=True
                    )

            # Object Tracker
            with gr.Group():
                tracker_type = gr.Dropdown(
                    label="Object Tracker",
                    choices=["ByteTrack"],
                    value="ByteTrack"
                )

            # Run
            btn = gr.Button("Run", variant="primary")

        # Output Column
        with gr.Column():
            output_video = gr.Video(label="Output Video")

        inputs = [
            input_video,
            detector_type,
            tracker_type,
            custom_detector_type,
            custom_detector_weights,
            custom_detector_names
        ]
        btn.click(fn=run_fast_track, inputs=inputs, outputs=[output_video])

if __name__ == "__main__":
    demo.launch()
