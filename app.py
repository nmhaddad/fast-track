""" Gradio application for fast_track. """

import cv2
import gradio as gr
import yaml

from fast_track import Pipeline
from fast_track.object_detection import get_detector
from fast_track.object_tracking import get_tracker


def run_fast_track(input_video: str, detector_name: str, tracker_name: str) -> str:
    """ Runs a fast_track pipeline with a selected detector and tracker.

    Args:
        input_video: path to processing video.
        detector_name: name of detector to use.
        tracker_name: name of tracker to use.

    Returns:
        A path to an output video.
    """
    with open('config/coco.yml', 'r') as f:
        config = yaml.safe_load(f)
    camera = cv2.VideoCapture(input_video)
    detector_params = config["detector"]
    detector_params.pop("weights_path")
    detector = get_detector(detector_name=detector_name,
                            names=config["names"],
                            image_shape=(camera.get(3), camera.get(4)),
                            detector_params=detector_params)
    tracker = get_tracker(tracker_name=tracker_name,
                          names=config["names"],
                          tracker_params=config["tracker"])
    with Pipeline(camera=camera, detector=detector, tracker=tracker) as p:
        outfile = p.run()
    return outfile


with gr.Blocks() as demo:
    gr.Markdown("Fast-Track Demo")
    with gr.Row():
        with gr.Column():
            input_video = gr.PlayableVideo(label="Input Video", interactive=True)
            detector = gr.Dropdown(label="Object Detector",
                                   choices=[
                                       "YOLO-NAS L",
                                       "YOLO-NAS M",
                                       "YOLO-NAS S",
                                       "YOLOv8 N",
                                       "YOLOv8 S",
                                       "YOLOv8 M",
                                       "YOLOv8 X",
                                       ],
                                    value="YOLO-NAS M")
            tracker = gr.Dropdown(label="Object Tracker",
                                  choices=["ByteTrack"],
                                  value="ByteTrack")
            btn = gr.Button("Run", variant="primary")
        with gr.Column():
            output_video = gr.Video(label="Output Video")
        btn.click(fn=run_fast_track, inputs=[input_video, detector, tracker], outputs=[output_video])

if __name__ == "__main__":
    demo.launch()
