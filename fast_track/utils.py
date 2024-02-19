""" Utilities file """

import base64
from io import BytesIO
import logging
from typing import List

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


COCO_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
    "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def save_video(frames: List[np.ndarray], outfile_path: str, fps: int = 30) -> None:
    """ Saves a list of frames to a video file.
    Args:
        frames: list
            List of frames to process into a video
        outfile_path: str
            Path to write the video to
    """
    h, w, *_ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    outfile = cv2.VideoWriter(outfile_path, fourcc, fps, (w, h))
    for frame in frames:
        outfile.write(frame)
    outfile.release()


def read_names_file(names_file: str) -> List[str]:
    """ Read a list of class names into a Python list.

    Args:
        names_file: path to a names file

    Returns:
        A list of names
    """
    with open(names_file, "r", encoding="utf8") as file:
        names = [line.rstrip() for line in file]
    return names


def encode_image(image: np.ndarray) -> str:
    """ Encodes an image to base64.

    Args:
        image: image to encode

    Returns:
        base64 string
    """
    image = Image.fromarray(image)
    # change mimetype to image/jpeg
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def generate_frame_caption(frame_base64: str, model: str = "gpt-4-vision-preview") -> np.ndarray:
    """ Generates a caption for a frame using GPT-4 Vision.

    Args:
        frame: frame to generate a caption for
        model: model to use for caption generation

    Returns:
        caption: generated caption
    """
    try:
        logger.info("generate_frame_caption | making request to OpenAI")
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}",
                            "detail": "low",
                        },
                    },
                ],
            }],
            max_tokens=300,
        )
        logger.info("generate_frame_caption | response received")
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"generate_frame_caption | error: {e}")
        return None
