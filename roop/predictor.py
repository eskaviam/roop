import threading
import numpy
import opennsfw2
from PIL import Image
from keras import Model

from roop.typing import Frame

PREDICTOR = None
THREAD_LOCK = threading.Lock()
MAX_PROBABILITY = 0.85


def get_predictor() -> Model:
    global PREDICTOR

    print("hello world")
    return PREDICTOR


def clear_predictor() -> None:
    global PREDICTOR
    print("hello world")
    PREDICTOR = None


def predict_frame(target_frame: Frame) -> bool:
    print("hello world")


def predict_image(target_path: str) -> bool:
    print("hello world")


def predict_video(target_path: str) -> bool:
    print("hello world")
