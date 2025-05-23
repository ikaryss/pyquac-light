from .datatools import Spectroscopy, peak_detection, RandomSpectroscopy
from .interactive_app import launch_app, InteractiveSpectroscopyApp

__all__ = [
    "Spectroscopy",
    "RandomSpectroscopy",
    "peak_detection",
    "launch_app",
    "InteractiveSpectroscopyApp",
]


def hello() -> str:
    return "Hello from pyquac-light!"
