from .datatools import (
    Spectroscopy,
    peak_detection,
    RandomSpectroscopy,
    SkeletonSpectroscopy,
)
from .interactive_app import launch_app, InteractiveSpectroscopyApp

__all__ = [
    "Spectroscopy",
    "RandomSpectroscopy",
    "SkeletonSpectroscopy",
    "peak_detection",
    "launch_app",
    "InteractiveSpectroscopyApp",
]


def hello() -> str:
    return "Hello from pyquac-light!"
