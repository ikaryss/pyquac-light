import numpy as np

from pyquac_light import SkeletonSpectroscopy


class DummySpectroscopy(SkeletonSpectroscopy):
    def pre_scan(self) -> None:
        print(">> pre_scan: power on, init hardware")

    def pre_column(self, x: float) -> None:
        print(f">> pre_column: stepping to x = {x}")

    def measure_point(self, x: float, y: float) -> float:
        # here you’d talk to your real device; for demo, return random
        z = np.random.random()
        print(f">> measure_point: at (x={x},y={y}) → z={z:.3f}")
        return z

    def post_scan(self) -> None:
        print(">> post_scan: shut off hardware")
