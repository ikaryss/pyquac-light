# pyquac-light

Interactive spectroscopy data analysis and visualization library for Jupyter notebooks.

## Installation

### With UV (recommended)

```bash
git clone https://github.com/ikaryss/pyquac-light.git
cd pyquac-light
uv sync
```

### Without UV

```bash
git clone https://github.com/ikaryss/pyquac-light.git
cd pyquac-light
pip install -e .
```

## Quick Start

### Basic data handling

```python
import numpy as np
from pyquac_light import Spectroscopy

# Create grid
x_arr = np.linspace(0, 10, 100)
y_arr = np.linspace(-5, 5, 50)
spec = Spectroscopy(x_arr=x_arr, y_arr=y_arr)

# Load from CSV
spec = Spectroscopy.load_csv('data.csv')

# Write data points
spec.write(x=5.0, y=2.0, z=0.8)

# Save results
spec.save_csv('output.csv')
```

### Interactive visualization

```python
from pyquac_light import InteractiveSpectroscopyApp, launch_app

# Basic usage
app = InteractiveSpectroscopyApp(spec)
widget = app.get_widget()
widget  # Display in Jupyter

# Or use the convenience function
widget = launch_app(spec)
widget  # Display in Jupyter

# Access picked points and fitted ridge
picked_points = app.picked_points  # List of (x, y) tuples
fitted_ridge = app.fitted_ridge    # np.poly1d object or None
```

### Customizing default settings

You can customize the default settings for the interactive app using the `default_settings` parameter:

```python
# Custom settings for a frequency sweep experiment
frequency_settings = {
    'x_label': 'Frequency (GHz)',
    'y_label': 'Power (dBm)',
    'z_label': 'S21 Magnitude (dB)',
    'save_parent_path': './frequency_data',
    'update_interval_ms': 500,
    'corridor_width': 0.15,
    'polynomial_degree': 3
}

app = launch_app(spec, default_settings=frequency_settings)

# Partial settings (other values use built-in defaults)
partial_settings = {
    'x_label': 'Time (μs)',
    'z_label': 'Voltage (V)',
    'sleep_ms': 200
}

app = launch_app(spec, default_settings=partial_settings)
```

#### Available settings

| Setting              | Type  | Default                       | Description                                 |
| -------------------- | ----- | ----------------------------- | ------------------------------------------- |
| `x_label`            | str   | `"X Axis"`                    | X-axis label for main plot                  |
| `y_label`            | str   | `"Y Axis"`                    | Y-axis label for main plot                  |
| `z_label`            | str   | `"Intensity"`                 | Z-axis label for slice plots                |
| `save_parent_path`   | str   | `"/shared-data/Spectroscopy"` | Default save directory                      |
| `update_interval_ms` | int   | `1000`                        | Live update interval in milliseconds        |
| `corridor_width`     | float | `0.2`                         | Default corridor width fraction             |
| `sleep_ms`           | int   | `50`                          | Sleep time between measurements in ms       |
| `polynomial_degree`  | int   | `2`                           | Default polynomial degree for ridge fitting |

**Note:** All settings can still be modified temporarily through the UI during runtime. The `default_settings` parameter only sets the initial values when the app starts.

### Custom measurements

1. When measuring each point separately:

```python
from pyquac_light import SkeletonSpectroscopy
import numpy as np

class MyDevice(SkeletonSpectroscopy):
    def __init__(
        self,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
    ):
        super().__init__(x_arr=x_arr, y_arr=y_arr)

    def pre_scan(self) -> None:
        print(">> pre_scan: power on, init hardware")
        pass

    def pre_column(self, x: float) -> None:
        print(f">> pre_column: stepping to x = {x}")
        pass

    def measure_point(self, x: float, y: float) -> float:
        # here you’d talk to your real device; for demo, return random
        z = np.random.random()
        print(f">> measure_point: at (x={x},y={y}) → z={z:.3f}")
        return z

    def post_scan(self) -> None:
        print(">> post_scan: shut off hardware")
        pass


# Use your device
device = MyDevice(x_arr=x_arr, y_arr=y_arr)
device.run_full_scan(sleep=0.1)
```

2. When measuring full column at time

```python
from pyquac_light import SkeletonSpectroscopy
import numpy as np

class MyDevice(ColumnSkeletonSpectroscopy):
    def __init__(
        self,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
    ):
        super().__init__(x_arr=x_arr, y_arr=y_arr)

    def pre_scan(self) -> None:
        print(">> pre_scan: power on, init hardware")

    def pre_column(self, x: float) -> None:
        print(f">> pre_column: stepping to x = {x}")

    def measure_column(self, x: float) -> np.ndarray:
        # here you’d talk to your real device; for demo, return random
        z_arr = np.random.random(len(self.y_arr))
        print(f">> measure_column: at x={x} → z={z_arr}")
        return z_arr

    def post_scan(self) -> None:
        print(">> post_scan: shut off hardware")
        pass


# Use your device
device = MyDevice(x_arr=x_arr, y_arr=y_arr)
device.run_full_scan(sleep=0.1)
```

## Key Features

- **Interactive widgets**: Real-time visualization with Plotly and ipywidgets
- **Peak detection**: Outlier-based peak finding algorithm
- **Ridge fitting**: Polynomial fitting for spectroscopy ridges
- **Corridor scanning**: Targeted measurements around fitted ridges
- **Data export**: Save as CSV, PNG, SVG, or HTML
- **Extensible**: Template-based system for custom measurement devices
- **Live updates**: Real-time data visualization during measurements
