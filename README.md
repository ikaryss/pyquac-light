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
from pyquac_light import InteractiveSpectroscopyApp

app = InteractiveSpectroscopyApp(spec)
widget = app.get_widget()
widget  # Display in Jupyter

# Access picked points and fitted ridge
picked_points = app.picked_points  # List of (x, y) tuples
fitted_ridge = app.fitted_ridge    # np.poly1d object or None
```

### Custom measurements

```python
from pyquac_light import SkeletonSpectroscopy
import numpy as np

class MyDevice(SkeletonSpectroscopy):
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

## Key Features

- **Interactive widgets**: Real-time visualization with Plotly and ipywidgets
- **Peak detection**: Outlier-based peak finding algorithm
- **Ridge fitting**: Polynomial fitting for spectroscopy ridges
- **Corridor scanning**: Targeted measurements around fitted ridges
- **Data export**: Save as CSV, PNG, SVG, or HTML
- **Extensible**: Template-based system for custom measurement devices
- **Live updates**: Real-time data visualization during measurements
