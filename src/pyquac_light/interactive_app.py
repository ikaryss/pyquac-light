import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from ipywidgets import VBox, HBox, Layout
import threading
import time
from typing import Optional

from .datatools import Spectroscopy
from .app_layout import build_interface, create_figure


class InteractiveSpectroscopyApp:
    """Interactive spectroscopy application with live data visualization."""

    def __init__(self, spec: Optional[Spectroscopy] = None):
        """
        Initialize the interactive app.

        Args:
            spec: Optional Spectroscopy instance. If None, starts with empty state.
        """
        self.spec = spec
        self.current_x_idx = 0
        self.current_y_idx = 0
        self.timer = None
        self.update_running = False

        # Create the UI components
        self.fig_widget = go.FigureWidget(create_figure())
        self.ui_container = build_interface()

        # Extract control widget references
        self._extract_control_references()

        # Set up event handlers
        self._setup_event_handlers()

        # Initialize display
        self._initialize_display()

    def _extract_control_references(self):
        """Extract references to specific control widgets from the UI."""
        # The controls are in an accordion (first child of the HBox)
        controls_accordion = self.ui_container.children[0].children[0]

        # Performance controls (index 3 in accordion)
        perf_controls = controls_accordion.children[3]
        self.mode_toggle = perf_controls.children[0]  # ToggleButtons "Static"/"Live"
        self.update_interval = perf_controls.children[2]  # IntText for update ms

        # Interaction controls (index 1 in accordion)
        interaction_controls = controls_accordion.children[1]
        self.show_crosshairs = interaction_controls.children[
            1
        ]  # Checkbox "Show click lines"

        # File controls (index 0 in accordion)
        file_controls = controls_accordion.children[0]
        self.load_csv_btn = file_controls.children[0]  # "Load CSV"
        self.save_csv_btn = file_controls.children[1]  # "Save CSV"

        # Fit controls (index 2 in accordion)
        fit_controls = controls_accordion.children[2]
        self.degree_input = fit_controls.children[1]  # IntText for degree
        self.fit_ridge_btn = fit_controls.children[2]  # "Fit Ridge" button
        self.show_fit_toggle = fit_controls.children[3]  # "Show Fit Curve" toggle

        # Footer coordinate display
        footer = self.ui_container.children[1]
        self.coord_display = footer.children[1]  # Text widget for coordinates

    def _setup_event_handlers(self):
        """Set up all event handlers for UI interactions."""
        # Performance controls
        self.mode_toggle.observe(self._on_mode_change, names="value")
        self.update_interval.observe(self._on_interval_change, names="value")

        # Interaction controls
        self.show_crosshairs.observe(self._on_crosshair_toggle, names="value")

        # File operations
        self.load_csv_btn.on_click(self._on_load_csv)
        self.save_csv_btn.on_click(self._on_save_csv)

        # Fit controls
        self.fit_ridge_btn.on_click(self._on_fit_ridge)
        self.show_fit_toggle.observe(self._on_show_fit_toggle, names="value")

        # Click interactions on the main heatmap
        self._setup_click_interactions()

    def _setup_click_interactions(self):
        """Set up click event handling on the main heatmap."""

        def on_click(trace, points, selector):
            if points.point_inds and self.spec is not None:
                # Get clicked coordinates
                x_val = points.xs[0]
                y_val = points.ys[0]

                # Find nearest grid indices
                x_idx = self._find_nearest_x_index(x_val)
                y_idx = self._find_nearest_y_index(y_val)

                # Store current indices
                self.current_x_idx = x_idx
                self.current_y_idx = y_idx

                # Update slices
                self._update_horizontal_slice(y_idx)
                self._update_vertical_slice(x_idx)

                # Update crosshairs if enabled
                self._update_crosshairs(x_val, y_val)

                # Update coordinate display
                self._update_coordinate_display(x_val, y_val)

        # Attach to heatmap trace (index 1 in the figure)
        self.fig_widget.data[1].on_click(on_click)

    def _initialize_display(self):
        """Initialize the display with current data."""
        if self.spec is not None:
            self._update_heatmap()
            # Set initial slices to middle of the data
            mid_x = len(self.spec.x_arr) // 2
            mid_y = len(self.spec.y_arr) // 2
            self.current_x_idx = mid_x
            self.current_y_idx = mid_y
            self._update_horizontal_slice(mid_y)
            self._update_vertical_slice(mid_x)

        # Start live updates if mode is Live
        if self.mode_toggle.value == "Live":
            self._start_live_updates()

    def _find_nearest_x_index(self, x_val: float) -> int:
        """Find the nearest x grid index for a given x value."""
        if self.spec is None:
            return 0
        return int(np.argmin(np.abs(self.spec.x_arr - x_val)))

    def _find_nearest_y_index(self, y_val: float) -> int:
        """Find the nearest y grid index for a given y value."""
        if self.spec is None:
            return 0
        return int(np.argmin(np.abs(self.spec.y_arr - y_val)))

    def _update_heatmap(self):
        """Update the main heatmap with current spectroscopy data."""
        if self.spec is None:
            return

        with self.fig_widget.batch_update():
            # Update heatmap data
            self.fig_widget.data[1].z = self.spec.z_matrix
            self.fig_widget.data[1].x = self.spec.x_arr
            self.fig_widget.data[1].y = self.spec.y_arr

    def _update_horizontal_slice(self, y_idx: int):
        """Update the horizontal slice (top panel) at given y index."""
        if self.spec is None:
            return

        z_slice = self.spec.z_matrix[y_idx, :]
        # Mask out NaN values
        valid_mask = ~np.isnan(z_slice)

        with self.fig_widget.batch_update():
            self.fig_widget.data[0].x = self.spec.x_arr[valid_mask]
            self.fig_widget.data[0].y = z_slice[valid_mask]

    def _update_vertical_slice(self, x_idx: int):
        """Update the vertical slice (right panel) at given x index."""
        if self.spec is None:
            return

        z_slice = self.spec.z_matrix[:, x_idx]
        # Mask out NaN values
        valid_mask = ~np.isnan(z_slice)

        with self.fig_widget.batch_update():
            self.fig_widget.data[3].x = z_slice[valid_mask]
            self.fig_widget.data[3].y = self.spec.y_arr[valid_mask]

    def _update_crosshairs(self, x_val: float, y_val: float):
        """Update crosshair lines on the main plot."""
        if not self.show_crosshairs.value or self.spec is None:
            self._hide_crosshair_lines()
            return

        # Add or update crosshair lines
        # We'll add them as additional traces if they don't exist
        self._show_crosshair_lines(x_val, y_val)

    def _show_crosshair_lines(self, x_val: float, y_val: float):
        """Show crosshair lines at the specified coordinates."""
        # Check if crosshair traces already exist (traces 4 and 5)
        if len(self.fig_widget.data) < 6:
            # Add horizontal crosshair line
            self.fig_widget.add_trace(
                go.Scatter(
                    x=self.spec.x_arr,
                    y=[y_val] * len(self.spec.x_arr),
                    mode="lines",
                    line=dict(color="red", width=1, dash="dash"),
                    name="crosshair_h",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            # Add vertical crosshair line
            self.fig_widget.add_trace(
                go.Scatter(
                    x=[x_val] * len(self.spec.y_arr),
                    y=self.spec.y_arr,
                    mode="lines",
                    line=dict(color="red", width=1, dash="dash"),
                    name="crosshair_v",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        else:
            # Update existing crosshair traces
            with self.fig_widget.batch_update():
                self.fig_widget.data[4].x = self.spec.x_arr
                self.fig_widget.data[4].y = [y_val] * len(self.spec.x_arr)
                self.fig_widget.data[5].x = [x_val] * len(self.spec.y_arr)
                self.fig_widget.data[5].y = self.spec.y_arr
                self.fig_widget.data[4].visible = True
                self.fig_widget.data[5].visible = True

    def _hide_crosshair_lines(self):
        """Hide crosshair lines."""
        if len(self.fig_widget.data) >= 6:
            with self.fig_widget.batch_update():
                self.fig_widget.data[4].visible = False
                self.fig_widget.data[5].visible = False

    def _update_coordinate_display(self, x_val: float, y_val: float):
        """Update the coordinate display in the footer."""
        self.coord_display.value = f"({x_val:.3f}, {y_val:.3e})"

    # Event handlers
    def _on_mode_change(self, change):
        """Handle performance mode changes."""
        if change["new"] == "Live":
            self._start_live_updates()
        else:  # Static
            self._stop_live_updates()

    def _on_interval_change(self, change):
        """Handle update interval changes."""
        # If currently in live mode, restart with new interval
        if self.mode_toggle.value == "Live":
            self._stop_live_updates()
            self._start_live_updates()

    def _on_crosshair_toggle(self, change):
        """Handle crosshair visibility toggle."""
        if change["new"]:
            # Show crosshairs at current position
            if self.spec is not None:
                x_val = self.spec.x_arr[self.current_x_idx]
                y_val = self.spec.y_arr[self.current_y_idx]
                self._show_crosshair_lines(x_val, y_val)
        else:
            # Hide crosshairs
            self._hide_crosshair_lines()

    def _on_load_csv(self, button):
        """Handle CSV loading."""
        # For now, just a placeholder - would need file dialog in real implementation
        print("Load CSV clicked - implement file dialog")

    def _on_save_csv(self, button):
        """Handle CSV saving."""
        if self.spec is not None:
            # For now, save to a default filename
            self.spec.save_csv("spectroscopy_data.csv")
            print("Data saved to spectroscopy_data.csv")
        else:
            print("No data to save")

    def _on_fit_ridge(self, button):
        """Handle ridge fitting."""
        if self.spec is None:
            print("No data loaded for fitting")
            return

        try:
            degree = self.degree_input.value
            ridge = self.spec.fit_ridge(deg=degree)

            # Update the fit curve trace (index 2)
            x_fit = self.spec.x_arr
            y_fit = ridge(x_fit)

            with self.fig_widget.batch_update():
                self.fig_widget.data[2].x = x_fit
                self.fig_widget.data[2].y = y_fit
                self.fig_widget.data[2].visible = self.show_fit_toggle.value

            print(f"Ridge fitted with degree {degree}")

        except Exception as e:
            print(f"Ridge fitting failed: {e}")

    def _on_show_fit_toggle(self, change):
        """Handle fit curve visibility toggle."""
        if len(self.fig_widget.data) > 2:
            self.fig_widget.data[2].visible = change["new"]

    def _start_live_updates(self):
        """Start the live update timer."""
        if self.update_running:
            return

        self.update_running = True

        def update_loop():
            while self.update_running and self.mode_toggle.value == "Live":
                self._update_heatmap()
                time.sleep(self.update_interval.value / 1000.0)  # Convert ms to seconds

        self.timer = threading.Thread(target=update_loop, daemon=True)
        self.timer.start()

    def _stop_live_updates(self):
        """Stop the live update timer."""
        self.update_running = False
        if self.timer and self.timer.is_alive():
            self.timer.join(timeout=1.0)

    def get_widget(self) -> VBox:
        """Get the complete UI widget."""
        # Replace the figure in the UI container with our interactive figure
        ui_hbox = self.ui_container.children[0]
        fig_container = ui_hbox.children[2]  # The VBox containing the figure
        fig_container.children = (self.fig_widget,)

        return self.ui_container


def launch_app(spec: Optional[Spectroscopy] = None) -> VBox:
    """
    Launch the interactive spectroscopy application.

    Args:
        spec: Optional Spectroscopy instance. If None, starts with empty state
              and user can load data via "Load CSV" button.

    Returns:
        VBox widget ready for display in notebook
    """
    app = InteractiveSpectroscopyApp(spec)
    return app.get_widget()
