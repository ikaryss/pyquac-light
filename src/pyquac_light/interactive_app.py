"jupyter ipywidgets+plotly app functionality"

import os
import threading
import time
import datetime
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from ipywidgets import VBox

from .datatools import Spectroscopy
from .app_layout import build_interface, create_figure


class InteractiveSpectroscopyApp:
    """Interactive spectroscopy application with live data visualization."""

    # Default settings for the application
    DEFAULT_SETTINGS = {
        "x_label": "X Axis",
        "y_label": "Y Axis",
        "z_label": "Intensity",
        "save_parent_path": "/shared-data/Spectroscopy",
        "update_interval_ms": 1000,
        "corridor_width": 0.2,
        "sleep_ms": 50,
        "polynomial_degree": 2,
    }

    def __init__(self, spec: Spectroscopy, default_settings: dict | None = None):
        """
        Initialize the interactive app.

        Args:
            spec: Spectroscopy instance with data to visualize and analyze.
            default_settings: Optional dictionary with default settings for the app.
                Available keys:
                - 'x_label' (str): Default X-axis label, default: 'X Axis'
                - 'y_label' (str): Default Y-axis label, default: 'Y Axis'
                - 'z_label' (str): Default Z-axis label for slice plots, default: 'Intensity'
                - 'save_parent_path' (str): Default save directory path, default: '/shared-data/Spectroscopy'
                - 'update_interval_ms' (int): Default live update interval in milliseconds, default: 1000
                - 'corridor_width' (float): Default corridor width fraction, default: 0.2
                - 'sleep_ms' (int): Default sleep time between measurements in milliseconds, default: 50
                - 'polynomial_degree' (int): Default polynomial degree for ridge fitting, default: 2

                If only some keys are provided, the remaining settings will use built-in defaults.
                All settings can still be modified temporarily through the UI.
        """
        self.spec = spec

        # Merge user settings with defaults (partial dict support)
        self.settings = self.DEFAULT_SETTINGS.copy()
        if default_settings:
            self.settings.update(default_settings)
        self.current_x_idx = 0
        self.current_y_idx = 0
        self.timer = None
        self.update_running = False

        # storage for user-picked points
        self.picked_points: list[tuple[float, float]] = []
        # storage for fitted ridge
        self.fitted_ridge: np.poly1d | None = None

        # Measurement state
        self.measurement_thread = None
        self.stop_event = threading.Event()
        self.is_measuring = False

        # Create the UI components
        self.fig_widget = go.FigureWidget(create_figure())
        self.ui_container = build_interface()

        # Extract control widget references
        self._extract_control_references()

        # set initial enable/disable states
        self._update_data_mgmt_buttons()

        # Initially disable fit controls
        self.fit_ridge_btn.disabled = True
        self.auto_fit_ridge_btn.disabled = (
            False  # Always enabled when data is available
        )
        self.show_fit_toggle.disabled = True

        # set initial footer placeholder
        self.coord_display.placeholder = (
            "Point Pick Mode OFF - Click to view coordinates"
        )

        # Set up event handlers
        self._setup_event_handlers()

        # Apply default settings to widgets
        self._apply_default_settings()

        # Initialize display
        self._initialize_display()

    def _apply_default_settings(self):
        """Apply the default settings to the UI widgets."""
        # Apply axis labels
        self.x_label_input.value = self.settings["x_label"]
        self.y_label_input.value = self.settings["y_label"]
        self.z_label_input.value = self.settings["z_label"]

        # Apply save path
        self.parent_path_input.value = self.settings["save_parent_path"]

        # Apply measurement settings
        self.width_input.value = self.settings["corridor_width"]
        self.cleanup_width_input.value = self.settings["corridor_width"]
        self.sleep_input.value = self.settings["sleep_ms"]

        # Apply performance settings
        self.update_interval.value = self.settings["update_interval_ms"]

        # Apply fitting settings
        self.degree_input.value = self.settings["polynomial_degree"]

    def _set_message(self, msg: str):
        """Show a status message (overwriting any previous)."""
        self.message_display.value = msg

    def _get_save_directory(self) -> Path | None:
        """
        Read self.parent_path_input.value, normalize it, verify it exists (or
        fall back to cwd if empty), then create /Date/ClassName subfolders.
        Returns the full Path or None on error (with a message already set).
        """
        raw = self.parent_path_input.value.strip()

        # 1) Determine the parent folder
        if raw:
            # Normalize mixed slashes → native OS separators
            norm = os.path.normpath(raw)
            p = Path(norm).expanduser()
            # Resolve relative fragments (but do not require the folder to already
            # exist if it's relative—resolve() will make it absolute)
            try:
                p = p.resolve(strict=False)
            except Exception:
                # fallback if resolve(strict=False) isn't available
                p = p.absolute()
            # Check existence & that it's a directory
            if not p.exists() or not p.is_dir():
                self._set_message(f"❌ Parent folder does not exist: {raw}")
                return None
        else:
            # Empty field → use current notebook directory
            p = Path().cwd()

        # 2) Build sub‐folders: dd-mm-yy / ClassName
        today = datetime.date.today().strftime("%d-%m-%y")
        clsname = type(self.spec).__name__
        full_dir = p / today / clsname

        # 3) Create them if needed
        try:
            full_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._set_message(f"❌ Could not create folder {full_dir}: {e}")
            return None

        return full_dir

    def _extract_control_references(self):
        """Extract references to specific control widgets from the UI."""
        # The controls are in an accordion (first child of the HBox)
        self.controls_accordion = self.ui_container.children[0].children[0]

        # Measurement controls (index 0 in accordion)
        measurement_controls = self.controls_accordion.children[0]
        self.full_measurement_btn = measurement_controls.children[0]
        self.corridor_measurement_btn = measurement_controls.children[1]
        self.width_input = measurement_controls.children[2].children[
            1
        ]  # FloatText for width
        self.sleep_input = measurement_controls.children[4].children[
            1
        ]  # IntText for sleep
        self.stop_btn = measurement_controls.children[6]

        # Performance controls (index 4 in accordion - was 3, +1 for measurement)
        perf_controls = self.controls_accordion.children[4]
        self.mode_toggle = perf_controls.children[0]  # ToggleButtons "Static"/"Live"
        self.update_interval = perf_controls.children[2]  # IntText for update ms

        # Interaction controls (index 2 in accordion - was 1, +1 for measurement)
        interaction_controls = self.controls_accordion.children[2]
        self.show_crosshairs = interaction_controls.children[1]
        self.point_pick_toggle = interaction_controls.children[0]
        self.clear_selection_btn = interaction_controls.children[2]
        # index 4 = "Data Management" label (we can ignore)
        # index 5 = drop_btn
        # index 6 = cleanup_btn
        # index 7 = width_input
        # index 8 = clear_all_btn
        self.drop_btn = interaction_controls.children[5]
        self.cleanup_btn = interaction_controls.children[6]
        # children[7] is the VBox([Label("Width"), FloatText])
        width_box = interaction_controls.children[7]
        self.cleanup_width_input = width_box.children[1]
        self.clear_all_btn = interaction_controls.children[8]

        # File controls (index 1 in accordion - was 0, +1 for measurement)
        file_controls = self.controls_accordion.children[1]
        self.save_csv_btn = file_controls.children[0]  # "Save CSV"
        self.save_png_btn = file_controls.children[1]  # "Save PNG"
        self.save_svg_btn = file_controls.children[2]  # "Save SVG"
        self.save_html_btn = file_controls.children[3]  # "Save HTML"
        extra_box = file_controls.children[4]
        self.extra_filename_input = extra_box.children[1]
        self.filename_example_label = file_controls.children[5]

        # Settings (index 5 in accordion - was 4, +1 for measurement)
        setting_controls = self.controls_accordion.children[5]
        # X‐Label text is in the first sub‐VBox
        x_box = setting_controls.children[0]  # VBox([Label, Text])
        self.x_label_input = x_box.children[1]  # the Text widget
        # Y‐Label text is in the second sub‐VBox
        y_box = setting_controls.children[1]
        self.y_label_input = y_box.children[1]
        # Z‐Label text is in the third sub‐VBox
        z_box = setting_controls.children[2]
        self.z_label_input = z_box.children[1]
        # Parent‐path is now the fourth
        self.parent_path_input = setting_controls.children[3].children[1]

        # Fit controls (index 3 in accordion - was 2, +1 for measurement)
        fit_controls = self.controls_accordion.children[3]
        self.degree_input = fit_controls.children[1]  # IntText for degree
        self.fit_ridge_btn = fit_controls.children[2]  # "Manual Fit Ridge" button
        self.auto_fit_ridge_btn = fit_controls.children[3]  # "Auto Fit Ridge" button
        self.show_fit_toggle = fit_controls.children[4]  # "Show Fit Curve" toggle

        # Footer coordinate display
        footer = self.ui_container.children[1]
        self.coord_display = footer.children[0].children[1]
        self.message_display = footer.children[1].children[1]

    def _setup_event_handlers(self):
        """Set up all event handlers for UI interactions."""
        # Measurement controls
        self.full_measurement_btn.on_click(self._on_full_measurement)
        self.corridor_measurement_btn.on_click(self._on_corridor_measurement)
        self.stop_btn.on_click(self._on_stop_measurement)

        # Performance controls
        self.mode_toggle.observe(self._on_mode_change, names="value")
        self.update_interval.observe(self._on_interval_change, names="value")

        # Interaction controls
        self.show_crosshairs.observe(self._on_crosshair_toggle, names="value")
        self.point_pick_toggle.observe(self._on_point_pick_toggle, names="value")
        self.clear_selection_btn.on_click(self._on_clear_selection)
        self.drop_btn.on_click(self._on_drop_points)
        self.cleanup_btn.on_click(self._on_clean_up)
        self.clear_all_btn.on_click(self._on_clear_all)

        # File operations
        self.save_csv_btn.on_click(self._on_save_csv)
        self.save_png_btn.on_click(self._on_save_png)
        self.save_svg_btn.on_click(self._on_save_svg)
        self.save_html_btn.on_click(self._on_save_html)
        self.extra_filename_input.observe(self._on_extra_filename_change, names="value")

        # Fit controls
        self.fit_ridge_btn.on_click(self._on_fit_ridge)
        self.auto_fit_ridge_btn.on_click(self._on_auto_fit_ridge)
        self.show_fit_toggle.observe(self._on_show_fit_toggle, names="value")

        # Axis‐label controls
        self.x_label_input.observe(self._on_x_label_change, names="value")
        self.y_label_input.observe(self._on_y_label_change, names="value")
        self.z_label_input.observe(self._on_z_label_change, names="value")

        # Click interactions on the main heatmap
        self._setup_click_interactions()

    def _setup_click_interactions(self):
        """Set up click event handling on the main heatmap."""

        def on_click(trace, points, selector):
            if points.point_inds:
                # Get clicked coordinates
                x_val = points.xs[0]
                y_val = points.ys[0]

                # If in point-pick mode, collect into list and show it
                if self.point_pick_toggle.value:
                    self.picked_points.append((x_val, y_val))
                    formatted = [f"({x:.3f}, {y:.3e})" for x, y in self.picked_points]
                    display = f"X-Subset Collection ({len(self.picked_points)} points): [{', '.join(formatted)}]"
                    self.coord_display.value = display
                    # enable fit button when points exist
                    self.fit_ridge_btn.disabled = False
                    # Update corridor measurement availability
                    self._update_corridor_measurement_state()
                else:
                    # Otherwise just show the last-clicked coordinate
                    self.coord_display.value = f"({x_val:.3f}, {y_val:.4e})"

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

                # Only update the detailed display when not picking points
                if not self.point_pick_toggle.value:
                    self._update_coordinate_display(x_val, y_val)

        # Attach to heatmap trace (index 1 in the figure)
        self.fig_widget.data[1].on_click(on_click)

    def _initialize_display(self):
        """Initialize the display with current data."""
        # build a nested Python list with None in place of NaN
        z_clean = [
            [None if np.isnan(val) else val for val in row]
            for row in self.spec.z_matrix
        ]
        with self.fig_widget.batch_update():
            self.fig_widget.data[1].x = self.spec.x_arr.tolist()
            self.fig_widget.data[1].y = self.spec.y_arr.tolist()
            self.fig_widget.data[1].z = z_clean

            # Apply default axis labels and colorbar title
            self.fig_widget.update_xaxes(
                title_text=self.settings["x_label"], row=2, col=1
            )
            self.fig_widget.update_yaxes(
                title_text=self.settings["y_label"], row=2, col=1
            )
            # self.fig_widget.data[1].colorbar.title = self.settings["z_label"]

            # compute half-cell widths
            dx = float(self.spec.x_arr[1] - self.spec.x_arr[0])
            dy = float(self.spec.y_arr[1] - self.spec.y_arr[0])
            x0, x1 = self.spec.x_arr[0] - dx / 2, self.spec.x_arr[-1] + dx / 2
            y0, y1 = self.spec.y_arr[0] - dy / 2, self.spec.y_arr[-1] + dy / 2

            #  – main heatmap (row 2, col 1): lock both X and Y
            self.fig_widget.update_xaxes(range=[x0, x1], row=2, col=1)
            self.fig_widget.update_yaxes(range=[y0, y1], row=2, col=1)
            #  – horizontal slice (row 1, col 1): lock X only
            self.fig_widget.update_xaxes(range=[x0, x1], row=1, col=1)
            #  – vertical slice (row 2, col 2): lock Y only
            self.fig_widget.update_yaxes(range=[y0, y1], row=2, col=2)

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
        return int(np.argmin(np.abs(self.spec.x_arr - x_val)))

    def _find_nearest_y_index(self, y_val: float) -> int:
        """Find the nearest y grid index for a given y value."""
        return int(np.argmin(np.abs(self.spec.y_arr - y_val)))

    def _update_heatmap(self):
        """Update the main heatmap with current spectroscopy data."""
        # build a nested Python list with None in place of NaN
        z_clean = [
            [None if np.isnan(val) else val for val in row]
            for row in self.spec.z_matrix
        ]

        with self.fig_widget.batch_update():
            # Update heatmap data
            self.fig_widget.data[1].z = z_clean

    def _update_horizontal_slice(self, y_idx: int):
        """Update the horizontal slice (top panel) at given y index."""
        z_slice = self.spec.z_matrix[y_idx, :]
        # Mask out NaN values
        valid_mask = ~np.isnan(z_slice)

        with self.fig_widget.batch_update():
            self.fig_widget.data[0].x = self.spec.x_arr[valid_mask].tolist()
            self.fig_widget.data[0].y = z_slice[valid_mask].tolist()

    def _update_vertical_slice(self, x_idx: int):
        """Update the vertical slice (right panel) at given x index."""
        z_slice = self.spec.z_matrix[:, x_idx]
        # Mask out NaN values
        valid_mask = ~np.isnan(z_slice)

        with self.fig_widget.batch_update():
            self.fig_widget.data[3].x = z_slice[valid_mask].tolist()
            self.fig_widget.data[3].y = self.spec.y_arr[valid_mask].tolist()

    def _update_crosshairs(self, x_val: float, y_val: float):
        """Update crosshair lines on the main plot."""
        if not self.show_crosshairs.value:
            self._hide_crosshair_lines()
            return

        # Add or update crosshair lines
        # We'll add them as additional traces if they don't exist
        self._show_crosshair_lines(x_val, y_val)

    def _show_crosshair_lines(self, x_val: float, y_val: float):
        """Update the pre-added crosshair traces at the specified coordinates."""

        # Compute the two lines
        # Horizontal: across all x at fixed y_val
        xs = list(self.spec.x_arr)
        ys_h = [y_val] * len(xs)

        # Vertical: across all y at fixed x_val
        ys = list(self.spec.y_arr)
        xs_v = [x_val] * len(ys)

        # Update the two pre-added traces (indices 4 and 5)
        with self.fig_widget.batch_update():
            # trace 4 is the horizontal line
            self.fig_widget.data[4].x = xs
            self.fig_widget.data[4].y = ys_h
            self.fig_widget.data[4].visible = True

            # trace 5 is the vertical line
            self.fig_widget.data[5].x = xs_v
            self.fig_widget.data[5].y = ys
            self.fig_widget.data[5].visible = True

    def _hide_crosshair_lines(self):
        """Hide crosshair lines."""
        with self.fig_widget.batch_update():
            self.fig_widget.data[4].visible = False
            self.fig_widget.data[5].visible = False

    def _update_coordinate_display(self, x_val: float, y_val: float):
        """Update the coordinate display in the footer."""
        self.coord_display.value = f"({x_val:.3f}, {y_val:.3e})"

    def _update_corridor_measurement_state(self):
        """Update corridor measurement button state based on fitted ridge."""
        self.corridor_measurement_btn.disabled = (
            self.fitted_ridge is None or self.is_measuring
        )

    def _update_measurement_button_states(self):
        """Update measurement button states based on current measurement status."""
        self.full_measurement_btn.disabled = self.is_measuring
        self._update_corridor_measurement_state()
        self.stop_btn.disabled = not self.is_measuring

    # Measurement event handlers
    def _on_full_measurement(self, button):
        """Handle full measurement button click."""
        if not isinstance(self.spec, Spectroscopy):
            self._set_message("❌ Full measurement requires Spectroscopy instance")
            return

        # Get parameters
        sleep_ms = self.sleep_input.value
        sleep_sec = sleep_ms / 1000.0

        # Get x_subset from picked points if any
        x_subset = None
        if self.picked_points:
            x_subset = [point[0] for point in self.picked_points]

        # Start measurement
        self._start_measurement(
            lambda: self.spec.run_full_scan(
                sleep=sleep_sec, x_subset=x_subset, stop_event=self.stop_event
            ),
            "Full measurement",
        )

    def _on_corridor_measurement(self, button):
        """Handle corridor measurement button click."""
        if not isinstance(self.spec, Spectroscopy):
            self._set_message("❌ Corridor measurement requires Spectroscopy instance")
            return

        if self.fitted_ridge is None:
            self._set_message("❌ No fitted ridge available for corridor measurement")
            return

        # Get parameters
        sleep_ms = self.sleep_input.value
        sleep_sec = sleep_ms / 1000.0
        width_frac = self.width_input.value

        # Get x_subset from picked points if any
        x_subset = None
        if self.picked_points:
            x_subset = [point[0] for point in self.picked_points]

        # Start measurement
        self._start_measurement(
            lambda: self.spec.run_corridor_scan(
                ridge=self.fitted_ridge,
                width_frac=width_frac,
                sleep=sleep_sec,
                x_subset=x_subset,
                stop_event=self.stop_event,
            ),
            "Corridor measurement",
        )

    def _on_stop_measurement(self, button):
        """Handle stop measurement button click."""
        self._stop_measurement()

    def _start_measurement(self, measurement_func, measurement_name):
        """Start a measurement in a separate thread."""
        if self.is_measuring:
            return

        self.is_measuring = True
        self.stop_event.clear()
        self._update_measurement_button_states()
        self._set_message(f"▶️ Starting {measurement_name}...")

        def measurement_wrapper():
            try:
                measurement_func()
                if not self.stop_event.is_set():
                    self._set_message(f"✅ {measurement_name} completed successfully")
                else:
                    self._set_message(f"⏹️ {measurement_name} stopped by user")
            except Exception as e:
                self._set_message(f"❌ {measurement_name} failed: {e}")
            finally:
                self.is_measuring = False
                self._update_measurement_button_states()

        self.measurement_thread = threading.Thread(
            target=measurement_wrapper, daemon=True
        )
        self.measurement_thread.start()

    def _stop_measurement(self):
        """Stop the current measurement."""
        if self.is_measuring:
            self.stop_event.set()
            self._set_message("⏹️ Stopping measurement...")

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
            x_val = self.spec.x_arr[self.current_x_idx]
            y_val = self.spec.y_arr[self.current_y_idx]
            self._show_crosshair_lines(x_val, y_val)
        else:
            # Hide crosshairs
            self._hide_crosshair_lines()

    def _on_save_csv(self, button):
        """Handle CSV saving."""
        outdir = self._get_save_directory()
        if outdir is None:
            return
        ts = datetime.datetime.now().strftime("%H-%M")
        extra = self.extra_filename_input.value.strip()
        suffix = f"-{extra}" if extra else ""
        fname = f"exp-{ts}{suffix}.csv"
        path = os.path.join(outdir, fname)
        self.spec.save_csv(path)
        self._set_message(f"✅ CSV saved to {path}")

    def _on_save_png(self, button):
        outdir = self._get_save_directory()
        if outdir is None:
            return
        ts = datetime.datetime.now().strftime("%H-%M")
        extra = self.extra_filename_input.value.strip()
        suffix = f"-{extra}" if extra else ""
        fname = f"exp-{ts}{suffix}.png"
        path = os.path.join(outdir, fname)
        # only the heatmap trace: assume it's trace index 1
        fig = self.fig_widget
        try:
            fig.write_image(path, scale=2)
            self._set_message(f"✅ PNG saved to {path}")
        except Exception as e:
            self._set_message(f"❌ PNG save failed: {e}")

    def _on_save_svg(self, button):
        outdir = self._get_save_directory()
        if outdir is None:
            return
        ts = datetime.datetime.now().strftime("%H-%M")
        extra = self.extra_filename_input.value.strip()
        suffix = f"-{extra}" if extra else ""
        fname = f"exp-{ts}{suffix}.svg"
        path = os.path.join(outdir, fname)
        fig = self.fig_widget
        try:
            fig.write_image(path, scale=2)
            self._set_message(f"✅ SVG saved to {path}")
        except Exception as e:
            self._set_message(f"❌ SVG save failed: {e}")

    def _on_save_html(self, button):
        outdir = self._get_save_directory()
        if outdir is None:
            return
        ts = datetime.datetime.now().strftime("%H-%M")
        extra = self.extra_filename_input.value.strip()
        suffix = f"-{extra}" if extra else ""
        fname = f"exp-{ts}{suffix}.html"
        path = os.path.join(outdir, fname)
        fig = self.fig_widget
        try:
            fig.write_html(
                path,
                include_plotlyjs="cdn",
                full_html=True,
                config={"displaylogo": False},
            )
            self._set_message(f"✅ HTML saved to {path}")
        except Exception as e:
            self._set_message(f"❌ HTML save failed: {e}")

    def _on_extra_filename_change(self, change):
        txt = change["new"].strip()
        if txt:
            example = f"exp-HH-MM-{txt}"
        else:
            example = "exp-HH-MM"
        self.filename_example_label.value = f"Example: {example}"

    def _update_data_mgmt_buttons(self):
        # Drop only if there are picked points
        self.drop_btn.disabled = len(self.picked_points) == 0
        # Clean-up only if a ridge is fitted
        self.cleanup_btn.disabled = self.fitted_ridge is None
        # Width input likewise only makes sense if there’s a ridge
        self.cleanup_width_input.disabled = self.fitted_ridge is None
        # Clear all always enabled (or you could only enable if spec has any data)

        # put a “•” on the Interaction tab if any points are picked
        base_title = "Interaction"
        if self.picked_points:
            self.controls_accordion.set_title(2, base_title + "\t•")
        else:
            self.controls_accordion.set_title(2, base_title)

    def _on_drop_points(self, button):
        if not self.picked_points:
            return
        xs = [x for x, y in self.picked_points]
        self.spec.drop(x=xs, y=None)
        self._set_message(f"Dropped {len(xs)} points")
        # clear selection
        self.picked_points.clear()
        self.coord_display.value = ""
        self._update_data_mgmt_buttons()
        self._update_heatmap()

    def _on_clean_up(self, button):
        if self.fitted_ridge is None:
            return
        width = self.cleanup_width_input.value
        self.spec.clean_up(ridge=self.fitted_ridge, width_frac=width)
        self._set_message(f"Cleaned up outside ±{width*100:.1f}% corridor")

        # redraw everything
        self._update_data_mgmt_buttons()
        self._update_heatmap()
        # ← refresh both slices so they no longer show dropped data:
        self._update_horizontal_slice(self.current_y_idx)
        self._update_vertical_slice(self.current_x_idx)

    def _on_clear_all(self, button):
        # 1) Clear the raw data
        self.spec.clear()
        self._set_message("All data cleared")

        # 2) Reset picks and fitted ridge
        self.picked_points.clear()
        self.fitted_ridge = None
        self.coord_display.value = ""

        # 3) Hide the fit curve trace (trace index 2) and clear its data
        with self.fig_widget.batch_update():
            self.fig_widget.data[2].x = []
            self.fig_widget.data[2].y = []
            self.fig_widget.data[2].visible = False

        # 4) Disable the “Show Fit Curve” toggle
        self.show_fit_toggle.value = False
        self.show_fit_toggle.disabled = True

        # 5) Refresh button states
        self._update_data_mgmt_buttons()
        self._update_corridor_measurement_state()

        # 6) Redraw heatmap & slices
        self._update_heatmap()
        self._update_horizontal_slice(self.current_y_idx)
        self._update_vertical_slice(self.current_x_idx)

    def _on_fit_ridge(self, button):
        """Handle ridge fitting."""
        if not self.picked_points:
            return

        try:
            degree = self.degree_input.value
            ridge = self.spec.fit_ridge(deg=degree, manual_peaks=self.picked_points)
            self.fitted_ridge = ridge
            x_fit = self.spec.x_arr
            y_fit = ridge(x_fit)
            with self.fig_widget.batch_update():
                self.fig_widget.data[2].x = x_fit.tolist()
                self.fig_widget.data[2].y = y_fit.tolist()
                self.fig_widget.data[2].visible = self.show_fit_toggle.value
            # enable show-fit toggle
            self.show_fit_toggle.disabled = False
            # Update corridor measurement availability
            self._update_corridor_measurement_state()
            self._set_message(
                f"Ridge fitted with degree {degree} and coefficients {ridge}"
            )
            self._update_data_mgmt_buttons()
        except Exception as e:
            self._set_message(f"Ridge fitting failed: {e}")

    def _on_auto_fit_ridge(self, button):
        """Handle automatic ridge fitting."""
        try:
            degree = self.degree_input.value

            # Get x_subset from picked points if any (only x components)
            x_subset = None
            if self.picked_points:
                x_subset = [point[0] for point in self.picked_points]

            # Call fit_ridge without manual_peaks for automatic detection
            ridge = self.spec.fit_ridge(deg=degree, x_subset=x_subset)
            self.fitted_ridge = ridge

            # Update the plot
            x_fit = self.spec.x_arr
            y_fit = ridge(x_fit)
            with self.fig_widget.batch_update():
                self.fig_widget.data[2].x = x_fit.tolist()
                self.fig_widget.data[2].y = y_fit.tolist()
                self.fig_widget.data[2].visible = self.show_fit_toggle.value

            # enable show-fit toggle
            self.show_fit_toggle.disabled = False
            # Update corridor measurement availability
            self._update_corridor_measurement_state()

            subset_info = (
                f" (x_subset: {len(x_subset)} points)"
                if x_subset
                else " (all x values)"
            )
            self._set_message(
                f"Auto ridge fitted with degree {degree}{subset_info} and coefficients {ridge}"
            )
            self._update_data_mgmt_buttons()
        except Exception as e:
            self._set_message(f"Auto ridge fitting failed: {e}")

    def _on_show_fit_toggle(self, change):
        """Handle fit curve visibility toggle."""
        if len(self.fig_widget.data) > 2:
            self.fig_widget.data[2].visible = change["new"]

    def _on_point_pick_toggle(self, change):
        """Toggle between coordinate-display mode and point-pick mode."""
        if change["new"]:
            # entering pick mode — display the existing list
            formatted = [f"({x:.3f}, {y:.3e})" for x, y in self.picked_points]
            display = f"X-Subset Collection ({len(self.picked_points)} points): [{', '.join(formatted)}]"
            self.coord_display.value = display
            self.coord_display.placeholder = ""
        else:
            # exiting pick mode — clear list display and go back to single-coord mode
            self.coord_display.value = ""
            self.coord_display.placeholder = (
                "Point Pick Mode OFF - Click to view coordinates"
            )
        self._update_data_mgmt_buttons()

    def _on_clear_selection(self, button):
        """Clear all collected points."""
        self.picked_points.clear()
        if self.point_pick_toggle.value:
            # update placeholder to reflect empty list
            self.coord_display.value = ""
            self.coord_display.placeholder = "X-Subset Collection (0 points): []"
        # Update button states
        self.fit_ridge_btn.disabled = True
        self._update_corridor_measurement_state()
        self._update_data_mgmt_buttons()

    def _on_x_label_change(self, change):
        new = change["new"]
        # update the X‐axis title of the main heatmap (row 2, col 1)
        with self.fig_widget.batch_update():
            self.fig_widget.update_xaxes(title_text=new, row=2, col=1)

    def _on_y_label_change(self, change):
        new = change["new"]
        # update the Y‐axis title of the main heatmap (row 2, col 1)
        with self.fig_widget.batch_update():
            self.fig_widget.update_yaxes(title_text=new, row=2, col=1)

    def _on_z_label_change(self, change):
        new = change["new"]
        # update the slice plot labels (Z represents the values in the slices)
        with self.fig_widget.batch_update():
            # Horizontal slice Y-axis (row 1, col 1)
            self.fig_widget.update_yaxes(title_text=new, row=1, col=1)
            # Vertical slice X-axis (row 2, col 2)
            self.fig_widget.update_xaxes(title_text=new, row=2, col=2)

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


def launch_app(spec: Spectroscopy, default_settings: dict | None = None) -> VBox:
    """
    Launch the interactive spectroscopy application.

    Args:
        spec: Spectroscopy instance with data to visualize and analyze.
        default_settings: Optional dictionary with default settings for the app.
            See InteractiveSpectroscopyApp.__init__ for available keys.

    Returns:
        VBox widget ready for display in notebook
    """
    app = InteractiveSpectroscopyApp(spec, default_settings)
    return app.get_widget()
