import ipywidgets as widgets
from plotly.graph_objs import FigureWidget, Heatmap, Scatter
from typing import Tuple, List


class SpectroscopyApp:
    """
    Jupyter app for live interaction with a Spectroscopy instance.
    """

    def __init__(self, spec, width: int = 600, height: int = 400):
        self.spec = spec
        # Track manual picks
        self.manual_peaks: List[Tuple[float, float]] = []

        # Plot setup
        self.fig = FigureWidget()
        self.heatmap = Heatmap(
            z=self.spec.z_matrix,
            x=self.spec.x_arr,
            y=self.spec.y_arr,
            colorscale="Viridis",
        )
        self.peak_scatter = Scatter(
            x=[], y=[], mode="markers", marker=dict(color="red", size=8)
        )
        self.ridge_scatter = Scatter(
            x=[], y=[], mode="lines", line=dict(color="white", width=2)
        )
        self.fig.add_traces([self.heatmap, self.peak_scatter, self.ridge_scatter])
        self.fig.update_layout(width=width, height=height)

        # Widgets
        self.pick_toggle = widgets.ToggleButton(
            value=False,
            description="Pick Mode",
            button_style="info",
            layout=widgets.Layout(width="80px"),
        )
        self.clear_button = widgets.Button(
            description="Clear Picks", layout=widgets.Layout(width="80px")
        )
        self.deg_int = widgets.BoundedIntText(
            value=2,
            min=1,
            max=10,
            description="Degree",
            layout=widgets.Layout(width="120px"),
        )
        self.fit_button = widgets.Button(
            description="Fit Ridge",
            button_style="success",
            layout=widgets.Layout(width="80px"),
        )
        self.show_ridge_cb = widgets.Checkbox(
            value=True, description="Show Ridge", layout=widgets.Layout(width="100px")
        )

        # Layout
        control_box = widgets.HBox(
            [
                self.pick_toggle,
                self.clear_button,
                self.deg_int,
                self.fit_button,
                self.show_ridge_cb,
            ]
        )
        self.app = widgets.VBox([self.fig, control_box])

        # Event handlers
        self.fig.data[0].on_click(self._on_click)
        self.clear_button.on_click(self._on_clear)
        self.fit_button.on_click(self._on_fit)
        self.show_ridge_cb.observe(self._on_show_ridge, names="value")

    def _on_click(self, trace, points, selector):
        if not self.pick_toggle.value:
            return
        for pt in points.point_inds:
            x = trace.x[pt]
            y = trace.y[pt]
            self.manual_peaks.append((x, y))
        self._update_peaks()

    def _update_peaks(self):
        xs, ys = zip(*self.manual_peaks) if self.manual_peaks else ([], [])
        with self.fig.batch_update():
            self.peak_scatter.x = xs
            self.peak_scatter.y = ys

    def _on_clear(self, _):
        self.manual_peaks.clear()
        self._update_peaks()

    def _on_fit(self, _):
        self.spec.ridge = self.spec.fit_ridge(
            deg=self.deg_int.value, manual_peaks=self.manual_peaks
        )
        self._update_ridge()

    def _update_ridge(self):
        if not hasattr(self.spec, "ridge"):
            return
        xs = self.spec.x_arr
        ys = self.spec.ridge(xs)
        with self.fig.batch_update():
            self.ridge_scatter.x = xs
            self.ridge_scatter.y = ys
            self.ridge_scatter.visible = self.show_ridge_cb.value

    def _on_show_ridge(self, change):
        self.ridge_scatter.visible = change["new"]

    def display(self):
        display(self.app)


def start_spectroscopy_app(spec: "Spectroscopy", **kwargs) -> SpectroscopyApp:
    """
    Create and display the spectroscopy tracking app connected to `spec`.
    """
    app = SpectroscopyApp(spec, **kwargs)
    app.display()
    return app
