import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from ipywidgets import (
    VBox,
    HBox,
    Box,
    Label,
    Text,
    Button,
    IntText,
    Checkbox,
    ToggleButton,
    Accordion,
    Layout,
)


# --- Figure Configuration ---
def create_figure():
    """Initialize and configure the main visualization figure."""
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.85, 0.15],
        row_heights=[0.15, 0.85],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    # Add visualization traces
    fig.add_trace(
        go.Scatter(x=None, y=None, mode="lines", name="horizontal"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=None,
            x=None,
            y=None,
            colorscale="Viridis",
            colorbar=dict(
                lenmode="fraction", len=0.85, x=1.05, y=0.425, yanchor="middle"
            ),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=None,
            y=None,
            mode="lines",
            name="fit",
            line_color="#000000",
            line_width=3,
            visible=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=None, y=None, mode="lines", name="vertical"),
        row=2,
        col=2,
    )

    # Configure axes
    fig.update_xaxes(title_text="X Axis", row=2, col=1)
    fig.update_yaxes(title_text="Y Axis", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(title_text="Mag, dB", row=1, col=1)
    fig.update_xaxes(title_text="Mag, dB", row=2, col=2)

    fig.update_layout(width=800, height=700, margin=dict(l=50, r=50, t=20, b=50))

    return fig


# --- Widget Configuration ---
CONTROL_LAYOUT = Layout(padding="5px", align_items="flex-start")
AUTO_WIDTH = Layout(width="auto")


def create_file_controls():
    """Create file operation buttons."""
    return VBox(
        [
            Button(description="Load CSV", icon="folder-open"),
            Button(description="Save CSV", icon="save"),
            Button(description="Save PNG", icon="file-image"),
            Button(description="Save SVG", icon="file-image"),
            Button(description="Save HTML", icon="html5"),
        ],
        layout=CONTROL_LAYOUT,
    )


def create_interaction_controls():
    """Create interaction mode controls."""
    return VBox(
        [
            ToggleButton(description="Point Pick Mode", layout=AUTO_WIDTH),
            Checkbox(
                description="Show click lines",
                style={"description_width": "initial"},
                layout=AUTO_WIDTH,
            ),
            Button(description="Clear Selection", icon="eraser", layout=AUTO_WIDTH),
        ],
        layout=CONTROL_LAYOUT,
    )


def create_fit_controls():
    """Create curve fitting controls."""
    return VBox(
        [
            Label(value="Degree", layout=AUTO_WIDTH),
            IntText(value=2, layout=AUTO_WIDTH),
            Button(description="Fit Ridge", icon="check", layout=AUTO_WIDTH),
            ToggleButton(
                description="Show Fit Curve", icon="chart-line", layout=AUTO_WIDTH
            ),
        ],
        layout=CONTROL_LAYOUT,
    )


def create_performance_controls():
    """Create performance tuning controls."""
    return VBox(
        [
            widgets.ToggleButtons(
                options=["Static", "Live"],
                value="Live",
                description="Mode",
                style={"description_width": "initial"},
                layout=AUTO_WIDTH,
            ),
            Label(value="Update ms", layout=AUTO_WIDTH),
            IntText(value=1000, layout=AUTO_WIDTH),
        ],
        layout=CONTROL_LAYOUT,
    )


def create_axis_settings():
    """Create axis configuration controls."""
    return VBox(
        [
            VBox(
                [Label("X Label"), Text(value="X Axis", layout=AUTO_WIDTH)],
                layout=CONTROL_LAYOUT,
            ),
            VBox(
                [Label("Y Label"), Text(value="Y Axis", layout=AUTO_WIDTH)],
                layout=CONTROL_LAYOUT,
            ),
        ],
        layout=CONTROL_LAYOUT,
    )


# --- UI Assembly ---
def build_interface():
    """Construct and return the complete application interface."""
    # Create controls accordion
    controls = Accordion(
        children=[
            create_file_controls(),
            create_interaction_controls(),
            create_fit_controls(),
            create_performance_controls(),
            create_axis_settings(),
        ],
        layout=Layout(width="250px", overflow="auto"),
    )

    # Set accordion section titles
    for idx, title in enumerate(
        ["File Ops", "Interaction", "Fitting", "Performance", "Settings"]
    ):
        controls.set_title(idx, title)

    # Create visualization area
    fig_widget = go.FigureWidget(create_figure())
    fig_container = VBox([fig_widget], layout=Layout(width="800px"))

    # Create footer
    coord_display = Text(
        placeholder="(x, y) pairs will appear here",
        disabled=True,
        layout=Layout(width="1000px", height="30px"),
    )
    footer = HBox(
        [Label("Points:"), coord_display],
        layout=Layout(padding="5px", height="40px", align_items="center"),
    )

    # Assemble final layout
    return VBox(
        [HBox([controls, Box(layout=Layout(width="10px")), fig_container]), footer],
        layout=Layout(width="100%", height="800px"),
    )
