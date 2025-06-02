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
    FloatText,
    Checkbox,
    ToggleButton,
    Accordion,
    Layout,
    HTML,
)


# --- Figure Configuration ---
def create_figure():
    """Initialize and configure the main visualization figure with Matplotlib‐style styling."""
    # Create 2×2 grid
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.85, 0.15],
        row_heights=[0.15, 0.85],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    # Matplotlib‐style axis theme (applied to every subplot)
    common_axis_style = dict(
        showline=True,
        linecolor="black",
        mirror=True,  # draw all four spines
        ticks="outside",
        ticklen=3.5,
        tickwidth=0.8,
        tickcolor="black",
        tickfont=dict(family="sans-serif", size=10, color="black"),
        tickangle=0,  # fixed tick angle
    )

    # Add traces...
    fig.add_trace(
        go.Scatter(
            x=None, y=None, line=dict(color="black", width=0.8), name="horizontal"
        ),
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
                lenmode="fraction",
                len=0.85,
                x=1.05,
                y=0.425,
                yanchor="middle",
                thickness=20,
                outlinewidth=0.8,
                outlinecolor="black",
                ticklen=3.5,
                tickwidth=0.8,
                tickcolor="black",
                tickfont=dict(family="sans-serif", size=10, color="black"),
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
        go.Scatter(
            x=None, y=None, name="vertical", line=dict(color="black", width=0.8)
        ),
        row=2,
        col=2,
    )

    # Crosshairs (hidden by default)
    for cname in ("crosshair_h", "crosshair_v"):
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="black", width=1, dash="dash"),
                name=cname,
                visible=False,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Apply Matplotlib‐style to all axes
    fig.update_xaxes(**common_axis_style)
    fig.update_yaxes(**common_axis_style)

    # Override heatmap‐specific y-axis settings
    fig.update_yaxes(showexponent="none", exponentformat="e", row=2, col=1)
    fig.update_traces(selector=dict(type="heatmap"), zhoverformat=".2f")
    fig.update_xaxes(showexponent="none", exponentformat="e", row=2, col=2)

    # Titles & label visibility
    fig.update_xaxes(title_text="X Axis", row=2, col=1)
    fig.update_yaxes(title_text="Y Axis", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(title_text="Mag, dB", row=1, col=1)
    fig.update_xaxes(title_text="Mag, dB", row=2, col=2)

    # Turn off all trace legends by default
    fig.update_traces(showlegend=False)

    # Final layout (including global separators)
    fig.update_layout(
        width=800,
        height=700,
        font=dict(family="sans-serif", size=10, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=50, r=50, t=20, b=50),
        uirevision="fixed",
        separators=".",  # decimal & thousands separators
    )

    fig.update_traces(hoverinfo="x+y+z")

    return fig


# --- Widget Configuration ---
CONTROL_LAYOUT = Layout(padding="5px", align_items="flex-start")
AUTO_WIDTH = Layout(width="auto", margin="2px 0")
SECTION_TITLE_STYLE = {"font_weight": "bold", "font_size": "14px"}
SECTION_LAYOUT = Layout(margin="10px 0 5px 0")
CONTROL_SPACING = {"padding": "3px 0"}


def create_measurement_controls():
    """Create measurement operation controls."""
    return VBox(
        [
            Button(description="Full Measurement", icon="play", layout=AUTO_WIDTH),
            Button(
                description="Corridor Measurement",
                icon="forward",
                layout=AUTO_WIDTH,
                disabled=True,
            ),
            VBox(
                [
                    Label("Width", layout=AUTO_WIDTH),
                    FloatText(value=0.2, min=0.01, max=0.5, layout=AUTO_WIDTH),
                ],
                layout=Layout(padding="2px"),
            ),
            Box(layout=Layout(height="10px")),  # small gap
            VBox(
                [
                    Label("Sleep (ms)", layout=AUTO_WIDTH),
                    IntText(value=50, min=1, layout=AUTO_WIDTH),
                ],
                layout=Layout(padding="2px"),
            ),
            Box(layout=Layout(height="20px")),  # bigger gap
            Button(description="Stop", icon="stop", layout=AUTO_WIDTH, disabled=True),
        ],
        layout=CONTROL_LAYOUT,
    )


def create_file_controls():
    """Create file operation buttons + extra-text field."""
    save_csv = Button(description="Save CSV", icon="save")
    save_png = Button(description="Save PNG", icon="file-image")
    save_svg = Button(description="Save SVG", icon="file-image")
    save_html = Button(description="Save HTML", icon="html5")

    # extra-text input
    extra_txt = Text(
        value="",
        placeholder="Optional: extra text",
        layout=AUTO_WIDTH,
    )
    extra_box = VBox(
        [Label("Filename suffix:"), extra_txt],
        layout=CONTROL_LAYOUT,
    )
    # example label
    example = Label("Example: exp-HH-MM", layout=Layout(padding="2px 5px"))

    return VBox(
        [
            save_csv,
            save_png,
            save_svg,
            save_html,
            extra_box,
            example,
        ],
        layout=CONTROL_LAYOUT,
    )


def create_interaction_controls():
    """Create interaction mode controls."""
    pick_mode = ToggleButton(description="Point Pick Mode", layout=AUTO_WIDTH)
    show_lines = Checkbox(
        description="Show click lines",
        style={"description_width": "initial"},
        layout=AUTO_WIDTH,
    )
    clear_sel = Button(description="Clear Selection", icon="eraser", layout=AUTO_WIDTH)

    # spacer
    separator = Box(layout=Layout(height="10px"))

    # --- new Data Management block ---
    dm_label = HTML("<b>Data Management</b>", layout=SECTION_LAYOUT)
    drop_btn = Button(description="Drop Picked Points", icon="trash", layout=AUTO_WIDTH)
    cleanup_btn = Button(
        description="Clean Up Corridor", icon="broom", layout=AUTO_WIDTH
    )

    # wrap Width exactly as in measurement section:
    width_input = VBox(
        [
            Label("Width"),
            FloatText(value=0.2, min=0.01, max=0.5, layout=AUTO_WIDTH),
        ],
        layout=CONTROL_LAYOUT,
    )

    clear_all_btn = Button(
        description="Clear All Data",
        icon="exclamation-triangle",
        button_style="danger",
        layout=AUTO_WIDTH,
    )

    return VBox(
        [
            # original controls
            pick_mode,
            show_lines,
            clear_sel,
            # break
            separator,
            # data-management controls, one per line
            dm_label,
            drop_btn,
            cleanup_btn,
            width_input,
            clear_all_btn,
        ],
        layout=CONTROL_LAYOUT,
    )


def create_fit_controls():
    """Create curve fitting controls."""
    return VBox(
        [
            Label(value="Degree", layout=AUTO_WIDTH),
            IntText(value=2, layout=AUTO_WIDTH),
            Button(description="Manual Fit Ridge", icon="check", layout=AUTO_WIDTH),
            Button(description="Auto Fit Ridge", icon="magic", layout=AUTO_WIDTH),
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
            VBox(
                [
                    Label("Save Parent Path"),
                    Text(
                        value="/shared-data/Spectroscopy",
                        placeholder="Abs path, or empty for cwd",
                        layout=AUTO_WIDTH,
                    ),
                ],
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
            create_measurement_controls(),
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
        ["Measurement", "File Ops", "Interaction", "Fitting", "Performance", "Settings"]
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
    message_display = Text(
        placeholder="Status messages will appear here",
        disabled=True,
        layout=Layout(width="1000px", height="30px"),
    )

    footer = VBox(
        [
            HBox([Label("Points:"), coord_display], layout=Layout(padding="2px")),
            HBox([Label("Status:"), message_display], layout=Layout(padding="2px")),
        ],
        layout=Layout(height="100px", padding="10px"),
    )

    # Assemble final layout
    return VBox(
        [HBox([controls, Box(layout=Layout(width="10px")), fig_container]), footer],
        layout=Layout(width="100%", height="850px"),
    )
