from trame_client.widgets.core import AbstractElement
from .. import module


class HtmlElement(AbstractElement):
    def __init__(self, _elem_name, children=None, **kwargs):
        super().__init__(_elem_name, children, **kwargs)
        if self.server:
            self.server.enable_module(module)


class AssemblyView(HtmlElement):
    def __init__(self, **kwargs):
        super().__init__(
            "vera-assembly-view",
            **kwargs,
        )
        self._attr_names += [
            "value",
            ("selected_i", "selectedI"),
            ("selected_j", "selectedJ"),
            ("color_preset", "colorPreset"),
            ("color_range", "colorRange"),
            ("active_style", ":activeStyle"),
            "dark",
            "busy",
            "decimals",
        ]
        self._event_names += [
            "click",
        ]

class SurfaceView(HtmlElement):
    def __init__(self, **kwargs):
        super().__init__(
            "vera-core-surface-view",
            **kwargs,
        )
        self._attr_names += [
            "value",
            ("selected_i", "selectedI"),
            ("selected_j", "selectedJ"),
            ("color_preset", "colorPreset"),
            ("color_range", "colorRange"),
            ("active_style", "activeStyle"),
            ("x_labels", "xLabels"),
            ("y_labels", "yLabels"),
            ("aspect_ratio", "aspectRatio"),
            ("cell_size", "cellSize"),
            "dark",
            "busy",
        ]
        self._event_names += [
            "click",
        ]
class AssemblySurfaceView(HtmlElement):
    def __init__(self, **kwargs):
        super().__init__(
            "vera-assembly-surface-view",
            **kwargs,
        )
        self._attr_names += [
            "value",
            ("selected_i", "selectedI"),
            ("selected_j", "selectedJ"),
            ("selected_surface", "selectedSurface"),
            ("color_preset", "colorPreset"),
            ("color_range", "colorRange"),
            ("active_style", "activeStyle"),
            ("cell_size", "cellSize"),
            "dark",
            "busy",
        ]
        self._event_names += [
            "click",
        ]
class CoreView(HtmlElement):
    def __init__(self, **kwargs):
        super().__init__(
            "vera-core-view",
            **kwargs,
        )
        self._attr_names += [
            "value",
            "labels",
            ("selected_i", "selectedI"),
            ("selected_j", "selectedJ"),
            ("aspect_ratio", "aspectRatio"),
            ("color_preset", "colorPreset"),
            ("color_range", "colorRange"),
            ("active_style", ":activeStyle"),
            ("x_labels", "xLabels"),
            ("y_labels", "yLabels"),
            ("assembly_size", "assemblySize"),
            ("core_cols", "coreCols"),
            "decimals",
            "dark",
            "scaling",
            "busy",
        ]
        self._event_names += [
            "click",
        ]

class CoreAxialView(HtmlElement):
    def __init__(self, **kwargs):
        super().__init__(
            "vera-core-axial-view",
            **kwargs,
        )
        self._attr_names += [
            "value",
            "labels",
            "mesh",
            ("x_range", "xRange"),
            ("y_range", "yRange"),
            ("selected_i", "selectedI"),
            ("selected_j", "selectedJ"),
            ("aspect_ratio", "aspectRatio"),
            ("color_preset", "colorPreset"),
            ("color_range", "colorRange"),
            ("x_labels", "xLabels"),
            ("y_labels", "yLabels"),
            ("cell_size", "cellSize"),
            "dark",
            "busy",
        ]
        self._event_names += [
            "click",
        ]
class AxialView(HtmlElement):
    def __init__(self, **kwargs):
        super().__init__(
            "vera-axial-view",
            **kwargs,
        )
        self._attr_names += [
            "value",
            ("selected_i", "selectedI"),
            ("selected_j", "selectedJ"),
            ("color_preset", "colorPreset"),
            ("color_range", "colorRange"),
            ("active_style", ":activeStyle"),
            ("x_labels", "xLabels"),
            ("y_labels", "yLabels"),
            ("x_scale", "xScale"),
            ("y_scale", "yScale"),
            ("x_sizes", "xSizes"),
            ("y_sizes", "ySizes"),
            "dark",
            "busy",
        ]
        self._event_names += [
            "click",
        ]


class ColorMapEditor(HtmlElement):
    def __init__(self, **kwargs):
        super().__init__(
            "vera-color-map-editor",
            **kwargs,
        )
        self._attr_names += [
            "value",
            ("color_preset", "colorPreset"),
            "busy",
        ]
        self._event_names += [
            "input",
        ]

class VerticalColorMapEditor(HtmlElement):
    def __init__(self, **kwargs):
        super().__init__(
            "vera-vertical-color-map-editor",
            **kwargs,
        )
        self._attr_names += [
            "value",
            ("color_preset", "colorPreset"),
            "busy",
            "units",
        ]
        self._event_names += [
            "input",
        ]