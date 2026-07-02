import numpy as np

from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

import vtk.util.numpy_support as np_s

from trame.ui.html import DivLayout
from trame.widgets import vtk, vuetify, html

from vera_core.app.core import VeraDataRegistry, VeraDtype
from vera_core.widgets import vera
from ..helpers import is_view_locked


_OPACITY_POINTS = [
    (-10.0, 1.0),
    (-1e-20, 1.0),
    (-5e-21, 0.0),
    (5e-21, 0.0),
    (1e-20, 1.0),
    (10.0, 1.0),
]

_COLOR_POINTS = [
    (0.000000, 0.0, 0.0, 0.5625),
    (0.216992, 0.0, 0.0, 1.0000),
    (0.712975, 0.0, 1.0, 1.0000),
    (0.960965, 0.5, 1.0, 0.5000),
    (1.208960, 1.0, 1.0, 0.0000),
    (1.704940, 1.0, 0.0, 0.0000),
    (1.952930, 0.5, 0.0, 0.0000),
]

_views = {}

def option_for(view_id):
    return {
        "name": f"volume_view_{view_id}",
        "label": "Volume View",
        "icon": "mdi-rotate-3d",
        "allowed_categories": [VeraDtype.PIN.title, VeraDtype.CHANNEL.title, VeraDtype.ASSEMBLY.title]
    }


def _is_active(state, view_id):
    if is_view_locked(state, view_id):
        return False
    option = state[f"grid_view_{view_id}"]
    return bool(option) and option.get("name") == f"volume_view_{view_id}"


def _build_view(server, view_id):
    """Construct one slot's VTK pipeline and template at startup."""
    ren = vtkRenderer()
    ren.SetBackground(0.1176, 0.1176, 0.1176)
    ren_win = vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.OffScreenRenderingOn()

    iren = vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    iren.SetRenderWindow(ren_win)

    opacity_fn = vtkPiecewiseFunction()
    for point in _OPACITY_POINTS:
        opacity_fn.AddPoint(*point)

    color_fn = vtkColorTransferFunction()
    for point in _COLOR_POINTS:
        color_fn.AddRGBPoint(*point)

    volume_property = vtkVolumeProperty()
    volume_property.SetColor(color_fn)
    volume_property.SetScalarOpacity(opacity_fn)

    volume_data = vtkImageData()
    mapper = vtkSmartVolumeMapper()
    mapper.SetInputData(volume_data)
    mapper.SetAutoAdjustSampleDistances(True)

    volume = vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)
    ren.AddVolume(volume)

    ren.GetActiveCamera().Pitch(90)
    ren.GetActiveCamera().OrthogonalizeViewUp()

    ctx = {
        "ren": ren,
        "ren_win": ren_win,
        "iren": iren,
        "color_fn": color_fn,
        "volume_data": volume_data,
        "reset_count": 0,
        "view_update": lambda: None,
        "reset_camera": lambda: None,
    }
    _views[view_id] = ctx
    with DivLayout(server, template_name=f"volume_view_{view_id}") as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style=(
            "flex: 1; min-width: 0;"
            "display: flex; flex-direction: column;"
        )):
            with html.Div(style="flex: 1; min-height: 0; position: relative;"):
                html_view = vtk.VtkRemoteView(
                ren_win,
                ref=f"volume_view_{view_id}",
                interactive_ratio=2,
                interactive_quality=40,
                still_quality=90,
            )
            ctx["view_update"] = html_view.update
            ctx["reset_camera"] = html_view.reset_camera

        with vuetify.VBtn(
            icon=True,
            click=html_view.reset_camera,
            absolute=True,
            style="top: 0; right: 2px",
            small=True,
        ):
            vuetify.VIcon("mdi-crop-free", small=True)

        with html.Div(style=(
            "flex: 0 0 auto; width: 70px; padding: 4px 0;"
            "display: flex; align-self: stretch;"
        )):
            vera.VerticalColorMapEditor(
                v_model=f"color_range_{view_id}_0",
                color_preset="jet",
            )

    return ctx

def _update_volume(server, registry : VeraDataRegistry, view_id):
    state = server.state
    ctx = _views.get(view_id)
    if ctx is None or not _is_active(state, view_id):
        return
    src_id = state[f"selected_src_id_{view_id}"]
    array_name = state[f"selected_array_{view_id}"]
    vera_out_file = registry.get(src_id)
    if vera_out_file is None or not array_name:
        return
    array = vera_out_file.array(array_name)
    if str(array.dataset_type).upper() not in option_for(0)["allowed_categories"]:
        return
    core = vera_out_file.core

    assembly_shape = array.shape[:2]
    reduced_core_map = core.reduced_core_map
    expanded_core_shape = (
        reduced_core_map.shape[0] * assembly_shape[0],
        reduced_core_map.shape[1] * assembly_shape[1],
    )
    volume_shape = (*expanded_core_shape, array.shape[2])
    volume_array = np.zeros(volume_shape, dtype=np.float32)

    for assembly_id in range(array.shape[3]):
        rows, cols = np.where(reduced_core_map == assembly_id + 1)
        if len(rows) == 0:
            continue
        if len(rows) > 1:
            raise ValueError(
                f"Assembly {assembly_id + 1} appears multiple times in "
                f"reduced_core_map; expected one match, found {len(rows)}."
            )
        core_row, core_col = int(rows[0]), int(cols[0])
        row_slice = slice(core_row * assembly_shape[0], (core_row + 1) * assembly_shape[0])
        col_slice = slice(core_col * assembly_shape[1], (core_col + 1) * assembly_shape[1])
        volume_array[row_slice, col_slice] = array[:, :, :, assembly_id]

    volume_array = np.repeat(volume_array, core.axial_mesh_pixels, axis=2)

    scalars = volume_array.transpose(2, 1, 0).ravel()   # single contiguous copy
    vtk_array = np_s.numpy_to_vtk(scalars, deep=True)

    volume_data = ctx["volume_data"]
    volume_data.SetDimensions(*volume_array.shape)
    pd = volume_data.GetPointData()
    while pd.GetNumberOfArrays() > 0:
        pd.RemoveArray(0)
    pd.SetScalars(vtk_array)
    volume_data.Modified()

    if ctx["reset_count"] < 2:
        ctx["ren"].ResetCameraClippingRange()
        ctx["ren"].ResetCamera()
        ctx["reset_count"] += 1
        ctx["reset_camera"]()

    ctx["view_update"]()


def _update_color(server, view_id):
    state = server.state
    ctx = _views.get(view_id)
    if ctx is None or not _is_active(state, view_id):
        return

    color_range = state[f"color_range_{view_id}_0"]
    original_range = (_COLOR_POINTS[0][0], _COLOR_POINTS[-1][0])
    color_fn = ctx["color_fn"]
    color_fn.RemoveAllPoints()
    for row in _COLOR_POINTS:
        new_value = np.interp(row[0], original_range, color_range)
        color_fn.AddRGBPoint(new_value, *row[1:])

    ctx["ren_win"].Render()
    ctx["view_update"]()
 

DARK_BG = (30 / 255, 30 / 255, 30 / 255)
LIGHT_BG = (1.0, 1.0, 1.0)

def _bg_for_theme(is_dark):
    return DARK_BG if is_dark else LIGHT_BG

def _update_background(server, view_id):
    state = server.state
    ctx = _views.get(view_id)
    if ctx is None:
        return
    ctx["ren"].SetBackground(*_bg_for_theme(state["dark_mode"]))
    if _is_active(state, view_id):
        ctx["ren_win"].Render()
        ctx["view_update"]()

def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]
    _build_view(server, view_id)

    @ctrl.add(f"reset_volume_{view_id}_camera")
    def _reset(*args, **kwargs):
        ctx = _views.get(view_id)
        if ctx is None:
            return
        ctx["ren"].ResetCameraClippingRange()
        ctx["ren"].ResetCamera()
        ctx["reset_camera"]()
        ctx["view_update"]()
    
    @state.change("dark_mode")
    def _on_theme_changed(**kwargs):
        _update_background(server, view_id)

    @state.change(f"grid_view_{view_id}", f"locked_{view_id}")
    def _on_slot_changed(**kwargs):
        if _is_active(state, view_id):
            _update_volume(server, registry, view_id)

    @state.change(f"selected_array_{view_id}", f"selected_src_id_{view_id}")
    def _on_selection_changed(**kwargs):
        _update_volume(server, registry, view_id)

    @state.change(f"color_range_{view_id}_0")
    def _on_color_changed(**kwargs):
        _update_color(server, view_id)

    @ctrl.add("on_vera_out_active_state_index_changed")
    def _on_state_index_changed(**kwargs):
        _update_volume(server, registry, view_id)