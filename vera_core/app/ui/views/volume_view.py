import asyncio

import numpy as np
import vtk.util.numpy_support as np_s
from trame.app import asynchronous
from trame.ui.html import DivLayout
from trame.widgets import html, vtk, vuetify
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

from vera_core.data.dtypes import VeraDtype
from vera_core.data.registry import VeraDataRegistry
from vera_core.data.thresholds import apply_thresholds
from vera_core.widgets import vera

from ..helpers import get_thresholds, get_time, is_view_locked

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

_CROP_SUBVOLUME = 1 << 13
_CROP_INVERTED_SUBVOLUME = ((1 << 27) - 1) & ~(1 << 13)
_FIT_POLL_S = 0.1
_FIT_TIMEOUT_TICKS = 50  # ~5 s


def _camera_state(cam):
    """Serialize the parts of a vtkCamera that define orientation and zoom."""
    return {
        "position": list(cam.GetPosition()),
        "focal_point": list(cam.GetFocalPoint()),
        "view_up": list(cam.GetViewUp()),
        "view_angle": cam.GetViewAngle(),
        "parallel": bool(cam.GetParallelProjection()),
        "parallel_scale": cam.GetParallelScale(),
    }


def _apply_camera(ctx, cam_dict):
    cam = ctx["ren"].GetActiveCamera()
    cam.SetPosition(*cam_dict["position"])
    cam.SetFocalPoint(*cam_dict["focal_point"])
    cam.SetViewUp(*cam_dict["view_up"])
    cam.SetViewAngle(cam_dict.get("view_angle", 30.0))
    cam.SetParallelProjection(cam_dict.get("parallel", False))
    if cam_dict.get("parallel"):
        cam.SetParallelScale(cam_dict.get("parallel_scale", 1.0))
    ctx["ren"].ResetCameraClippingRange()


def _size_is_real(size):
    return size[0] > 10 and size[1] > 10 and tuple(size) != (300, 300)


def _schedule_fit(view_id):
    """Fit the camera once the client has synced its real canvas size"""
    ctx = _views.get(view_id)
    if ctx is None or ctx["fit_task_running"]:
        return

    async def _fit():
        try:
            last = None
            stable = 0
            for _ in range(_FIT_TIMEOUT_TICKS):
                size = ctx["ren_win"].GetSize()
                if _size_is_real(size) and size == last:
                    stable += 1
                    if stable >= 3:
                        break
                else:
                    stable = 0
                last = size
                await asyncio.sleep(_FIT_POLL_S)
            cam = ctx["pending_camera"]
            ctx["pending_camera"] = None
            if cam is not None:
                _apply_camera(ctx, cam)
            else:
                ctx["ren"].ResetCamera()
                ctx["ren"].ResetCameraClippingRange()
            ctx["fitted"] = True
            ctx["ren_win"].Render()
            ctx["view_update"]()
        finally:
            ctx["fit_task_running"] = False

    ctx["fit_task_running"] = True
    asynchronous.create_task(_fit())


def option_for(view_id):
    return {
        "name": f"volume_view_{view_id}",
        "label": "Volume View",
        "icon": "mdi-rotate-3d",
        "allowed_categories": [VeraDtype.PIN.title, VeraDtype.CHANNEL.title],
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
        "mapper": mapper,
        "has_data": False,
        "fitted": False,
        "fit_task_running": False,
        "pending_camera": None,
        "view_update": lambda: None,
        "reset_camera": lambda: None,
    }
    _views[view_id] = ctx
    with DivLayout(server, template_name=f"volume_view_{view_id}") as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style=("flex: 1; min-width: 0;display: flex; flex-direction: column;")):
            with html.Div(style="flex: 1; min-height: 0; position: relative;"):
                html_view = vtk.VtkRemoteView(
                    ren_win,
                    ref=f"volume_view_{view_id}",
                    interactive_ratio=2,
                    interactive_quality=40,
                    still_quality=90,
                )
                with html.Div(
                    style=("position: absolute; top: 4px; right: 4px;display: flex; gap: 2px;")
                ):
                    with vuetify.VMenu(offset_y=True, close_on_content_click=False):
                        with vuetify.Template(v_slot_activator="{ on, attrs }"):
                            with vuetify.VBtn(icon=True, small=True, v_bind="attrs", v_on="on"):
                                vuetify.VIcon("mdi-box-cutter", small=True)
                        with vuetify.VCard(style="padding: 8px 12px; min-width: 240px;"):
                            vuetify.VSwitch(
                                v_model=f"crop_enabled_{view_id}",
                                label="Crop",
                                dense=True,
                                hide_details=True,
                            )
                            with vuetify.VBtnToggle(
                                v_model=f"crop_mode_{view_id}",
                                mandatory=True,
                                dense=True,
                                disabled=(f"!crop_enabled_{view_id}",),
                                style="margin: 8px 0;",
                            ):
                                with vuetify.VBtn(value="keep", small=True):
                                    html.Span("Keep inside")
                                with vuetify.VBtn(value="remove", small=True):
                                    html.Span("Remove")
                            vuetify.VRangeSlider(
                                v_model=f"crop_x_{view_id}",
                                min=0,
                                max=1,
                                step=0.01,
                                label="X",
                                dense=True,
                                hide_details=True,
                                disabled=(f"!crop_enabled_{view_id}",),
                            )
                            vuetify.VRangeSlider(
                                v_model=f"crop_y_{view_id}",
                                min=0,
                                max=1,
                                step=0.01,
                                label="Y",
                                dense=True,
                                hide_details=True,
                                disabled=(f"!crop_enabled_{view_id}",),
                            )
                            vuetify.VRangeSlider(
                                v_model=f"crop_z_{view_id}",
                                min=0,
                                max=1,
                                step=0.01,
                                label="Z",
                                dense=True,
                                hide_details=True,
                                disabled=(f"!crop_enabled_{view_id}",),
                            )
                    with vuetify.VBtn(icon=True, small=True, click=html_view.reset_camera):
                        vuetify.VIcon("mdi-crop-free", small=True)
            ctx["view_update"] = html_view.update
            ctx["reset_camera"] = html_view.reset_camera

        with html.Div(
            style=(
                "flex: 0 0 auto; width: 70px; padding: 4px 0;display: flex; align-self: stretch;"
            )
        ):
            vera.VerticalColorMapEditor(
                v_model=f"color_range_{view_id}_0",
                color_preset="jet",
                units=(f"color_units_{view_id}",),
            )

    return ctx


def _apply_crop(ctx, state, view_id):
    """Configure the mapper's cropping box from normalized slider state."""
    mapper = ctx["mapper"]
    if not state[f"crop_enabled_{view_id}"]:
        mapper.CroppingOff()
        return

    xmin, xmax, ymin, ymax, zmin, zmax = ctx["volume_data"].GetBounds()
    fx = [float(v) for v in state[f"crop_x_{view_id}"]]
    fy = [float(v) for v in state[f"crop_y_{view_id}"]]
    fz = [float(v) for v in state[f"crop_z_{view_id}"]]

    def span(lo, hi, frac):
        return lo + frac * (hi - lo)

    mapper.SetCroppingRegionPlanes(
        span(xmin, xmax, fx[0]),
        span(xmin, xmax, fx[1]),
        span(ymin, ymax, fy[0]),
        span(ymin, ymax, fy[1]),
        span(zmin, zmax, fz[0]),
        span(zmin, zmax, fz[1]),
    )
    if state[f"crop_mode_{view_id}"] == "remove":
        mapper.SetCroppingRegionFlags(_CROP_INVERTED_SUBVOLUME)
    else:
        mapper.SetCroppingRegionFlags(_CROP_SUBVOLUME)
    mapper.CroppingOn()


def _update_crop(server, view_id):
    state = server.state
    ctx = _views.get(view_id)
    if ctx is None or not _is_active(state, view_id):
        return
    _apply_crop(ctx, state, view_id)
    ctx["ren_win"].Render()
    ctx["view_update"]()


def _update_volume(server, registry: VeraDataRegistry, view_id):
    state = server.state
    ctx = _views.get(view_id)
    if ctx is None or not _is_active(state, view_id):
        return
    src_id = state[f"selected_src_id_{view_id}"]
    array_name = state[f"selected_array_{view_id}"]
    vera_out_file = registry.get(src_id)
    if vera_out_file is None or not array_name:
        return
    array = vera_out_file.get_dataset(array_name, state_idx=get_time(state, view_id))
    if array.dataset_type.title not in option_for(0)["allowed_categories"]:
        return

    thresholds_to_apply = get_thresholds(state, view_id)
    if thresholds_to_apply:
        array = apply_thresholds(array, thresholds_to_apply)
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

    volume_array = np.repeat(volume_array, core.get_axial_mesh_pixels(dataset=array), axis=2)
    for axis in range(3):
        if volume_array.shape[axis] < 2:
            volume_array = np.repeat(volume_array, 2, axis=axis)
    axial_mesh = core.get_axial_mesh(dataset=array)
    if axial_mesh is None:
        return
    axial_dim = np.diff(axial_mesh)
    pin_pitch = core.pin_pitch
    cm_per_axial_pixel = float(axial_dim.sum() / volume_array.shape[2])
    spacing = (pin_pitch, pin_pitch, cm_per_axial_pixel)

    scalars = volume_array.transpose(2, 1, 0).ravel()  # single contiguous copy
    vtk_array = np_s.numpy_to_vtk(scalars, deep=True)

    volume_data = ctx["volume_data"]
    volume_data.SetDimensions(*volume_array.shape)
    volume_data.SetSpacing(*spacing)
    pd = volume_data.GetPointData()
    while pd.GetNumberOfArrays() > 0:
        pd.RemoveArray(0)
    pd.SetScalars(vtk_array)
    volume_data.Modified()
    ctx["has_data"] = True

    if ctx["pending_camera"] is not None or not ctx["fitted"]:
        _schedule_fit(view_id)

    _apply_crop(ctx, state, view_id)
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
    state[f"crop_enabled_{view_id}"] = False
    state[f"crop_mode_{view_id}"] = "keep"
    state[f"crop_x_{view_id}"] = [0.0, 1.0]
    state[f"crop_y_{view_id}"] = [0.0, 1.0]
    state[f"crop_z_{view_id}"] = [0.0, 1.0]
    state[f"camera_{view_id}"] = None

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

    @ctrl.add("snapshot_volume_cameras")
    def _snapshot_camera(**kwargs):
        """Write this slot's live camera into state."""
        ctx = _views.get(view_id)
        if ctx is None:
            return
        if _is_active(state, view_id):
            state[f"camera_{view_id}"] = _camera_state(ctx["ren"].GetActiveCamera())
        else:
            state[f"camera_{view_id}"] = None

    @state.change(f"camera_{view_id}")
    def _on_camera_changed(**kwargs):
        ctx = _views.get(view_id)
        if ctx is None:
            return
        cam = state[f"camera_{view_id}"]
        if not cam:
            return
        if ctx["has_data"] and _is_active(state, view_id) and ctx["fitted"]:
            _apply_camera(ctx, cam)
            ctx["ren_win"].Render()
            ctx["view_update"]()
        else:
            ctx["pending_camera"] = cam
            if ctx["has_data"]:
                _schedule_fit(view_id)

    @state.change("dark_mode")
    def _on_theme_changed(**kwargs):
        _update_background(server, view_id)

    @state.change(f"grid_view_{view_id}", f"locked_{view_id}")
    def _on_slot_changed(**kwargs):
        if _is_active(state, view_id):
            _update_volume(server, registry, view_id)

    @state.change(f"selected_array_{view_id}", f"selected_src_id_{view_id}", "thresholds")
    def _on_selection_changed(**kwargs):
        if _is_active(state, view_id):
            _update_volume(server, registry, view_id)

    @state.change(f"color_range_{view_id}_0")
    def _on_color_changed(**kwargs):
        if _is_active(state, view_id):
            _update_color(server, view_id)

    @state.change(
        f"crop_enabled_{view_id}",
        f"crop_mode_{view_id}",
        f"crop_x_{view_id}",
        f"crop_y_{view_id}",
        f"crop_z_{view_id}",
    )
    def _on_crop_changed(**kwargs):
        _update_crop(server, view_id)

    @ctrl.add("on_vera_out_active_state_index_changed")
    def _on_state_index_changed(**kwargs):
        if _is_active(state, view_id):
            _update_volume(server, registry, view_id)
