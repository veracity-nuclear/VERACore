import copy

import numpy as np

from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
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
from trame.widgets import vtk, vuetify

from vera_core.app.core.vera_data import VeraDataRegistry, VeraDataSource


# Single shared template; all volume cards render the same VTK view.
SHARED_TEMPLATE_NAME = "volume_view"

_initialized = False
_reset_camera_count = 0


def option_for(view_id):
    # Same name for every card -> ServerTemplate resolves to the one shared template.
    return {
        "name": SHARED_TEMPLATE_NAME,
        "label": "Volume View",
        "icon": "mdi-rotate-3d",
    }


def initialize(server, registry : VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    # Register the option in this card's menu (always).
    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    # Build VTK pipeline and DivLayout only once.
    global _initialized
    if _initialized:
        return
    _initialized = True

    ren = vtkRenderer()
    ren_win = vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.OffScreenRenderingOn()
    ren.SetBackground(1, 1, 1)

    iren = vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    iren.SetRenderWindow(ren_win)

    axes = vtkAxesActor()
    orientation_marker = vtkOrientationMarkerWidget()
    orientation_marker.SetOrientationMarker(axes)
    orientation_marker.SetInteractor(iren)
    # orientation_marker.EnabledOn()
    # orientation_marker.InteractiveOn()

    opacity_points = [
        (-10.0, 1),
        (-0.00000000000000000001, 1),
        (-0.000000000000000000005, 0.0),
        (0.000000000000000000005, 0.0),
        (0.00000000000000000001, 1),
        (10.0, 1),
    ]
    opacity_transfer_function = vtkPiecewiseFunction()
    for point in opacity_points:
        opacity_transfer_function.AddPoint(*point)

    original_color_points = [
        (0.000000, 0.0, 0.0, 0.5625),
        (0.216992, 0.0, 0.0, 1.0000),
        (0.712975, 0.0, 1.0, 1.0000),
        (0.960965, 0.5, 1.0, 0.5000),
        (1.208960, 1.0, 1.0, 0.0000),
        (1.704940, 1.0, 0.0, 0.0000),
        (1.952930, 0.5, 0.0, 0.0000),
    ]
    color_transfer_function = vtkColorTransferFunction()
    for point in original_color_points:
        color_transfer_function.AddRGBPoint(*point)

    volume_property = vtkVolumeProperty()
    volume_property.SetColor(color_transfer_function)
    volume_property.SetScalarOpacity(opacity_transfer_function)

    volume_data = vtkImageData()
    volume_mapper = vtkSmartVolumeMapper()
    volume_mapper.SetInputData(volume_data)

    volume = vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    ren.AddVolume(volume)

    ren.GetActiveCamera().Pitch(90)
    ren.GetActiveCamera().OrthogonalizeViewUp()

    @state.change("color_range")
    def update_color_points(color_range, **kwargs):
        original_range = (original_color_points[0][0], original_color_points[-1][0])
        new_color_points = copy.deepcopy(original_color_points)
        for i, row in enumerate(new_color_points):
            new_value = np.interp(row[0], original_range, color_range)
            new_color_points[i] = (new_value, *row[1:])

        color_transfer_function.RemoveAllPoints()
        for point in new_color_points:
            color_transfer_function.AddRGBPoint(*point)

        ren_win.Render()
        ctrl.view_update()

    @state.change("selected_array")
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_volume_view(selected_array, **kwargs):
        global _reset_camera_count
        vera_out_file = registry.default_source
        array = vera_out_file.array(selected_array)

        assembly_shape = array.shape[:2]
        reduced_core_map = vera_out_file.core.reduced_core_map
        reduced_map_shape = reduced_core_map.shape
        expanded_core_shape = (
            reduced_map_shape[0] * assembly_shape[0],
            reduced_map_shape[1] * assembly_shape[1],
        )
        volume_shape = (*expanded_core_shape, array.shape[2])
        volume_array = np.zeros(volume_shape, dtype=array.dtype)

        for assembly_id in range(array.shape[3]):
            target = assembly_id + 1
            rows, cols = np.where(reduced_core_map == target)
            if len(rows) == 0:
                continue
            if len(rows) > 1:
                raise ValueError(
                    f"Assembly {target} appears multiple times in reduced_core_map. "
                    f"Expected exactly one match, found {len(rows)}."
                )
            core_row = int(rows[0])
            core_col = int(cols[0])
            row_range = (
                core_row * assembly_shape[0],
                (core_row + 1) * assembly_shape[0],
            )
            col_range = (
                core_col * assembly_shape[1],
                (core_col + 1) * assembly_shape[1],
            )
            volume_array[slice(*row_range), slice(*col_range)] = array[:, :, :, assembly_id]

        axial_mesh_pixels = vera_out_file.core.axial_mesh_pixels
        volume_array = np.repeat(volume_array, axial_mesh_pixels, axis=2)

        raveled = volume_array.copy().transpose(2, 1, 0).ravel()
        vtk_array = np_s.numpy_to_vtk(raveled, deep=True)

        volume_data.SetDimensions(*volume_array.shape)
        pd = volume_data.GetPointData()
        while pd.GetNumberOfArrays() > 0:
            pd.RemoveArray(0)
        pd.SetScalars(vtk_array)
        volume_data.Modified()

        if _reset_camera_count < 2:
            ren.ResetCameraClippingRange()
            ren.ResetCamera()
            _reset_camera_count += 1
            ctrl.reset_camera()

        ctrl.view_update()

        # Keep orientation_marker reference alive.
        om = orientation_marker  # noqa

    with DivLayout(server, template_name=SHARED_TEMPLATE_NAME) as layout:
        layout.root.style = "height: 100%;"
        html_view = vtk.VtkRemoteView(ren_win, ref="volume_view")
        ctrl.reset_camera = html_view.reset_camera
        ctrl.view_update = html_view.update

        with vuetify.VBtn(
            icon=True,
            click=html_view.reset_camera,
            absolute=True,
            style="top: 0; right: 2px",
            small=True,
        ):
            vuetify.VIcon("mdi-crop-free", small=True)