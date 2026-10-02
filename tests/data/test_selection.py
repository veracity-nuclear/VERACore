import pytest

from vera_core.app.ui.views.selection import point_label, ui_selection
from vera_core.data.dtypes import VeraDim, VeraDtype
from .slice_support import CORE

T = VeraDtype
SEL = ui_selection(1, 2, 3, 4, 2)  # j, i, layer, assembly, surface


def test_ui_selection_derives_node_from_pin():
    assert SEL[VeraDim.NODE] == 3  # clip(i + 2j, 0, 3)
    assert (SEL[VeraDim.PIN_Y], SEL[VeraDim.PIN_X]) == (1, 2)


@pytest.mark.parametrize(
    "dtype, label",
    [
        (T.PIN, " | B-4 @(3,2) | z = 70.0"),
        (T.RADIAL, " | B-4 @(3,2)"),
        (T.NODAL_ENERGY, " | B-4 @(NODE 4) | z = 70.0"),
        (T.NODAL_SURFACE, " | B-4 @(NODE 4) EAST | z = 70.0"),
        (T.ASSY_SURFACE, " | B-4 EAST | z = 70.0"),
        (T.ASSY_ENERGY, " | B-4 | z = 70.0"),
        (T.POINT_DETECTOR, " | B-4 | z = 70.0"),
        (T.AXIAL, " | z = 70.0"),
        (T.RADIAL_ASSEMBLY, " | B-4"),
        (T.SCALAR, ""),
    ],
    ids=str,
)
def test_point_label_names_only_present_dims(dtype, label):
    assert point_label(CORE, dtype, SEL) == label
