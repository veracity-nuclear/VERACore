from vera_core.data.dtypes import Surface, VeraDim, VeraDtype
from vera_core.data.model import VeraOutCore

from ..helpers import convert_ji_to_node


def ui_selection(
    j: int, i: int, layer: int, assembly: int, surface: int, group: int | None = None
) -> dict[VeraDim, int]:
    """Selected pin (j, i), layer, assembly and surface as semantic indices.

    The node is derived from (j, i), so the same selection serves pin and
    nodal data.
    """
    sel = {
        VeraDim.PIN_Y: j,
        VeraDim.PIN_X: i,
        VeraDim.NODE: int(convert_ji_to_node(j, i)),
        VeraDim.AXIAL: layer,
        VeraDim.ASSEMBLY: assembly,
        VeraDim.SURFACE: surface,
    }
    if group is not None and group >= 1:
        sel |= {VeraDim.GROUP: group - 1}
    return sel


def point_label(core: VeraOutCore, dtype: VeraDtype, selection: dict[VeraDim, int]) -> str:
    """Where the point sits, naming only the dims the dtype has.

    Example: `` | C-3 @(2,1) | z = 37.5``
    """
    dims = dtype.dim_axes
    where = []
    if VeraDim.GROUP in dims and VeraDim.GROUP in selection:
        where.append(f"Group {selection[VeraDim.GROUP] + 1}")
    if VeraDim.ASSEMBLY in dims:
        assembly = selection[VeraDim.ASSEMBLY]
        where.append(core.reduced_core_map_label(assembly, dtype.is_computational()))
    if VeraDim.PIN_Y in dims:
        where.append(f"@({selection[VeraDim.PIN_X] + 1},{selection[VeraDim.PIN_Y] + 1})")
    if VeraDim.NODE in dims:
        where.append(f"@(NODE {selection[VeraDim.NODE] + 1})")
    if VeraDim.SURFACE in dims:
        where.append(Surface(selection[VeraDim.SURFACE]).str)
    parts = [" ".join(where)] if where else []
    if VeraDim.AXIAL in dims:
        z = core.get_axial_mesh_means(dataset_type=dtype)[selection[VeraDim.AXIAL]]
        parts.append(f"z = {z}")
    return "".join(f" | {part}" for part in parts)
