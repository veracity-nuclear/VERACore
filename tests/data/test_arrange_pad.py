import numpy as np
import pytest

from vera_core.data.dtypes import VeraDataset, VeraDim, VeraDtype

T = VeraDtype
SF, GR, ND, AX, AS = VeraDim.SURFACE, VeraDim.GROUP, VeraDim.NODE, VeraDim.AXIAL, VeraDim.ASSEMBLY


def data(dtype, shape):
    return VeraDataset(np.arange(np.prod(shape), dtype=float).reshape(shape), dtype, "d", "W")


def test_pad_inserts_absent_dim_at_its_order_position():
    ds = data(T.ASSY_SURFACE, (6, 2, 1, 5, 9))
    [out] = ds.arrange(order=(SF, ND, AS), split=(), pad=(ND,), axial=3, group=1)
    np.testing.assert_array_equal(out, np.asarray(ds)[:, 1, :, 3, :])
    assert out.shape == (6, 1, 9)


def test_pad_is_noop_when_dim_present():
    ds = data(T.NODAL_SURFACE, (6, 2, 4, 5, 9))
    a = ds.arrange(order=(SF, ND, AS), split=(GR,), axial=0)
    b = ds.arrange(order=(SF, ND, AS), split=(GR,), pad=(ND,), axial=0)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_pad_after_selection_keeps_length_one_axis():
    ds = data(T.NODAL, (4, 5, 9))
    [out] = ds.arrange(order=(ND, AS), pad=(ND,), node=2, axial=1)
    np.testing.assert_array_equal(out, np.asarray(ds)[2:3, 1, :])


def test_pad_with_split_pads_every_part():
    ds = data(T.ASSY_ENERGY, (3, 1, 5, 9))
    parts = ds.arrange(order=(ND, AX, AS), split=(GR,), pad=(ND,))
    assert [p.shape for p in parts] == [(1, 5, 9)] * 3
    np.testing.assert_array_equal(parts[2][0], np.asarray(ds)[2, 0])


def test_padded_output_keeps_metadata():
    [out] = data(T.ASSEMBLY, (1, 5, 9)).arrange(order=(ND, AX, AS), pad=(ND,))
    assert isinstance(out, VeraDataset)
    assert (out.dataset_type, out.name, out.physical_units) == (T.ASSEMBLY, "d", "W")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"order": (AX, AS), "pad": (ND,)}, "must also appear in order: node"),
        ({"order": (ND, AX, AS), "pad": (ND, ND)}, "Duplicate dimensions in pad"),
    ],
)
def test_pad_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        data(T.ASSEMBLY, (1, 5, 9)).arrange(**kwargs)
