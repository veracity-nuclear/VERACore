import numpy as np
import pytest

from vera_core.data import model
from vera_core.data.dtypes import DerivationMethod, VeraAxes, VeraDataset, VeraDtype
from vera_core.data.model import VeraDataSource, VeraOutCore, VeraOutState
from vera_core.data.readers.mock import DictDatasetSource, build_core, build_source

from .conftest import FakeCalculator, make_source


def loads(source, state_idx, name):
    return source.states[state_idx].source.load_count.get(name, 0)


# ---------------------------------------------------------------- construction


@pytest.mark.parametrize("requested, expected", [(-5, 0), (1, 1), (99, 2)])
def test_initial_active_state_is_clamped(requested, expected):
    core = build_core()
    states = build_source(3, core_kwargs={}).states
    src = VeraDataSource(core=core, states=states, provenance="x", active_state_idx=requested)
    assert src.active_state_index == expected


def test_name_defaults_to_empty_string(source):
    assert source.name == ""


def test_provenance_and_core(source):
    assert source.provenance == "<memory>"
    assert isinstance(source.core, VeraOutCore)


def test_failed_calculator_load_leaves_none(monkeypatch):
    def fail(filename):
        raise OSError("nope")

    monkeypatch.setattr(model, "VERAout", fail)
    src = build_source()
    src2 = VeraDataSource(core=src.core, states=src.states, provenance="p", filename="f.h5")
    assert src2.vera_calculator is None


def test_calculator_loaded_when_filename_given(monkeypatch):
    monkeypatch.setattr(model, "VERAout", lambda filename: FakeCalculator())
    src = build_source()
    src2 = VeraDataSource(core=src.core, states=src.states, provenance="p", filename="f.h5")
    assert isinstance(src2.vera_calculator, FakeCalculator)


def test_empty_source_without_caching_constructs():
    src = VeraDataSource(core=build_core(), states=[], provenance="x", state_caching=False)
    assert src.time_axes() == {"state_count": []}


def test_empty_source_with_caching_constructs():
    VeraDataSource(core=build_core(), states=[], provenance="x", state_caching=True)


# ---------------------------------------------------------------- active state and caching


def test_active_state_is_cached_on_construction(source):
    source.active_state.get("keff")
    assert loads(source, 0, "keff") == 1


def test_inactive_states_read_lazily(source):
    source.states[1].get("keff")
    source.states[1].get("keff")
    assert loads(source, 1, "keff") == 2


def test_switching_moves_the_cache(source):
    source.active_state_index = 1
    source.states[0].get("keff")
    source.states[1].get("keff")
    assert loads(source, 0, "keff") == 2  # cached read, then a fresh lazy read
    assert loads(source, 1, "keff") == 1


def test_setting_same_index_does_not_reload(source):
    source.active_state_index = 0
    assert loads(source, 0, "keff") == 1


def test_setter_clamps(source):
    source.active_state_index = 99
    assert source.active_state_index == 2
    source.active_state_index = -1
    assert source.active_state_index == 0


def test_caching_disabled():
    src = build_source(state_caching=False)
    src.active_state_index = 1
    assert loads(src, 0, "keff") == 0
    assert loads(src, 1, "keff") == 0


# ---------------------------------------------------------------- time axes


def test_time_axes_from_first_state(source):
    assert source.time_axes() == {"state_count": [0, 1, 2], "exposure": [0.0, 10.0, 20.0]}


def test_add_state_extends_every_axis(source):
    extra = build_source(exposures=[99.0]).states[0]
    source.add_state(extra)
    assert source.time_axes()["state_count"] == [0, 1, 2, 3]
    assert source.time_axes()["exposure"][-1] == 99.0


def test_replace_state_updates_values(source):
    source.replace_state(1, build_source(exposures=[55.0]).states[0])
    assert source.time_axes()["exposure"] == [0.0, 55.0, 20.0]


def test_remove_state_updates_axes_and_returns_state(source):
    target = source.states[1]
    assert source.remove_state(1) is target
    assert source.time_axes() == {"state_count": [0, 1], "exposure": [0.0, 20.0]}


def test_removing_last_state_clears_axes():
    src = build_source(1)
    src.remove_state(0)
    assert src.time_axes() == {"state_count": []}


def test_add_state_to_emptied_source_rebuilds_axes():
    src = build_source(1, state_caching=False)
    state = src.remove_state(0)
    src.add_state(state)
    assert src.time_axes() == {"state_count": [0], "exposure": [0.0]}


def test_removing_active_last_state_keeps_active_state_valid(source):
    source.active_state_index = 2
    source.remove_state(2)
    assert source.active_state is source.states[source.active_state_index]


# ---------------------------------------------------------------- close


def test_close_runs_callback_and_closes_calculator():
    calls = []
    src = build_source()
    src2 = VeraDataSource(
        core=src.core, states=src.states, provenance="p", close_callback=lambda: calls.append(1)
    )
    src2.vera_calculator = FakeCalculator()
    src2.close()
    assert calls == [1]
    assert src2.vera_calculator.closed


def test_close_without_handles_is_noop(source):
    source.vera_calculator = None
    source.close()


# ---------------------------------------------------------------- dataset access


def test_default_datasets(source):
    assert source.default_datasets() == {
        "PIN": "pin_powers",
        "ASSEMBLY": "assembly_powers",
        "AXIAL": "axial_powers",
        "RADIAL": "pin_radial_powers",
        "SCALAR": "exposure",
    }


def test_get_dataset_returns_owned_copy(source):
    first = source.get_dataset("axial_powers")
    first[:] = -1
    assert (source.get_dataset("axial_powers") != -1).all()


def test_get_dataset_masks_reflected_on_odd_quarter_core(source):
    assert np.isnan(source.get_dataset("pin_powers")).any()
    assert not np.isnan(source.get_dataset("pin_powers", mask_reflected=False)).any()


def test_get_dataset_does_not_mask_even_quarter_core():
    q = np.array([[1, 2], [3, 4]])
    cm = np.block([[q[::-1, ::-1], q[::-1, :]], [q[:, ::-1], q]])
    core = VeraOutCore(
        DictDatasetSource(
            {
                "core_map": cm,
                "core_sym": np.array([4]),
                "axial_mesh": np.linspace(0.0, 100.0, 5),
                "npin": np.array([3]),
            }
        )
    )
    state = VeraOutState(
        DictDatasetSource({"pin_powers": np.ones((3, 3, 4, 4))}, dataset_dtypes=core.shape_to_dtype),
        0,
        core,
    )
    src = VeraDataSource(core=core, states=[state], provenance="x")
    assert core.is_even()
    assert not np.isnan(src.get_dataset("pin_powers")).any()


def test_get_dataset_by_state_index(source):
    np.testing.assert_array_equal(source.get_dataset("exposure", state_idx=2), [20.0])


def test_pin_volumes_come_from_core_regardless_of_state(source):
    vols = source.get_dataset("pin_volumes", mask_reflected=False, state_idx=2)
    np.testing.assert_array_equal(vols, source.core.pin_volumes)


def test_get_dataset_missing_raises(source):
    with pytest.raises(RuntimeError, match="nope"):
        source.get_dataset("nope")


def test_metadata_lookups(source):
    assert source.get_dataset_dtype("pin_powers") == VeraDtype.PIN
    assert source.get_dataset_units("pin_powers") == "W"
    assert source.get_dataset_shape("pin_powers") == (3, 3, 4, 9)


def test_metadata_lookups_for_missing_name(source):
    assert source.get_dataset_dtype("nope") == VeraDtype.UNKNOWN
    assert source.get_dataset_units("nope") == "Unitless"
    # The docstring says an empty tuple; the annotation and code say None.
    assert source.get_dataset_shape("nope") is None

# ---------------------------------------------------------------- diffs


def test_diff_same_mesh_is_plain_subtraction(source):
    comp = make_source()
    source.add_new_diff_dataset("pin_powers", comp, "pin_powers", "d", ref_scale=2.0, units="W")
    for ref_state, comp_state in zip(source.states, comp.states):
        expected = ref_state.get("pin_powers") * 2.0 - comp_state.get("pin_powers")
        diff = ref_state.get("d")
        np.testing.assert_allclose(diff, expected)
        assert (diff.name, diff.physical_units, diff.dataset_type) == ("d", "W", VeraDtype.PIN)


def test_diff_is_categorized_on_each_state(source):
    source.add_new_diff_dataset("keff", make_source(), "keff", "dk")
    assert all("dk" in s.categorized_ds_names[VeraDtype.SCALAR] for s in source.states)


def test_diff_interpolates_across_axial_meshes(source):
    comp = make_source(core_kwargs={"nax": 8})
    source.add_new_diff_dataset("pin_powers", comp, "pin_powers", "d", interpolation_order=1)
    ref_means = source.core.axial_mesh_means
    comp_means = comp.core.axial_mesh_means
    ref = source.states[0].get("pin_powers")[1, 1, :, 4]
    comp_col = comp.states[0].get("pin_powers")[1, 1, :, 4]
    expected = ref - np.interp(ref_means, comp_means, comp_col)
    np.testing.assert_allclose(source.states[0].get("d")[1, 1, :, 4], expected)


def test_diff_only_on_overlapping_states(source):
    source.add_new_diff_dataset("keff", make_source(2), "keff", "dk")
    assert ["dk" in s for s in source.states] == [True, True, False]


def test_diff_against_self_same_name_rejected(source):
    with pytest.raises(ValueError, match="itself"):
        source.add_new_diff_dataset("keff", source, "keff", "dk")


def test_diff_against_self_different_names_allowed(source):
    source.add_new_diff_dataset("exposure", source, "keff", "d")
    state = source.states[1]
    np.testing.assert_allclose(state.get("d"), state.get("exposure") - state.get("keff"))


def test_diff_duplicate_name_rejected(source):
    with pytest.raises(ValueError, match="already exists"):
        source.add_new_diff_dataset("keff", make_source(), "keff", "exposure")


def test_diff_shape_mismatch_outside_axial_rejected(source):
    comp = make_source(core_kwargs={"npin": 4})
    with pytest.raises(ValueError, match="outside the axial"):
        source.add_new_diff_dataset("pin_powers", comp, "pin_powers", "d")


def test_diff_with_no_compatible_states_rejected(source):
    with pytest.raises(ValueError, match="No overlapping"):
        source.add_new_diff_dataset("keff", make_source(), "axial_powers", "d")


def test_failed_diff_adds_nothing(source):
    with pytest.raises(ValueError):
        source.add_new_diff_dataset("keff", make_source(), "missing", "d")
    assert not any("d" in s for s in source.states)


# ---------------------------------------------------------------- derivations


def derive(src, axes, method=DerivationMethod.AVERAGE, name="der"):
    src.add_new_derived_dataset("pin_powers", name, method, axes)
    return [s.get(name) for s in src.states]


def test_derivation_requires_calculator():
    src = make_source(calculator=False)
    with pytest.raises(RuntimeError):
        derive(src, VeraAxes.CORE)


@pytest.mark.parametrize(
    "axes, dtype, shape",
    [
        (VeraAxes.CORE, VeraDtype.SCALAR, (1,)),
        (VeraAxes.AXIAL, VeraDtype.AXIAL, (4,)),
        (VeraAxes.ASSEMBLY, VeraDtype.ASSEMBLY, (1, 4, 9)),
        (VeraAxes.RADIAL, VeraDtype.RADIAL, (3, 3, 9)),
        (VeraAxes.RADIAL_ASSEMBLY, VeraDtype.RADIAL_ASSEMBLY, (9,)),
        (VeraAxes.NODE, VeraDtype.NODAL, (4, 4, 9)),
    ],
)
def test_average_over_each_axes(source, axes, dtype, shape):
    for ds in derive(source, axes):
        assert ds.dataset_type == dtype
        assert ds.shape == shape
        assert ds.name == "der"


def test_average_core_value(source):
    [first, *_] = derive(source, VeraAxes.CORE)
    assert first[0] == pytest.approx(source.states[0].get("pin_powers").mean())


def test_rms_core_value(source):
    [first, *_] = derive(source, VeraAxes.CORE, DerivationMethod.RMS)
    data = source.states[0].get("pin_powers")
    assert first[0] == pytest.approx(np.sqrt((data**2).mean()))


def test_stddev_core_value(source):
    [first, *_] = derive(source, VeraAxes.CORE, DerivationMethod.STDDEV)
    assert first[0] == pytest.approx(source.states[0].get("pin_powers").std())


def test_derived_dataset_is_categorized(source):
    derive(source, VeraAxes.AXIAL)
    assert source.get_dataset_dtype("der") == VeraDtype.AXIAL


def test_unimplemented_axes_rejected(source):
    with pytest.raises(ValueError, match="not implemented"):
        derive(source, VeraAxes.RADIAL_NODE)


def test_derived_duplicate_name_rejected(source):
    with pytest.raises(ValueError, match="already exists"):
        derive(source, VeraAxes.CORE, name="keff")


def test_derivation_of_missing_array_adds_nothing(source):
    source.add_new_derived_dataset("missing", "der", DerivationMethod.AVERAGE, VeraAxes.CORE)
    assert not any("der" in s for s in source.states)


def test_failed_derivation_adds_nothing(source):
    source.states[2].add_derived_dataset("der", VeraDataset([0.0], VeraDtype.SCALAR))
    with pytest.raises(ValueError):
        derive(source, VeraAxes.CORE)
    assert "der" not in source.states[0]
