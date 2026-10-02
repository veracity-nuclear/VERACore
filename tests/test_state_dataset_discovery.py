import tempfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import h5py
import numpy as np

_VERA_DATA_PATH = Path(__file__).resolve().parents[1] / "vera_core" / "app" / "core" / "vera_data.py"
_VERA_DATA_SPEC = spec_from_file_location("vera_data", _VERA_DATA_PATH)
_VERA_DATA = module_from_spec(_VERA_DATA_SPEC)
_VERA_DATA_SPEC.loader.exec_module(_VERA_DATA)

VeraDtype = _VERA_DATA.VeraDtype
VeraOutCore = _VERA_DATA.VeraOutCore
VeraOutState = _VERA_DATA.VeraOutState


def test_discovers_postprocessed_assembly_datasets_by_name_fallback():
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        with h5py.File(tmp.name, "w") as f:
            core = f.create_group("CORE")
            core.create_dataset("axial_mesh", data=np.array([0.0, 1.0, 2.0, 3.0]))
            core.create_dataset("core_map", data=np.array([[1.0, 2.0], [np.nan, np.nan]]))
            core.create_dataset("core_sym", data=np.array(1))

            state = f.create_group("STATE_0001")
            state.create_dataset("assembly_powers", data=np.ones((3, 2)))
            state.create_dataset("assembly_exposures", data=np.ones((2,)))
            state.create_dataset("assembly_wrong_nass_2d", data=np.ones((3, 3)))
            state.create_dataset("assembly_wrong_nass_1d", data=np.ones((3,)))
            state.create_dataset("something_else", data=np.ones((3, 2)))

        with h5py.File(tmp.name, "r") as f:
            vc = VeraOutCore(f)
            vc.core_dtypes = lambda *args, **kwargs: VeraDtype.UNKNOWN
            vs = VeraOutState(f, 1, vc)

            assert "assembly_powers" in vs.categorized_ds_names[VeraDtype.ASSEMBLY]
            assert "assembly_exposures" in vs.categorized_ds_names[VeraDtype.RADIAL_ASSEMBLY]
            assert "assembly_wrong_nass_2d" not in vs.full_core_keys
            assert "assembly_wrong_nass_1d" not in vs.full_core_keys
            assert "something_else" not in vs.full_core_keys
