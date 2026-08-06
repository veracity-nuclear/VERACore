"""In-memory data sources for tests and synthetic cases.

Builds the same objects the h5 reader does, from plain numpy arrays, so the
whole data layer can be exercised without an HDF5 file. `build_source` is the
main entry point; the `spec` helpers describe the cases worth covering.
"""

import numpy as np

from ..dtypes import VeraDataset, VeraDtype
from ..model import DatasetSource, VeraDataSource, VeraOutCore, VeraOutState


class DictDatasetSource(DatasetSource):
    """Names to in-memory arrays.

    Mirrors H5DatasetSource: scalars report shape (1,), unknown names return
    None, and loaded arrays are typed by shape when a dtype map is supplied.
    """

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        provenance: str = "<memory>",
        dataset_dtypes: dict[tuple, VeraDtype] | None = None,
        units: dict[str, str] | None = None,
    ):
        self._a = {k: np.asarray(v) for k, v in arrays.items()}
        self._provenance = provenance
        self._dataset_dtypes = dataset_dtypes or {}
        self._units = units or {}
        self.load_count: dict[str, int] = {}

    @property
    def provenance(self) -> str:
        return self._provenance

    def names(self) -> list[str]:
        return list(self._a)

    def shape(self, name: str) -> tuple[int, ...] | None:
        if name not in self._a:
            return None
        s = self._a[name].shape
        return (1,) if s == () else tuple(s)

    def load(self, name: str) -> VeraDataset | None:
        if name not in self._a:
            return None
        self.load_count[name] = self.load_count.get(name, 0) + 1
        raw = self._a[name]
        arr = raw if raw.ndim else np.array([raw])
        dtype = self._dataset_dtypes.get(tuple(np.shape(raw)), VeraDtype.UNKNOWN)
        return VeraDataset(arr, dtype, self._units.get(name, "unitless"))


def core_map(nass_side: int = 5, core_sym: int = 4) -> np.ndarray:
    """A square assembly map.

    Full core (sym 1) numbers every position uniquely. Quarter core (sym 4)
    mirrors ids about the centre, so distinct ids are fewer than cells --
    which is how VeraOutCore infers symmetry when core_sym is absent.
    """
    n = nass_side
    if core_sym == 1:
        return np.arange(1, n * n + 1, dtype=np.int64).reshape(n, n)
    half = n // 2
    quadrant = np.arange(1, (half + 1) ** 2 + 1, dtype=np.int64).reshape(half + 1, half + 1)
    full = np.zeros((n, n), dtype=np.int64)
    full[half:, half:] = quadrant
    full[:half, half:] = quadrant[:0:-1, :]
    full[:, :half] = full[:, :half:-1]
    return full


def core_arrays(
    *,
    nass_side: int = 5,
    core_sym: int | None = 4,
    map_sym: int | None = None,
    nax: int = 4,
    npin: int | None = 3,
    with_axial_mesh: bool = True,
    with_pin_volumes: bool = True,
    with_comp_core: bool = False,
    with_detectors: bool = False,
    aspect_ratio: float | None = None,
    apitch: float | None = None,
    labels: bool = False,
) -> dict[str, np.ndarray]:
    """Arrays for a synthetic CORE group. Every option maps to a real branch."""
    # map_sym controls the map's layout; core_sym controls whether the file
    # declares it. Differing lets tests exercise symmetry inference.
    cm = core_map(nass_side, map_sym if map_sym is not None else (core_sym or 1))
    nass = int(np.count_nonzero(np.unique(cm)))
    out: dict[str, np.ndarray] = {"core_map": cm}

    if core_sym is not None:
        out["core_sym"] = np.array([core_sym])
    if with_axial_mesh:
        out["axial_mesh"] = np.linspace(0.0, 100.0, nax + 1)
    if npin is not None:
        out["npin"] = np.array([npin])
        if with_pin_volumes:
            pv = np.ones((npin, npin, nax, nass))
            pv[0, 0] = 0.0  # one non-fuel pin position
            out["pin_volumes"] = pv
    if aspect_ratio is not None:
        out["aspect_ratio"] = np.array([aspect_ratio])
    if apitch is not None:
        out["apitch"] = np.array([apitch])
    if with_comp_core:
        out["computational_core_map"] = cm.astype(float)
        out["computational_axial_mesh"] = np.linspace(0.0, 100.0, nax + 1)
    if with_detectors:
        out["detector_map"] = cm.copy()
        out["detector_axial_mesh"] = np.linspace(5.0, 95.0, nax)
    if labels:
        out["xlabel"] = np.array([c.encode() for c in "ABCDEFGHIJ"[:nass_side]])
        out["ylabel"] = np.array([str(i).encode() for i in range(1, nass_side + 1)])
    return out


def build_core(**kwargs) -> VeraOutCore:
    """A VeraOutCore over synthetic arrays. kwargs go to core_arrays()."""
    overrides = kwargs.pop("overrides", None)
    return VeraOutCore(DictDatasetSource(core_arrays(**kwargs)), overrides=overrides)


def state_arrays(core: VeraOutCore, exposure: float, scale: float = 1.0) -> dict:
    """Datasets for one state point, shaped to match `core`."""
    npy, npx, nax, nass = core.core_shape
    rng = np.random.default_rng(int(exposure * 100))
    return {
        "pin_powers": rng.random((npy, npx, nax, nass)) * scale,
        "pin_radial_powers": rng.random((npy, npx, nass)) * scale,
        "assembly_powers": rng.random((1, nax, nass)) * scale,
        "axial_powers": rng.random((nax,)) * scale,
        "keff": np.array([1.0 + exposure / 1000.0]),
        "exposure": np.array([exposure]),
    }


def build_source(
    n_states: int = 3,
    *,
    exposures: list[float] | None = None,
    core_kwargs: dict | None = None,
    state_caching: bool = True,
) -> VeraDataSource:
    """A complete in-memory VeraDataSource, as open_vera_file_data_source builds."""
    core = build_core(**(core_kwargs or {}))
    exposures = exposures if exposures is not None else [i * 10.0 for i in range(n_states)]
    states = [
        VeraOutState(
            DictDatasetSource(
                state_arrays(core, exp),
                provenance="<memory>",
                dataset_dtypes=core.shape_to_dtype,
                units={"pin_powers": "W"},
            ),
            idx,
            core,
        )
        for idx, exp in enumerate(exposures)
    ]
    return VeraDataSource(
        core=core,
        states=states,
        provenance="<memory>",
        filename=None,
        close_callback=None,
        state_caching=state_caching,
    )
