from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from scipy.interpolate import make_interp_spline

from .dtypes import (
    NUM_NODES,
    CoreOverride,
    DerivationMethod,
    VeraAxes,
    VeraDataset,
    VeraDtype,
    build_core_dtypes,
)
from .vera_tools.VERAout import VERAout


def nan_out_reflected(cm: np.ndarray, core_sym: int, array: VeraDataset):
    """Nans out reflected region if dataset has quarter core symmetry"""
    ax, ay = cm.shape
    dtype = array.dataset_type
    has_reflected_pins = dtype in (VeraDtype.PIN, VeraDtype.CHANNEL, VeraDtype.RADIAL)
    if has_reflected_pins and core_sym == 4:
        hpy = array.shape[0] // 2
        hpx = array.shape[1] // 2
        match array.dataset_type:
            case VeraDtype.PIN | VeraDtype.CHANNEL:
                array[:hpy, :, :, :ax] = np.nan
                array[:, :hpx, :, cm[:, 0] - 1] = np.nan
            case VeraDtype.RADIAL:
                array[:hpy, :, :ax] = np.nan
                array[:, :hpx, cm[:, 0] - 1] = np.nan
    elif (dtype == VeraDtype.COMP_NODAL or dtype == VeraDtype.NODAL) and core_sym == 4:
        array[: int(NUM_NODES / 2), :, :ax] = np.nan
        array[0, :, cm[:, 0] - 1] = np.nan
        array[2, :, cm[:, 0] - 1] = np.nan
    elif dtype == VeraDtype.COMP_NODAL_ENERGY and core_sym == 4:
        array[:, : int(NUM_NODES / 2), :, :ax] = np.nan
        array[:, 0, :, cm[:, 0] - 1] = np.nan
        array[:, 2, :, cm[:, 0] - 1] = np.nan
    elif dtype == VeraDtype.COMP_NODAL_SURFACE and core_sym == 4:
        array[:, :, : int(NUM_NODES / 2), :, :ax] = np.nan
        array[:, :, 0, :, cm[:, 0] - 1] = np.nan
        array[:, :, 2, :, cm[:, 0] - 1] = np.nan
    return array


class DatasetSource(ABC):
    """Where one group's datasets come from. One subclass per backend."""

    @property
    @abstractmethod
    def provenance(self) -> str:
        """Where this dataset came from"""

    @abstractmethod
    def names(self) -> list[str]:
        """Dataset names this source can supply. Drives the store's manifest."""

    @abstractmethod
    def shape(self, name: str) -> tuple[int, ...] | None:
        """Shape without reading. None if absent."""

    @abstractmethod
    def load(self, name: str) -> VeraDataset | None:
        """Read and wrap. None if absent."""

    # non-abstract: subclasses inherit unless they can do better
    def has(self, name: str) -> bool:
        return self.shape(name) is not None


class DatasetStore:
    """Lazily exposes a source's datasets, caching reads on request.

    Reads go through the source on every `get` unless the name has been cached
    via `cache_dataset`/`cache_all` or pinned. `names()` is snapshotted at
    construction and drives caching; lookups also fall through to the source, so
    a name outside the manifest (a cross-group path, say) still resolves.
    Pinned datasets are in-memory only and never evicted.
    """

    def __init__(self, source: DatasetSource):
        self._source = source
        self._names = frozenset(source.names())
        self._cache: dict[str, VeraDataset] = {}
        self._pinned: dict[str, VeraDataset] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._pinned or key in self._names or self._source.has(key)

    def __getitem__(self, key: str) -> VeraDataset:
        ds = self.get(key)
        if ds is None:
            raise KeyError(f"{key} not found in source")
        return ds

    @property
    def provenance(self) -> str:
        """Where this dataset came from"""
        return self._source.provenance

    @property
    def source(self) -> DatasetSource:
        """The backend this store reads from."""
        return self._source

    def _make_dataset(self, name: str) -> VeraDataset | None:
        return self._source.load(name)

    def get(self, key: str, fallback: VeraDataset | None = None) -> VeraDataset | None:
        if key in self._pinned:
            return self._pinned[key]
        if key in self._cache:
            return self._cache[key]
        if self._source.has(key):
            ds = self._make_dataset(key)
            return ds if ds is not None else fallback
        return fallback

    def shape(self, name: str) -> tuple[int, ...] | None:
        """Shape without reading. None if absent."""
        return self._source.shape(name)

    def pin(self, key: str, dataset: VeraDataset) -> None:
        """Attach an in-memory dataset that is never evicted."""
        self._pinned[key] = dataset

    def cache_dataset(self, name: str) -> None:
        """Read a dataset and hold it resident."""
        if name not in self._names:
            raise KeyError(name)
        ds = self._make_dataset(name)
        if ds is not None:
            self._cache[name] = ds

    def uncache_dataset(self, name: str) -> None:
        """Drop a resident dataset so access reverts to lazy."""
        self._cache.pop(name, None)

    def cache_all(self) -> None:
        """Load every managed dataset into memory."""
        for name in self._names:
            self.cache_dataset(name)

    def uncache_all(self) -> None:
        """Drop every resident dataset back to lazy."""
        self._cache.clear()


def _make_ji_safe(j: int, i: int, array: np.ndarray):
    """Return j,i clipped to array"""
    max_row, max_col = np.shape(array)
    j = np.clip(j, 0, max_row - 1)
    i = np.clip(i, 0, max_col - 1)
    return j, i


def _nearest_nonzero_ij(array, j, i):
    """Nearest non-zero cell to (j, i) by squared Euclidean distance."""
    # rows are [row, col] == [j, i]
    cells = np.argwhere(array > 0)
    if cells.size == 0:
        return None
    d = (cells[:, 0] - j) ** 2 + (cells[:, 1] - i) ** 2
    nj, ni = cells[np.argmin(d)]
    return int(nj), int(ni)


FALLBACK_AXIAL_MESH = np.array([0, 20, 40, 60])
DEFAULT_AXIAL_MESH_STEP = 10
DEFAULT_PIN_PITCH = 1.26  # cm


class CorePropMissing(Exception):
    """The file doesn't specify everything needed to build the core.

    missing:  {key: {"label": str, "allow_none": bool}}  — what to ask for
    inferred: {key: {"value": Any, "source": str}}       — what was determined
    """

    def __init__(self, missing: dict, inferred: dict):
        self.missing = missing
        self.inferred = inferred
        super().__init__(f"Core props missing: {', '.join(missing)}")


def col_label(index: int) -> str:
    label = ""
    while index >= 0:
        index, rem = divmod(index, 26)
        label = chr(ord("A") + rem) + label
        index -= 1
    return label


class VeraOutCore(DatasetStore):
    """Core-level geometry shared by every state: maps, meshes, labels, shapes.

    Reads from a DatasetSource, so the same class serves file, stream and
    synthetic sources.
    """

    def __init__(
        self,
        dataset_source: DatasetSource,
        overrides: CoreOverride | None = None,
    ):
        """Build the core from a source covering the CORE group.

        args:
            dataset_source: supplies core_map, core_sym, axial_mesh, etc.
            overrides: user-supplied npin / nax, used when the source
                underspecifies the lattice. Outranks anything discovered.

        Raises CorePropMissing when the pin lattice or axial extent can't be
        determined, and RuntimeError when core_map is absent.
        """
        if overrides is None:
            overrides = {}

        super().__init__(source=dataset_source)
        self.core_map = self.get("core_map")
        if self.core_map is None:
            raise RuntimeError("[ERROR] core_map not found in source, unable to visualize data")
        core_sym = self.get("core_sym")
        self.core_sym = core_sym[0] if core_sym is not None else None
        if self.core_sym is None:
            print("[Warning] core_sym not found in source, will try to infer from core map")
        self.pin_volumes = self.get("pin_volumes")
        aspect_ratio = self.get("aspect_ratio")
        self.aspect_ratio = aspect_ratio[0] if aspect_ratio is not None else 1
        self.axial_mesh = self.get("axial_mesh")
        self._determine_core_shape(self.core_map, overrides)
        self._check_missing()
        if not self.has_axial_mesh():
            self.axial_mesh = FALLBACK_AXIAL_MESH
        self._determine_computational_core_shape()
        for geometry_checkers in (
            "STATE_0001/pin_powers",
            "pin_factors",
            "pin_heated_surface_area",
            "pin_volumes",
        ):
            ref_shape = self.shape(geometry_checkers)
            if ref_shape is None:
                continue
            _, _, _, ref_nass = ref_shape
            if ref_nass == self.comp_nass:
                self.nass = ref_nass
                self.core_map = self.comp_core_map
                break
        self._determine_detectors()
        self.shape_to_dtype = build_core_dtypes(
            npiny=self.npy,
            npinx=self.npx,
            nax=self.nax,
            nass=self.nass,
            comp_nax=self.comp_nax,
            comp_nass=self.comp_nass,
            ndet=self.ndet,
            ndax=self.ndax,
            continous_det=self.is_continous_detector,
        )
        self._compute_reduced_core_maps()
        self._determine_core_labels()
        self._compute_axial_mesh_pixels()
        self._compute_non_fuel_locs()
        self._compute_axial_mesh_means()

    def _determine_core_shape(self, cm: np.ndarray, overrides: CoreOverride | None = None):
        if overrides is None:
            overrides = {}
        self.nass = int(np.count_nonzero(np.unique(cm[~np.isnan(cm)])))
        if self.core_sym is None:
            if int(np.count_nonzero(cm[~np.isnan(cm)])) == self.nass:
                self.core_sym = 1
            else:
                self.core_sym = 4
        self.nax = None
        self.npy = None
        self.npx = None
        self._npin_src = self._nax_src = "Could not find"
        if self.has_axial_mesh():
            self.nax = len(self.axial_mesh) - 1
            self._nax_src = "/CORE/axial_mesh"
        self._pin_pitch = DEFAULT_PIN_PITCH
        self.apitch = DEFAULT_PIN_PITCH * 17
        for pin_ds_name in ("npin", "num_pins"):
            npins = self.get(pin_ds_name)
            if npins is not None:
                self.npy = self.npx = int(npins[0])
                self._npin_src = f"/CORE/{pin_ds_name}"
                break

        for geometry_ds_name in (
            "pin_factors",
            "pin_heated_surface_area",
            "pin_volumes",
            "STATE_0001/pin_powers",
        ):
            core_geometry = self.shape(geometry_ds_name)
            if core_geometry and len(core_geometry) == 4:
                self.npy, self.npx, _nax, _nass = core_geometry
                is_state = geometry_ds_name.startswith("STATE_")
                if self.nax is None:
                    self.nax = _nax
                    self._nax_src = (
                        f"/CORE/{geometry_ds_name}" if not is_state else geometry_ds_name
                    )
                elif self.nax != _nax:
                    raise RuntimeError(
                        f"Mismatch between axial levels {_nax} from {geometry_ds_name} and axial levels {self.nax} from {self._nax_src}"
                    )
                self._npin_src = f"/CORE/{geometry_ds_name}" if not is_state else geometry_ds_name

        if "npin" in overrides:
            self.npy = self.npx = overrides["npin"]
            self._npin_src = "Overrides"
        if "nax" in overrides:
            nax = overrides["nax"]
            self.nax = nax
            self.axial_mesh = np.linspace(0, nax * DEFAULT_AXIAL_MESH_STEP, nax + 1)
            self._nax_src = "Overrides"

        apitch = self.get("apitch")
        if apitch is not None and self.npx:
            self.apitch = apitch[0]
            self._pin_pitch = float(apitch[0] / self.npx)

        if not self.has_axial_mesh() and self.nax:
            self.axial_mesh = np.linspace(0, self.nax * DEFAULT_AXIAL_MESH_STEP, self.nax + 1)

    def _check_missing(self):
        missing = {}
        inferred = {"nass": {"value": int(self.nass), "source": "/CORE/core_map"}}
        if self.npy is None or self.npx is None:
            missing["npin"] = {"label": "Pins across an assembly", "allow_none": True}
        else:
            inferred["npin"] = {"value": int(self.npy), "source": self._npin_src}
        if self.nax is None:
            missing["nax"] = {"label": "Number of axial layers", "allow_none": False}
        else:
            inferred["nax"] = {"value": int(self.nax), "source": self._nax_src}
        if missing:
            raise CorePropMissing(missing, inferred)

    def _determine_computational_core_shape(self):
        self.comp_nass = None
        self.comp_nax = None

        if self.shape("computational_core_map") is None:
            print(
                "Could not find computational_core_map, unable to determine computational core shape"
            )
            return
        if self.shape("computational_axial_mesh"):
            comp_axial_mesh = self.get("computational_axial_mesh")
        elif self.shape("STATE_0001/NODAL_XS/AXIALMESH"):  # fallback location
            comp_axial_mesh = self.get("STATE_0001/NODAL_XS/AXIALMESH")
        else:
            print(
                "Could not find computational axial_mesh, unable to determine computational core shape"
            )
            return
        self.comp_core_map = self.get("computational_core_map")
        self.comp_nass = int(
            np.count_nonzero(np.unique(self.comp_core_map[~np.isnan(self.comp_core_map)]))
        )
        self.comp_axial_mesh = comp_axial_mesh
        self.comp_nax = len(self.comp_axial_mesh) - 1
        self.comp_core_map[np.isnan(self.comp_core_map)] = 0
        self._is_comp_rolled = np.count_nonzero(self.comp_core_map) == np.count_nonzero(
            np.unique(self.comp_core_map)
        )

    def _determine_detectors(self, n_points=500):
        self.detector_map = None
        self.ndet = None
        self.det_axial_mesh_means = None
        self.ndax = self.nax
        self.is_continous_detector = False
        self.detector_map = self.get("detector_map")
        if self.detector_map is None:
            return
        self.ndet = int(
            np.count_nonzero(np.unique(self.detector_map[~np.isnan(self.detector_map)]))
        )
        if not self.shape("detector_axial_mesh"):
            return
        self.det_axial_mesh_means = self.get("detector_axial_mesh")
        self.ndax = len(self.det_axial_mesh_means)
        self.is_continous_detector = self.det_axial_mesh_means.ndim == 2
        if not self.is_continous_detector:
            return

    def _compute_reduced_core_maps(self) -> None:
        """Compute the reduced core map based upon the core_sym"""
        sym = self.core_sym
        has_comp_core = self.has_comp_core()
        if sym == 1:
            self.reduced_core_map = self.core_map[:].copy()
            self.reduced_core_map_start_index = 0
            self.comp_map_start_index = 0
        elif sym == 4:
            w, h = self.core_map[:].shape
            start_w = w // 2
            start_h = h // 2
            self.reduced_core_map = self.core_map[start_w:, start_h:]
            self.reduced_core_map_start_index = start_w
            if has_comp_core and not self._is_comp_rolled:
                cw, ch = self.comp_core_map[:].shape
                cstart_w = cw // 2
                cstart_h = ch // 2
                self.comp_core_map = self.comp_core_map[cstart_w:, cstart_h:]
                self.comp_map_start_index = cstart_w
            elif self.has_comp_core():
                self.comp_map_start_index = start_w
            if self.detector_map is not None:
                self.detector_map = self.detector_map[start_w:, start_h:]
        else:
            raise Exception(f"Unhandled symmetry: {sym}")

    def _determine_core_labels(self):
        """Assign row/column labels, falling back to letters and numbers.

        Must be called AFTER `self.reduced_core_map` is set.
        """
        num_rows, num_cols = self.reduced_core_map.shape
        start_index = (
            self.reduced_core_map_start_index
            if hasattr(self, "reduced_core_map_start_index")
            else 0
        )

        if (raw_xlabels := self.get("xlabel")) is not None:
            xlabels = [char.decode() for char in raw_xlabels[start_index:]]
        else:
            xlabels = list(reversed([col_label(i) for i in range(num_cols)]))
        self.reduced_core_map_column_labels = xlabels

        if (raw_ylabels := self.get("ylabel")) is not None:
            ylabels = [char.decode() for char in raw_ylabels[start_index:]]
        else:
            ylabels = [str(num) for num in range(start_index + 1, start_index + num_rows + 1)]
        self.reduced_core_map_row_labels = ylabels
        if not self.has_comp_core():
            return
        comp_num_rows, comp_num_cols = self.comp_core_map.shape
        self.comp_core_map_column_labels = list(
            reversed([col_label(i) for i in range(comp_num_cols)])
        )
        self.comp_core_map_row_labels = [
            str(idx) for idx in range(start_index + 1, start_index + comp_num_rows + 1)
        ]

    def _compute_axial_mesh_pixels(self) -> None:
        """Compute the number of pixels that we will be displaying in
        the axial direction for each length in the axial mesh.
        """
        diff_array = np.diff(self.axial_mesh[:])

        # The min diff will be three pixels high. The rest will be computed based
        # upon the min diff.
        MIN_DIFF_PIXELS_HEIGHT = 3
        pixel_height = np.min(diff_array) / MIN_DIFF_PIXELS_HEIGHT
        pixel_height_array = diff_array / pixel_height
        self.axial_mesh_pixels = np.round(pixel_height_array).astype(np.int64)

        if self.has_comp_axial_mesh():
            comp_diff_array = np.diff(self.comp_axial_mesh)
            comp_pixel_height = np.min(comp_diff_array) / MIN_DIFF_PIXELS_HEIGHT
            comp_pixel_height_array = comp_diff_array / comp_pixel_height
            self.comp_axial_mesh_pixels = np.round(comp_pixel_height_array).astype(np.int64)

    def _compute_non_fuel_locs(self) -> None:
        """Locate non-fuel positions as the pins with zero volume."""
        if self.pin_volumes is not None:
            self.non_fuel_locs = np.where(self.pin_volumes == 0)
        else:
            self.non_fuel_locs = None

    def _compute_axial_mesh_means(self):
        """Midpoint between each pair of neighboring mesh boundaries."""
        self.axial_mesh_means = self._midpoints(self.axial_mesh)
        self.gross_axial_mesh = self.axial_mesh_means
        if self.has_comp_axial_mesh():
            self.comp_axial_mesh_means = self._midpoints(self.comp_axial_mesh)
            self.gross_axial_mesh = np.union1d(self.gross_axial_mesh, self.comp_axial_mesh_means)
        if self.det_axial_mesh_means is not None:
            self.gross_axial_mesh = np.union1d(self.gross_axial_mesh, self.det_axial_mesh_means)
        else:
            # no detector axial mesh means were found in the core, use axial_mesh as reference
            self.det_axial_mesh_means = self.axial_mesh_means

    @staticmethod
    def _midpoints(mesh, decimals=4):
        mesh = np.asarray(mesh, dtype=np.float64)
        return np.round((mesh[:-1] + mesh[1:]) / 2.0, decimals)

    def has_axial_mesh(self):
        return hasattr(self, "axial_mesh") and self.axial_mesh is not None

    def has_comp_core(self) -> bool:
        return hasattr(self, "comp_core_map") and self.comp_core_map is not None

    def has_comp_axial_mesh(self) -> bool:
        return hasattr(self, "comp_axial_mesh") and self.comp_axial_mesh is not None

    def is_even(self) -> bool:
        return self.core_map.shape[0] % 2 == 0

    @property
    def pin_pitch(self):
        """The pitch (cm) of a single fuel pin"""
        return self._pin_pitch

    @property
    def core_shape(self) -> tuple[int, ...]:
        """Shape of core (num_piny, num_pinx, num_axial_levels, num_assemblys).
        npy/npx are 0 when the core has no pin lattice."""
        if self.npy is None or self.npx is None or self.nax is None:
            raise RuntimeError("Core pin lattice is undetermined; complete characteristics first")
        return (self.npy, self.npx, self.nax, self.nass)

    @property
    def comp_core_shape(self) -> tuple[int, ...]:
        """Shape of computational core (num_piny, num_pinx, num_computational_axial_levels, num_computational_assemblys)
        npy/npx are 0 when the core has no pin lattice."""
        if self.npy is None or self.npx is None or self.comp_nax is None or self.comp_nass is None:
            raise RuntimeError("Core pin lattice is undetermined; complete characteristics first")
        return (self.npy, self.npx, self.comp_nax, self.comp_nass)

    @property
    def detector_shape(self) -> tuple[int, ...]:
        """Shape of detector shape (ndax, ndet)"""
        if self.npy is None or self.npx is None or self.ndax is None or self.ndet is None:
            raise RuntimeError("Detector shape is undetermined")
        return (self.npy, self.npx, self.ndax, self.ndet)

    def core_dtypes(self, dataset_shape: tuple[int, ...]) -> VeraDtype:
        """Get VeraDtype associated with dataset_shape

        args:
            dataset_shape: the shape of the dataset

        Return VeraDtype.UNKNOWN if shape is not known.
        """
        return self.shape_to_dtype.get(dataset_shape, VeraDtype.UNKNOWN)

    def get_core_shape(
        self, dataset: VeraDataset = None, dataset_type: VeraDtype = VeraDtype.UNKNOWN
    ):
        """get corresponding core shape for `dataset`"""
        dtype = dataset.dataset_type if dataset is not None else dataset_type
        if dtype.is_computational() and self.has_comp_core():
            return self.comp_core_shape
        elif (
            dtype.is_detector()
            and self.detector_map is not None
            and self.det_axial_mesh_means is not None
        ):
            return self.detector_shape
        elif dtype != VeraDtype.UNKNOWN:
            return self.core_shape
        else:
            raise RuntimeError(f"Could not find core shape for dataset of type {str(dtype)}")

    def get_map(
        self,
        dataset: VeraDataset | None = None,
        dataset_type: VeraDtype = VeraDtype.UNKNOWN,
    ):
        """get corresponding core map for `dataset`"""
        dtype = dataset.dataset_type if dataset is not None else dataset_type
        if dtype.is_computational() and self.has_comp_core():
            return self.comp_core_map
        elif dtype.is_detector() and self.detector_map is not None:
            return self.detector_map
        elif dtype != VeraDtype.UNKNOWN:
            return self.reduced_core_map
        else:
            raise RuntimeError(f"Could not find map for dataset of type {str(dtype)}")

    def get_axial_mesh(
        self, dataset: VeraDataset = None, dataset_type: VeraDtype = VeraDtype.UNKNOWN
    ):
        """get corresponding axial mesh for `dataset`"""
        dtype = dataset.dataset_type if dataset is not None else dataset_type
        if dtype.is_computational() and self.has_comp_axial_mesh():
            return self.comp_axial_mesh
        elif dtype != VeraDtype.UNKNOWN:
            return self.axial_mesh
        else:
            raise RuntimeError(f"Could not find axial mesh for dataset of type {str(dtype)}")

    def get_axial_mesh_means(
        self, dataset: VeraDataset = None, dataset_type: VeraDtype = VeraDtype.UNKNOWN
    ):
        """get corresponding axial mesh means for `dataset`"""
        dtype = dataset.dataset_type if dataset is not None else dataset_type
        if dtype.is_computational() and self.has_comp_axial_mesh():
            return self.comp_axial_mesh_means
        elif dtype.is_detector() and self.det_axial_mesh_means is not None:
            return self.det_axial_mesh_means
        elif dtype != VeraDtype.UNKNOWN:
            return self.axial_mesh_means
        else:
            raise RuntimeError(f"Could not find axial mesh means for dataset of type {str(dtype)}")

    def get_axial_mesh_pixels(
        self, dataset: VeraDataset = None, dataset_type: VeraDtype = VeraDtype.UNKNOWN
    ):
        """get corresponding axial mesh pixels for `dataset`"""
        dtype = dataset.dataset_type if dataset is not None else dataset_type
        if dtype.is_computational() and self.has_comp_axial_mesh():
            return self.comp_axial_mesh_pixels
        elif dtype != VeraDtype.UNKNOWN:
            return self.axial_mesh_pixels
        else:
            raise RuntimeError(f"Could not find axial mesh means for dataset of type {str(dtype)}")

    def row_assembly_indices(self, assembly_idx, is_comp=False, is_detector=False) -> np.ndarray:
        """Get indices of all assemblies in the same row as this assembly"""
        # The core map and reduced core map use 1-based indexing
        if is_comp and is_detector:
            raise RuntimeError("No comp detector map currently")
        cm = self.comp_core_map if is_comp and self.has_comp_core() else self.reduced_core_map
        cm = cm if not is_detector else self.detector_map
        row = np.where(cm == assembly_idx + 1)[0][0]
        ids = cm[row]
        # Remove any zeros
        return ids - 1

    def col_assembly_indices(self, assembly_idx, is_comp=False, is_detector=False) -> np.ndarray:
        """Get indices of all assemblies in the same column as this assembly"""
        if is_comp and is_detector:
            raise RuntimeError("No comp detector map currently")
        cm = self.comp_core_map if is_comp and self.has_comp_core() else self.reduced_core_map
        cm = cm if not is_detector else self.detector_map
        col = np.where(cm == assembly_idx + 1)[1][0]
        ids = cm[:, col]
        # Remove any zeros
        return ids - 1

    def reduced_core_map_assembly(self, i, j, is_comp=False, is_detector=False) -> int:
        """Get the index of the assembly at reduced core map position i, j

        clamps to the nearest real assembly in the same map, so the result
        is always a valid assembly when the map has any. Returns -1 only if the
        chosen map has no assemblies at all.
        """
        if is_comp and is_detector:
            raise RuntimeError("No comp detector map currently")
        cm = self.comp_core_map if is_comp and self.has_comp_core() else self.reduced_core_map
        cm = cm if not is_detector else self.detector_map
        j, i = _make_ji_safe(j, i, cm)
        snapped = _nearest_nonzero_ij(cm, j, i)
        if snapped is None:
            return -1
        j, i = snapped
        return int(cm[j, i]) - 1

    def reduced_core_map_ij(self, assembly_idx, is_comp=False) -> tuple[int, int]:
        """Return the (column, row) position of an assembly in the reduced map."""
        target = assembly_idx + 1
        cm = self.comp_core_map if is_comp and self.has_comp_core() else self.reduced_core_map
        rows, cols = np.where(cm == target)
        if len(rows) == 0:
            raise ValueError(
                f"Assembly index {assembly_idx} was not found in reduced_core_map. \nLooked for value {target}."
            )
        if len(rows) > 1:
            raise ValueError(
                f"Assembly index {assembly_idx} appears multiple times in reduced_core_map. \n Looked for value {target}; found {len(rows)} matches."
            )
        j = int(rows[0])
        i = int(cols[0])
        return i, j

    def reduced_core_map_label(self, assembly_idx, is_comp=False) -> str:
        """Return the combined column-row label for an assembly (e.g. C-9)."""
        row_label = self.reduced_core_map_row_label(assembly_idx, is_comp)
        col_label = self.reduced_core_map_column_label(assembly_idx, is_comp)
        return f"{col_label}-{row_label}"

    def reduced_core_map_row_label(self, assembly_idx, is_comp=False) -> str:
        """Return the row-number label for an assembly."""
        i, j = self.reduced_core_map_ij(assembly_idx, is_comp)
        if is_comp and self.has_comp_core():
            return self.comp_core_map_row_labels[j]
        else:
            return self.reduced_core_map_row_labels[j]

    def reduced_core_map_column_label(self, assembly_idx, is_comp=False) -> str:
        """Return the column-letter label for an assembly."""
        i, j = self.reduced_core_map_ij(assembly_idx, is_comp)
        labels = (
            self.reduced_core_map_column_labels[i]
            if not is_comp
            else self.comp_core_map_column_labels[i]
        )
        return labels


class VeraOutState(DatasetStore):
    def __init__(self, source: DatasetSource, idx: int, core: VeraOutCore):
        self.categorized_ds_names = {dataset_type: set() for dataset_type in VeraDtype}
        self.dataset_dtypes = {}
        self._core = core
        self._index = idx
        super().__init__(source)
        source_names = frozenset(source.names())
        for name in source_names:
            ds_shape = self.shape(name)
            if ds_shape is None:
                continue
            ds_dtype = core.core_dtypes(ds_shape)
            if ds_dtype == VeraDtype.UNKNOWN:
                continue
            self.categorized_ds_names[ds_dtype].add(name)
        self.diff_datasets = dict()
        self.derived_datasets = dict()

    @property
    def core(self) -> VeraOutCore:
        """Reference to this state's core data"""
        return self._core

    @property
    def scalar_datasets(self) -> list[str]:
        """List of dataset names categorized as SCALAR."""
        return sorted(self.categorized_ds_names[VeraDtype.SCALAR])

    @property
    def grouped_full_core_keys(self) -> list[tuple[str, list[str]]]:
        """Group the available dataset names by category for grouped UI display.

        Builds in _CATEGORY_ORDER and, for each one that has any
        datasets, returns a (category_name, sorted_names) pair. Empty categories
        are skipped.
        """
        groups = []
        for dtype in VeraDtype:
            names = sorted(self.categorized_ds_names[dtype])
            if names:
                groups.append((str(dtype), names))
        return groups

    @property
    def full_core_keys(self) -> list[str]:
        """Flat, category-ordered list of all dataset names."""
        return [name for _, names in self.grouped_full_core_keys for name in names]

    def add_diff_dataset(self, dataset_name: str, dataset: VeraDataset) -> None:
        """Attach an in-memory diff dataset to this state.

        The dataset is set as an attribute and tracked in diff_datasets.
        It is not categorized and not written to the file.
        """
        self.pin(dataset_name, dataset)
        self.diff_datasets.update({dataset_name: dataset.dataset_type})
        ds_dtype = dataset.dataset_type
        if ds_dtype == VeraDtype.UNKNOWN:
            return
        self.categorized_ds_names[ds_dtype].add(dataset_name)

    def add_derived_dataset(self, dataset_name: str, dataset: VeraDataset) -> None:
        """Attach an in-memory derived dataset to this state.

        Like add_diff_dataset, but also categorizes the dataset by shape so it
        shows up in the grouped key listings. Not written to the file.
        """
        self.pin(dataset_name, dataset)
        self.derived_datasets.update({dataset_name: dataset.dataset_type})
        ds_dtype = dataset.dataset_type
        if ds_dtype == VeraDtype.UNKNOWN:
            return
        self.categorized_ds_names[ds_dtype].add(dataset_name)


_TIME_AXIS_NAMES = (
    "exposure",
    "core_exposure",
    "exposure_efpd",
    "time_us",
)


class VeraDataSource:
    """One loaded VERA calculation: a core, its states, and lookup across both.

    Built by a reader rather than opened directly, so it is independent of where
    the data came from. Derivation needs the pyvera calculator and is therefore
    file-only; diffs work for any source.
    """

    def __init__(
        self,
        core: VeraOutCore,
        states: list[VeraOutState],
        provenance: str,
        filename: str | None = None,
        name: str | None = None,
        close_callback: Callable[[], None] | None = None,
        active_state_idx: int = 0,
        state_caching: bool = True,
    ):
        self._state_caching = state_caching
        self.vera_calculator = None
        if filename:
            try:
                self.vera_calculator = VERAout(
                    filename=filename
                )  # from pyvera, use this for calculating avgs
            except Exception as e:
                print(
                    "[Warning] could not load file for VERAout to read from for derivation:", str(e)
                )
                self.vera_calculator = None
        self._core = core
        self._states = states
        self._time_axes_dirty = True
        self.active_state_index = max(0, min(active_state_idx, len(self._states) - 1))
        self._determine_time_axes()
        self._provenance = provenance
        self._close_callback = close_callback
        self.name = name if name is not None else ""

    def _sample_time_axis_value(self, state, name):
        sampler = getattr(state.source, "sample", None)

        if sampler is not None:
            value = sampler(name, ())
        else:
            dataset = state.get(name)
            value = dataset.item() if dataset is not None else None

        if value is None:
            return None

        return np.asarray(value).item()

    def _determine_time_axes(self):
        """
        Determine time-axis names from the first state.
        NOTE this relies on the assumption that all states contains same time dataset
        """
        self._time_axes = {
            "state_count": list(range(len(self._states))),
        }

        if not self._states:
            return

        first_state = self._states[0]

        for name in _TIME_AXIS_NAMES:
            if name not in first_state:
                continue

            self._time_axes[name] = [
                self._sample_time_axis_value(state, name) for state in self._states
            ]

    def time_axes(self):
        return self._time_axes

    @property
    def provenance(self) -> str:
        """Where the data came from"""
        return self._provenance

    @property
    def core(self) -> VeraOutCore:
        """The core-level data shared across all states."""
        return self._core

    def close(self):
        """Release any open handles/ports/etc held by the source."""
        if self._close_callback is not None:
            self._close_callback()
        if self.vera_calculator is not None:
            self.vera_calculator.h5f.close()

    def default_datasets(self) -> dict[str, str]:
        categorized_ds_names = self.active_state.categorized_ds_names
        default_names = {
            category.title: sorted(names)[0]
            for category, names in categorized_ds_names.items()
            if category != VeraDtype.UNKNOWN and names
        }
        if not default_names:
            return {}
        if "pin_powers" in categorized_ds_names.get(VeraDtype.PIN, ()):
            default_names[VeraDtype.PIN.title] = "pin_powers"
        return default_names

    @property
    def states(self):
        """All state points in the source, in order."""
        return self._states

    @property
    def active_state(self):
        """The currently selected state."""
        return self.states[self.active_state_index]

    @property
    def active_state_index(self):
        """Index of the active state within states."""
        return self._active_state_index

    @active_state_index.setter
    def active_state_index(self, index: int):
        """Set the active state, clamping to range and no-opping if unchanged.

        Switching states uncaches the previous active state and caches the new
        one
        """
        if len(self._states) == 0:
            return
        index = max(0, min(index, len(self._states) - 1))
        if hasattr(self, "_active_state_index"):
            if self._active_state_index == index:
                return
            elif self._state_caching and self.active_state_index < len(self.states):
                # Clear the cache from the active state
                self.active_state.uncache_all()

        self._active_state_index = index
        if self._state_caching:
            self.active_state.cache_all()

    def add_state(self, state):
        """
        Add one state.
        Assumes it has the same time axes as the existing states.
        """
        self._states.append(state)
        self._time_axes["state_count"].append(len(self._states) - 1)
        if len(self._states) == 1:
            self._determine_time_axes()
            return
        for name, values in self._time_axes.items():
            if name == "state_count":
                continue
            values.append(self._sample_time_axis_value(state, name))

    def replace_state(self, idx, state):
        """
        Replace one state.
        Assumes the replacement has the same time axes.
        """
        self._states[idx] = state
        for name, values in self._time_axes.items():
            if name == "state_count":
                continue
            values[idx] = self._sample_time_axis_value(state, name)

    def remove_state(self, idx):
        """
        Remove one state.
        Removing a state cannot change which time axes exist unless this was
        the final state. (Again assuming all states had uniform time dataset headers)
        """
        removed = self._states.pop(idx)
        if self.active_state_index >= len(self._states):
            self.active_state_index = max(0, len(self._states) - 1)
        if not self._states:
            self._time_axes = {
                "state_count": [],
            }
            return removed
        for name, values in self._time_axes.items():
            if name == "state_count":
                continue
            values.pop(idx)

        # state_count is always [0, 1, ..., N - 1].
        self._time_axes["state_count"].pop()

        return removed

    def add_new_diff_dataset(
        self,
        ref_dataset_name: str,
        comp_src: "VeraDataSource",
        comp_dataset_name: str,
        new_diff_name: str,
        interpolation_order: int = 1,
        ref_scale: float = 1.0,
        comp_scale: float = 1.0,
        units: str = "unitless",
    ):
        """Create a difference dataset (ref minus comp) on each state."""
        if self is comp_src and ref_dataset_name == comp_dataset_name:
            raise ValueError("Cannot diff a dataset against itself, trivially zero")
        if any(new_diff_name in s for s in self._states):
            raise ValueError(
                f"A dataset named {new_diff_name} already exists in this source, "
                "please pick a unique name."
            )
        pending: list[tuple[VeraOutState, VeraDataset]] = []
        for idx, state in enumerate(self._states):
            if idx >= len(comp_src.states):
                break
            comp_state = comp_src.states[idx]
            if ref_dataset_name not in state or comp_dataset_name not in comp_state:
                continue
            ref_data: VeraDataset = state.get(ref_dataset_name) * ref_scale
            comp_data: VeraDataset = comp_state.get(comp_dataset_name) * comp_scale

            if ref_data.dataset_type != comp_data.dataset_type:
                continue

            has_axial_dim = ref_data.dataset_type.has_axial_dim()
            ax = ref_data.dataset_type.axial_dim_idx if has_axial_dim else None
            ref_other = tuple(d for i, d in enumerate(ref_data.shape) if i != ax)
            comp_other = tuple(d for i, d in enumerate(comp_data.shape) if i != ax)

            if ref_other != comp_other:
                raise ValueError(
                    f"'{ref_dataset_name}' {ref_data.shape} and '{comp_dataset_name}' "
                    f"{comp_data.shape} differ outside the axial dimension"
                )

            if not has_axial_dim:
                diff = ref_data - comp_data
            else:
                ref_means = self.core.get_axial_mesh_means(dataset=ref_data)
                comp_means = comp_src.core.get_axial_mesh_means(dataset=comp_data)
                if ref_means.shape == comp_means.shape and np.allclose(ref_means, comp_means):
                    diff = ref_data - comp_data
                else:
                    spl = make_interp_spline(comp_means, comp_data, k=interpolation_order, axis=ax)
                    diff = ref_data - spl(ref_means, extrapolate=False)
            diff.physical_units = units
            diff.name = new_diff_name
            pending.append((state, diff))
        if not pending:
            raise ValueError(f"No overlapping/compatible states to diff for '{new_diff_name}'")
        for state, diff in pending:
            state.add_diff_dataset(new_diff_name, diff)

    def _run_avg_over_axes(self, data, axes: VeraAxes = VeraAxes.CORE):
        """Reduce data over the given axes using the VERAout calculator.

        Dispatches to the matching VERAout averaging routine and wraps the result
        as a VeraDataset of the corresponding type; raises ValueError for an
        unsupported axes value.
        """
        match axes:
            case VeraAxes.ASSEMBLY:
                der = VeraDataset(
                    self.vera_calculator.Assembly(data)[np.newaxis, ...],
                    VeraDtype.ASSEMBLY,
                )
            case VeraAxes.AXIAL:
                der = VeraDataset(self.vera_calculator.Axial(data), VeraDtype.AXIAL)
            case VeraAxes.CORE:
                der = VeraDataset(np.array([self.vera_calculator.Average(data)]), VeraDtype.SCALAR)
            case VeraAxes.NODE:
                der = VeraDataset(self.vera_calculator.Node(data), VeraDtype.NODAL)
            case VeraAxes.RADIAL:
                der = VeraDataset(self.vera_calculator.Radial(data), VeraDtype.RADIAL)
            case VeraAxes.RADIAL_ASSEMBLY:
                der = VeraDataset(
                    self.vera_calculator.Radial_Assembly(data),
                    VeraDtype.RADIAL_ASSEMBLY,
                )
            case _:
                raise ValueError(f"Derivation: {axes} not implemented")
        return der

    def add_new_derived_dataset(
        self,
        source_array_name: str,
        new_dataset_name: str,
        der_method: DerivationMethod,
        axes: VeraAxes,
        use_factors: bool = True,
    ):
        """Create a derived dataset from a source array using a reduction over axes."""
        if not self.vera_calculator:
            raise RuntimeError("Could not find necessary factor datasets to perform derivation.")
        pending = []
        for state in self._states:
            if new_dataset_name in state:
                raise ValueError(
                    f"A dataset named {new_dataset_name} already exists in this source, please pick a unique name."
                )
            data = state.get(source_array_name, None)
            if data is None:
                continue
            match der_method:
                case DerivationMethod.AVERAGE:
                    der = self._run_avg_over_axes(data, axes)
                case DerivationMethod.STDDEV:
                    mean = self._run_avg_over_axes(data)
                    var = self._run_avg_over_axes((data - mean) ** 2, axes)
                    der = np.sqrt(var)
                case DerivationMethod.RMS:
                    der = np.sqrt(self._run_avg_over_axes(data**2, axes))
            der.name = new_dataset_name
            pending.append((state, new_dataset_name, der))
        for state, new_dataset_name, der in pending:
            state.add_derived_dataset(new_dataset_name, der)

    def _get_dataset(self, ds_name: str, state_idx: int | None = None) -> VeraDataset | None:
        """Resolve a named array to its VeraDataset, without masking.

        Core arrays (pin_volumes) come from the core and ignore `state_idx`.
        Everything else comes from `states[state_idx]`, or the active state when
        None. Negative indices are rejected rather than wrapping.

        Returns None if the name is unknown. Raises IndexError if `state_idx` is
        out of range. Reads are lazy: an uncached name is read on access.
        """
        state_idx = (
            max(0, min(state_idx, len(self.states) - 1)) if state_idx is not None else state_idx
        )
        if state_idx is not None and not 0 <= state_idx < len(self.states):
            raise IndexError(f"{state_idx} out of index range")
        arrays_on_core = [
            "pin_volumes",
        ]
        if ds_name in arrays_on_core:
            ds_src = self.core
        elif state_idx is not None:
            ds_src = self.states[state_idx]
        else:
            ds_src = self.active_state
        ds = ds_src.get(ds_name, None)
        if not isinstance(ds, VeraDataset):
            return None
        return ds

    def get_dataset(
        self,
        array_name: str,
        *,
        mask_reflected: bool = True,
        state_idx: int | None = None,
    ) -> VeraDataset:
        """Return a named array, masked and ready for use.

        On odd-sized quarter cores, reflected positions are set to NaN so
        mirrored data isn't double-counted. Even cores and mask_reflected=False
        skip this.

        Always returns a writable copy that the caller owns.

        Raises RuntimeError if the name is not found, IndexError if `state_idx`
        is out of range.
        """
        raw = self._get_dataset(array_name, state_idx)
        if raw is None:
            raise RuntimeError(f"Could not find dataset/array named {array_name}.")
        array = raw.copy()
        cm = (
            self.core.reduced_core_map
            if not array.dataset_type.is_computational()
            else self.core.comp_core_map
        )
        is_even_core = self.core.is_even()
        if mask_reflected and not is_even_core:
            array = nan_out_reflected(cm, self.core.core_sym, array)
        return array

    def get_dataset_dtype(self, array_name: str, state_idx: int | None = None) -> VeraDtype:
        """Return the VeraDtype of a named array, or UNKNOWN if not found.

        Resolves against the core for core arrays and `state_idx` (or the active
        state) otherwise. Raises IndexError if `state_idx` is out of range.
        """
        ds = self._get_dataset(array_name, state_idx)
        return ds.dataset_type if ds is not None else VeraDtype.UNKNOWN

    def get_dataset_units(self, array_name: str, state_idx: int | None = None) -> str:
        """Return the physical units of a named array, or "Unitless" if not found.

        Resolves against the core for core arrays and `state_idx` (or the active
        state) otherwise. Raises IndexError if `state_idx` is out of range.
        """
        ds = self._get_dataset(array_name, state_idx)
        return ds.physical_units if ds is not None else "Unitless"

    def get_dataset_shape(self, array_name: str, state_idx: int | None = None) -> tuple | None:
        """Return the shape of a named array, or an empty tuple if not found.

        Resolves against the core for core arrays and `state_idx` (or the active
        state) otherwise. Raises IndexError if `state_idx` is out of range.
        """
        ds = self._get_dataset(array_name, state_idx)
        return tuple(np.shape(ds)) if ds is not None else None
