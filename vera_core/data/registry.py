import numpy as np

from .dtypes import VeraAxes, VeraDtype
from .model import VeraDataSource


def _recipe_sources(recipe: dict) -> set[str]:
    if recipe.get("all_sources", False):
        return set()
    return {recipe[key] for key in ("src_id", "ref_src_id", "comp_src_id") if recipe.get(key)}


class VeraDataRegistry:
    """Holds the open data sources keyed by id, with one marked as default.

    Provides lookup, the shared maximum state index, and fan-out of the active
    state across all sources.
    """

    def __init__(self):
        """Create an empty registry with no default source."""
        self._srcs: dict[str, VeraDataSource] = {}
        self.default_src_id: str = None
        self.global_axial_mesh = np.asarray([], dtype=np.float64)
        self._recipes: list[dict] = []
        self._auto_derivation: list[dict] = []

    def _compose_global_axial_mesh(self):
        global_axial_mesh = np.asarray([], dtype=np.float64)
        for src in self._srcs.values():
            core = src.core
            global_axial_mesh = np.union1d(global_axial_mesh, core.gross_axial_mesh)
        self.global_axial_mesh = global_axial_mesh

    def add_src(self, src: VeraDataSource, src_id: str):
        """Register a source under src_id, making it default if it's the first.
        Raises ValueError if src_id is already registered.
        """
        if src_id in self._srcs:
            raise ValueError(f"{src_id} already exists in the registry, skipped adding")
        self._srcs[src_id] = src
        core = src.core
        if self.default_src_id is None:
            self.default_src_id = src_id
            self.global_axial_mesh = core.gross_axial_mesh.copy()
        else:
            self.global_axial_mesh = np.union1d(self.global_axial_mesh, core.gross_axial_mesh)
        src.name = src_id
        for auto_derive_recipe in self._auto_derivation:
            try:
                src.add_new_derived_dataset(
                    source_array_name=auto_derive_recipe["source_array"],
                    new_dataset_name=auto_derive_recipe["name"],
                    der_method=auto_derive_recipe["method"],
                    axes=VeraAxes[auto_derive_recipe["axes"]],
                )
            except Exception as e:
                print(
                    f"Internal warning in auto applying derivation: {str(auto_derive_recipe)}, {str(e)}. This is auto so not a user error and can be safely ignored."
                )

    @property
    def default_src(self) -> VeraDataSource | None:
        """The default source, or None if the registry is empty."""
        if self.default_src_id is None:
            return None
        return self._srcs[self.default_src_id]

    @property
    def max_state(self) -> int:
        """Largest valid state index across all sources, or 0 if none."""
        if not self._srcs:
            return 0
        return max(max(len(src.states) for src in self._srcs.values()) - 1, 0)

    def get_axial_index(self, z: np.float64):
        return int(np.searchsorted(self.global_axial_mesh, z))

    def src_axial_idx_to_global_idx(self, src_id: str, ds_dtype: VeraDtype, idx: int) -> int:
        if src_id not in self._srcs:
            raise ValueError("src_id not in stored src_ids")
        core = self._srcs[src_id].core
        src_axial_mesh = core.get_axial_mesh_means(dataset_type=ds_dtype)
        physical_layer = src_axial_mesh[idx]
        global_idx = self.get_axial_index(physical_layer)
        assert self.global_axial_mesh[global_idx] == physical_layer
        global_idx = np.clip(global_idx, 0, len(self.global_axial_mesh) - 1)
        return int(global_idx)

    def global_axial_idx_to_src_idx(self, src_id: str, ds_dtype: VeraDtype, idx: int):
        if src_id not in self._srcs:
            raise ValueError("src_id :", src_id, "is not in registry")
        core = self._srcs[src_id].core
        physical_layer = self.global_axial_mesh[idx]
        src_axial_mesh = core.get_axial_mesh_means(dataset_type=ds_dtype)
        if src_axial_mesh.ndim == 2:
            src_axial_mesh = src_axial_mesh.ravel()
        src_idx = np.searchsorted(src_axial_mesh, physical_layer)
        src_idx = np.clip(src_idx, 0, len(src_axial_mesh) - 1)
        return int(src_idx)

    def has_src(self):
        return self.default_src_id is not None

    def get(self, src_id: str | None) -> VeraDataSource | None:
        """Return the source for src_id, or None if it isn't registered."""
        if src_id is None:
            return None
        return self._srcs.get(src_id, None)

    def get_ds_dtype(self, src_id: str, ds_name: str) -> VeraDtype:
        if src_id not in self._srcs:
            return VeraDtype.UNKNOWN
        src = self._srcs[src_id]
        return src.get_dataset_dtype(ds_name)

    def src_ids(self):
        """Return a view of all registered source ids."""
        return self._srcs.keys()

    def change_active_state(self, src_id: str, nstate: int) -> None:
        """Set the active state index for one source. Raises ValueError if src_id isn't registered."""
        if src_id not in self._srcs:
            raise ValueError(f"Could not find {src_id} in registry")
        self._srcs[src_id].active_state_index = nstate

    def change_all_active_state(self, nstate: int) -> None:
        """Set the active state index on every registered source."""
        for src in self._srcs.values():
            src.active_state_index = nstate

    def shared_time_axes(self):
        if not self._srcs:
            return []
        srcs = iter(self._srcs.values())
        shared_axes = set(next(srcs).time_axes())
        for src in srcs:
            shared_axes &= set(src.time_axes())
        return sorted(shared_axes)

    def time_axis_value(self, axis: str, state_index: int) -> float:
        """Value of `axis` at `state_index`, read from the source `selected_time`
        indexes against (the max-state source). Exact for that source; other
        sources with different sampling won't align perfectly"""
        if not self._srcs:
            return float(state_index)
        if axis == "state_count":
            return float(state_index)
        ref = max(self._srcs.values(), key=lambda s: len(s.states))
        axes = ref.time_axes()
        if axis in axes and 0 <= state_index < len(axes[axis]):
            return float(axes[axis][state_index])
        return float(state_index)

    def remove_src(self, src_id: str) -> None:
        """Remove a source, close its file handles, and reassigns the default. Raises ValueError if src_id isn't registered."""
        if src_id not in self._srcs:
            raise ValueError(f"Could not find {src_id} in registry")
        src = self._srcs.pop(src_id)
        src.close()
        if self.default_src_id == src_id:
            self.default_src_id = next(iter(self._srcs), None)
        self._recipes = [
            recipe for recipe in self._recipes if src_id not in _recipe_sources(recipe)
        ]
        self._compose_global_axial_mesh()

    def all_sources(self) -> dict[str, str]:
        src_paths = {}
        for src_id, src in self._srcs.items():
            src_paths[src_id] = src.provenance
        return src_paths

    def clear(self):
        for src in self._srcs.values():
            src.close()
        self._srcs = {}
        self._recipes = []
        self._auto_derivation = []
        self.global_axial_mesh = np.asarray([], dtype=np.float64)
        self.default_src_id = None

    def apply_recipe(self, recipe: dict):
        kind = recipe["kind"]
        if kind == "derive":
            src_array_name = recipe["source_array"]
            all_sources = recipe.get("all_sources", False)
            if not all_sources:
                src = self.get(recipe["src_id"])
                if src is None:
                    raise ValueError(
                        f"Cannot apply derivation, src with src id : {recipe['src_id']} not found in registry."
                    )
                if src.get_dataset_dtype(src_array_name) == VeraDtype.UNKNOWN:
                    raise ValueError(
                        f"Cannot apply derivation to src with src id : {recipe['src_id']}, {src_array_name} not found in src."
                    )
                srcs = [src]
            else:
                srcs = [
                    src
                    for src in self._srcs.values()
                    if src.get_dataset_dtype(src_array_name) != VeraDtype.UNKNOWN
                ]
                self._auto_derivation.append(recipe)
            for src in srcs:
                src.add_new_derived_dataset(
                    source_array_name=src_array_name,
                    new_dataset_name=recipe["name"],
                    der_method=recipe["method"],
                    axes=VeraAxes[recipe["axes"]],
                )
        elif kind == "diff":
            ref_src = self.get(recipe.get("ref_src_id"))
            comp_src = self.get(recipe.get("comp_src_id"))
            if ref_src is None or comp_src is None:
                raise ValueError(
                    "Cannot apply diff, could not find reference source or comparison source in the registry."
                )
            ref_src.add_new_diff_dataset(
                recipe["ref_array"],
                comp_src,
                recipe["comp_array"],
                recipe["name"],
                recipe["interp_degree"],
                recipe["ref_scale"],
                recipe["comp_scale"],
                recipe["units"],
            )
        else:
            raise ValueError(f"Unknown recipe kind: {kind}")
        self._recipes.append(recipe)

    def replace_src(self, src_id: str, src: VeraDataSource):
        if src_id not in self._srcs:
            raise ValueError(f"Could not find {src_id} in registry")
        old = self._srcs[src_id]
        self._srcs[src_id] = src
        src.name = src_id
        old.close()
        self._compose_global_axial_mesh()
        self._replay_recipes(src_id)

    def _replay_recipes(self, src_id: str):
        src = self.get(src_id)
        if src is None:
            return src
        for recipe in self._recipes:
            all_sources = recipe.get("all_sources", False)
            derive_recipe_src_id = recipe.get("src_id")
            diff_recipe_src_id = recipe.get("ref_src_id")
            kind = recipe["kind"]
            if not (kind == "derive" and (derive_recipe_src_id == src_id or all_sources)) and not (
                kind == "diff" and diff_recipe_src_id == src_id
            ):
                continue
            try:
                if kind == "derive":
                    src.add_new_derived_dataset(
                        source_array_name=recipe["source_array"],
                        new_dataset_name=recipe["name"],
                        der_method=recipe["method"],
                        axes=VeraAxes[recipe["axes"]],
                    )
                elif kind == "diff":
                    src.add_new_diff_dataset(
                        recipe["ref_array"],
                        self.get(recipe["comp_src_id"]),
                        recipe["comp_array"],
                        recipe["name"],
                        recipe["interp_degree"],
                        recipe["ref_scale"],
                        recipe["comp_scale"],
                        recipe["units"],
                    )
            except Exception:
                pass

    def __contains__(self, src_id):
        return src_id in self._srcs
