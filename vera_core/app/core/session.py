from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from trame_server.core import State
from . import VeraDataRegistry


SESSION_GLOBAL_KEYS = [
    "selected_assembly_ij", 
    "selected_layer", 
    "selected_time", 
    "selected_i", 
    "selected_j", 
    "selected_surface", 
    "dark_mode",
    "max_layer", 
    "thresholds", 
    ]

SESSION_VERSION = 1

@dataclass
class ViewSession:
    view_id: int
    option: dict
    layout: dict | None
    selected_src_id: str
    selected_array: str
    selected_label: str
    multi_selected: list
    multi_label: str
    locked: bool

@dataclass
class Session:
    version: int
    file_paths: dict[str, str]
    core_overrides : dict[str, int]
    default_src_id: str | None
    globals: dict[str, object]
    views: list[ViewSession]
    recipes: list[dict] = field(default_factory=list)

def derive_recipe(src_id, source_array, name, method, axes, use_factors, exclude_non_fuel_rods) -> dict:
    return {"kind": "derive", 
            "src_id": src_id, 
            "source_array": source_array,
            "name": name, 
            "method": method, 
            "axes": axes, 
            "use_factor" : use_factors, 
            "exclude_fuel_rods" : exclude_non_fuel_rods,
    }

def diff_recipe(ref_src_id, ref_array, comp_src_id, comp_array, name, interp_degree, ref_scale, comp_scale, units) -> dict:
    return {"kind": "diff", 
            "ref_src_id": ref_src_id, 
            "ref_array": ref_array,
            "comp_src_id": comp_src_id, 
            "comp_array": comp_array,
            "name": name, 
            "interp_degree": interp_degree,
            "ref_scale" : ref_scale,
            "comp_scale" : comp_scale,
            "units" : units,
    }

def recipe_sources(recipe) -> set:
    """The src_ids a recipe depends on — used for load validation and save filtering."""
    if recipe["kind"] == "derive":
        return {recipe["src_id"]}
    if recipe["kind"] == "diff":
        return {recipe["ref_src_id"], recipe["comp_src_id"]}
    return set()

def build_session(state : State, registry : "VeraDataRegistry", all_view_ids : list) -> Session:
    placed = state.grid_layout or []
    views = []
    vid_layouts = {entry["i"] : entry for entry in placed}
    for vid in all_view_ids:
        option = state[f"grid_view_{vid}"]
        views.append(ViewSession(
            view_id=vid,
            option=option,
            layout=vid_layouts.get(vid, None),
            selected_src_id=state[f"selected_src_id_{vid}"],
            selected_array=state[f"selected_array_{vid}"],
            selected_label=state[f"selected_label_{vid}"],
            multi_selected=list(state[f"multi_selected_{vid}"]),
            multi_label=state[f"multi_label_{vid}"],
            locked=bool(state[f"locked_{vid}"]),
        ))

    file_paths = registry.all_sources()
    globals_ = {k: state[k] for k in SESSION_GLOBAL_KEYS if state.has(k)}
    recipes = state.derived_recipes if state.has("derived_recipes") else []
    all_recipes = state.recipes if state.has("recipes") else []
    recipes = [r for r in all_recipes if recipe_sources(r) <= set(file_paths)]
    core_overrides = state["core_overrides"]

    session = Session(version=SESSION_VERSION, 
                      file_paths=file_paths, 
                      core_overrides=core_overrides, 
                      default_src_id=registry.default_src_id,
                      globals=globals_, 
                      views=views, 
                      recipes=recipes,
            )
    return session

def save_session(state, registry : VeraDataRegistry, all_view_ids : list, out_path: str):
    session = build_session(state, registry, all_view_ids)
    Path(out_path).write_text(json.dumps(asdict(session), indent=2))
