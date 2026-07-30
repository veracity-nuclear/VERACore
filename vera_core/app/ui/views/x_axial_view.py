from . import _axial_view_base as base


def option_for(view_id):
    return base.option_for(view_id, "x")


def initialize(server, registry, view_id):
    base.build_axial_view(server, registry, view_id, "x")
