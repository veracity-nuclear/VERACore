from trame.ui.html import DivLayout

OPTION = {
    "name": "empty",
    "label": "Undefined content",
    "icon": "mdi-help-circle-outline",
}


def option_for(view_id):
    return OPTION


def initialize(server, *args):
    with DivLayout(server, template_name="empty") as layout:
        layout.root.add_child("Some empty content...")
