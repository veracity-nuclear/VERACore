from trame.ui.html import DivLayout

OPTION = {
    "name": "empty",
    "label": "Undefined content",
    "icon": "mdi-help-circle-outline",
}
def option_for(view_id):
    return OPTION

def initialize(server, *args):
    state = server.state

    # if OPTION not in state.grid_options:
    #     state.grid_options.append(OPTION)

    with DivLayout(server, template_name="empty") as layout:
        layout.root.add_child("Some empty content...")
