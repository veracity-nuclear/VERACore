from pyvera.io.VERAout import VERAout
from vera_core.app.core.vera_out_file import VeraOutFile, Derivation

filename = "p9.h5"

# vera_out = VERAout(filename=filename)
# print(vera_out.axial_mesh)

# print(vera_out.states[0]["pin_powers"])

# import inspect

# # List all members of the class
# members = inspect.getmembers(vera_out)

# # Filter to get only data attributes (excluding methods)
# attributes = [m for m in members if not inspect.isroutine(m[1])]
# print(vars(vera_out).keys())
# pin_powers = vera_out.states[0]["pin_powers"]
# print(vera_out.Average(pin_powers))
v= VeraOutFile(filename)
print(v.active_state_index)
avg = v.add_new_derived_dataset("pin_powers", "new_name", derivation=Derivation.ASSEMBLY)
print(avg[24, :])