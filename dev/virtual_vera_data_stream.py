import zmq, h5py
import numpy as np
import msgpack
import time
import msgpack_numpy as m
import argparse

parser = argparse.ArgumentParser(description="A script that mimics a possible vera data stream, pulls data from h5 file")

parser.add_argument("filename", help="The name of the file to send as a stream")
parser.add_argument("port", help="The port to stream/publish data to")

args = parser.parse_args()

m.patch()

context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind(f"tcp://0.0.0.0:{args.port}")

with h5py.File(args.filename, "r") as f:
    core = f["CORE"]
    axial_mesh = core["axial_mesh"][()]
    core_map = core["core_map"][()]
    core_sym = core["core_sym"][()]
    pin_volumes = core["pin_volumes"][()]
    core_data = {"axial_mesh" : axial_mesh, "core_sym" : core_sym,
                 "core_map" : core_map, "pin_volumes" : pin_volumes}
    for state_key in [key for key in f.keys() if key.startswith("STATE_")]:
        state = f[state_key]
        pin_powers = state["pin_powers"]
        core_shape = np.shape(pin_powers)
        full_core_datasets = {}
        scalar_datasets = {}
        for dataset_name in state.keys():
            dataset_shape = np.shape(state[dataset_name])
            if dataset_shape == core_shape:
                full_core_datasets.update({dataset_name: state[dataset_name][()]})
            if dataset_shape in [(1,), ()]:
                scalar_datasets.update({dataset_name: state[dataset_name][()]})
        payload = {"core" : core_data,
                   "state" : state_key,
                   "data" : {"full_core_datasets" : full_core_datasets,
                   "scalar_datasets" : scalar_datasets}}
        publisher.send(msgpack.packb(payload))
        print(f"Sent {state_key}")
        time.sleep(1)

