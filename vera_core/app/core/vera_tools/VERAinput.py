import os
from typing import Any, Optional, Dict
import xml.etree.ElementTree as ET
import json
import numpy as np
import h5py
from .geom_utils import num_units_from_hex_rings, reshape_hex_map

# Guard import for verain package
try:
    from verain.parser import InputParser

    _VERAIN_AVAILABLE = True
except ImportError:
    InputParser = None  # type: ignore
    _VERAIN_AVAILABLE = False


_type_conversion = {
    "int": int,
    "double": float,
    "string": str,
    "bool": bool,
    "Array(int)": lambda i: np.asarray(i[1:-1].split(","), dtype=int),
    "Array(double)": lambda x: np.asarray(x[1:-1].split(","), dtype=float),
    "Array(string)": lambda s: np.asarray(s[1:-1].split(","), dtype=str),
}


class VERAinput:
    """Class to read and write VERA input files.

    Parameters
    ----------
    filename : str
        Filename of the VERA input file
    filetype : str, optional
        Type of the input file (XML, H5, JSON). If not provided, the type is determined from the file extension
    print_status : bool, optional
        Print status messages
    **kwargs
        Additional keyword arguments
    """

    def __init__(self, filename: str, filetype: Optional[str] = None, print_status: bool = True, **kwargs):
        if not os.path.exists(filename):
            print(f"VERA Input file {filename} not found", exception=FileNotFoundError)
        if print_status:
            print(f"Reading VERA Input file {filename}")
        if filetype is None:
            filetype = filename.split(".")[-1]

        # Store filetype for export operations
        self.filetype = filetype.upper()

        if filetype.upper() == "XML":
            self.importXML(filename)
        elif filetype.upper() == "H5":
            self.importHDF5(filename, **kwargs)
        elif filetype.upper() == "JSON":
            self.importJSON(filename)
        elif filetype.upper() == "VERA" or filetype.upper() == "INP":
            self.importASCII(filename)
        else:
            raise TypeError(f"Cannot determine file type for: {filename}")

    @property
    def reactor_type(self):
        """Returns the reactor type from the XML input"""
        if "reactor_type" in self._data["CORE"]:
            return self._data["CORE"]["reactor_type"]
        return "PWR"

    def importXML(self, xmlfile: str) -> None:
        """Read VERA input file in XML format.

        Parameters
        ----------
        xmlfile : str
            Filename of the XML file
        """
        print("Reading XML file")
        xml = ET.parse(xmlfile)
        root = xml.getroot()
        self._data = self._walkTree(root)

    def _walkTree(self, root: ET.Element) -> dict:
        """Walk the XML tree and convert to dictionary.

        Parameters
        ----------
        root : ET.Element
            XML element

        Returns
        -------
        dict
            Dictionary representation of the XML tree
        """
        tree_dict = {}
        for child in root:
            if child.tag == "ParameterList":
                tree_dict[child.attrib["name"]] = self._walkTree(child)
            elif child.tag == "Parameter":
                try:
                    tree_dict[child.attrib["name"]] = _type_conversion[child.attrib["type"]](child.attrib["value"])
                except KeyError:
                    print("unknown type " + child.attrib["type"])
                    tree_dict[child.attrib["name"]] = str(child.attrib["value"])
            else:
                assert False
        return tree_dict

    def importJSON(self, jsonfile: str) -> None:
        """Read VERA input file in JSON format.

        Parameters
        ----------
        jsonfile : str
            Filename of the JSON file
        """
        print("Reading JSON file")
        with open(jsonfile, "r", encoding="utf-8") as infile:
            self._data = json.load(infile)
        self._data = self._list2numpy(self._data)

    def importASCII(self, asciifile: str) -> None:
        """Read VERA input file in ASCII format.

        Parameters
        ----------
        asciifile : str
            Filename of the ASCII VERAIn file

        Raises
        ------
        ImportError
            If verain package is not available
        """
        self.check_verain_available()
        print("Reading ASCII VERAIn file")
        assert InputParser is not None  # Type hint for Pylance
        parser = InputParser()
        self._data = parser.parse_file(asciifile)
        parser.validate_input(self._data)
        errors = parser.get_validation_errors()
        if errors:
            print(
                f"Found {len(errors)} validation errors in VERAIn file: {asciifile}\n"
                f"\n".join(parser.get_validation_errors())
            )

    def _list2numpy(self, mydict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert lists to numpy arrays in a dictionary.

        Parameters
        ----------
        mydict : dict
            Dictionary to convert

        Returns
        -------
        np.ndarray
            Numpy array representation of the dictionary
        """
        for k, v in mydict.items():
            if isinstance(v, dict):
                mydict[k] = self._list2numpy(v)
            elif isinstance(v, list):
                mydict[k] = np.array(v)
        return mydict

    def exportJSON(self, jsonfile: str) -> None:
        """Write VERA input to JSON file.

        Parameters
        ----------
        jsonfile : str
            Filename of the JSON file
        """

        class NumpyEncoder(json.JSONEncoder):
            """Special json encoder for numpy types"""

            # pylint wanted to rename obj to o. I think obj is more descriptive
            # pylint:disable=arguments-renamed
            def default(self, o):
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.floating):
                    return float(o)
                if isinstance(o, np.bool_):
                    return bool(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return json.JSONEncoder.default(self, o)

        print(f"Writing VERA Input to JSON file: {jsonfile}")
        if os.path.exists(jsonfile):
            print(f"JSON file {jsonfile} already exists (overwriting)")
        with open(jsonfile, "w", encoding="") as outfile:
            json.dump(self._data, outfile, cls=NumpyEncoder)

    def importHDF5(self, h5file: str, path: str = "input") -> None:
        """Read VERA input file in HDF5 format.

        Parameters
        ----------
        h5file : str
            Filename of the HDF5 file
        path : str, optional
            Path in the HDF5 file to read
        """
        print("Reading HDF5 file")
        with h5py.File(h5file, "r") as h5f:
            h5_obj = h5f[path]
            if isinstance(h5_obj, h5py.Group):
                self._data = self._walkGroup(h5_obj)
            else:
                raise TypeError(f"Expected HDF5 Group at path '{path}', got {type(h5_obj).__name__}")

    def _walkGroup(self, group: h5py.Group) -> dict:
        """Walk the HDF5 group and convert to dictionary.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group

        Returns
        -------
        h5py.Group
            Dictionary representation of the HDF5 group
        """
        tree_dict = {}
        for k, v in group.items():
            if isinstance(v, h5py.Group):
                tree_dict[k] = self._walkGroup(v)
            else:
                data = v[()]
                if isinstance(data, np.ndarray):
                    if data.dtype.kind == "S":
                        data = np.array(data, dtype=str)
                tree_dict[k] = data
        return tree_dict

    def exportHDF5(self, h5file: str, path: str = "input") -> None:
        """Write VERA input to HDF5 file.

        Parameters
        ----------
        h5file : str
            Filename of the HDF5 file
        path : str, optional
            Path in the HDF5 file to write
        """
        print(f"Writing VERA Input to HDF5 file: {h5file}")
        if os.path.exists(h5file):
            print(f"HDF5 file {h5file} already exists (overwriting)")
        with h5py.File(h5file, "w") as h5f:
            self._dumpHDF5(self._data, h5f, path)

    def _dumpHDF5(self, mydict: dict, h5f: h5py.File, path: str) -> None:
        """Write dictionary to HDF5 file.

        Parameters
        ----------
        mydict : dict
            Dictionary to write
        h5f : h5py.File
            HDF5 file
        path : str
            Path in the HDF5 file to write
        """
        for k, v in mydict.items():
            if isinstance(v, dict):
                self._dumpHDF5(v, h5f, path + "/" + k)
            elif isinstance(v, np.ndarray):
                try:
                    h5f[path + "/" + k] = v
                except (TypeError, ValueError):
                    h5f[path + "/" + k] = [s.encode("ascii", "ignore") for s in v.flat]
            else:
                h5f[path + "/" + k] = v

    def convert_maps(self):
        """Convert VERA input maps to 2D arrays."""
        # pylint thinks that self["CORE/shape"] is a dictionary which has no member 'reshape'
        # pylint:disable=no-member
        print("Converting VERA input maps to 2D")
        if "shape" in self["CORE"]:
            self._data["CORE"]["shape"] = self["CORE/shape"].reshape((self["CORE/core_size"], self["CORE/core_size"]))
        elif "hex_rings" in self["CORE"]:
            self._data["CORE"]["shape"] = reshape_hex_map(
                np.array([1 for i in range(num_units_from_hex_rings(self["CORE/hex_rings"]))])
            )
        self._reshape_coremap("assm_map")
        self._reshape_coremap("crd_map")
        self._reshape_coremap("crd_bank")
        self._reshape_coremap("insert_map")
        self._reshape_coremap("det_map")
        self._reshape_sized_maps(self._data)

    def exportASCII(self, asciifile: str) -> None:
        """Write VERA input to ASCII file.

        Parameters
        ----------
        asciifile : str
            Filename of the ASCII VERAIn file

        Raises
        ------
        ImportError
            If verain package is not available
        TypeError
            If the original file format is not VERA or INP
        """
        VERAinput.check_verain_available()

        if self.filetype not in ["VERA", "INP"]:
            raise TypeError(f"exportASCII only supported for VERA/INP filetypes, got {self.filetype}")

        print(f"Writing VERA Input to ASCII file: {asciifile}")

        assert InputParser is not None  # Type hint for Pylance
        parser = InputParser()
        parser.write_verain_file(asciifile, self._data)

    def _reshape_sized_maps(self, d: dict, path: str = "", npin: int = 0, hex_rings: int = 0) -> dict:
        """Reshape sized maps to 2D arrays.

        Parameters
        ----------
        d : dict
            Dictionary to reshape
        path : str, optional
            Path in the dictionary
        npin : int, optional
            Number of pins

        Returns
        -------
        dict
            Reshaped dictionary
        """
        _reshape_keys = ["cell_map", "xcentoffset", "ycentoffset"]
        if "num_pins" in d.keys():
            npin = d["num_pins"]
        elif "hex_rings" in d.keys():
            hex_rings = d["hex_rings"]
        for k, v in d.items():
            if isinstance(v, dict):
                v = self._reshape_sized_maps(v, path + "/" + k, npin=npin, hex_rings=hex_rings)
            elif isinstance(v, np.ndarray) and k in _reshape_keys:
                if npin:
                    d[k] = v.reshape((npin, npin))
                elif hex_rings:
                    d[k] = reshape_hex_map(v, v.dtype)
        return d

    def _reshape_coremap(self, mapname: str) -> None:
        """Reshape core maps to 2D arrays.

        Parameters
        ----------
        mapname : str
            Name of the map
        """
        if mapname not in self["CORE"]:
            return
        oldmap = self["CORE"][mapname][()]
        core_shape = self["CORE/shape"]

        # pylint thinks that self["CORE/shape"] is a dictionary which has no member 'shape'
        # pylint:disable=no-member
        if oldmap.shape == core_shape.shape:
            return
        if len(oldmap) != np.sum(core_shape):
            newmap = oldmap.reshape(core_shape.shape)
        else:
            newmap = np.empty_like(core_shape, dtype=oldmap.dtype)
            ipos = 0
            for row, map_row in enumerate(self["CORE/shape"]):
                for col, val in enumerate(map_row):
                    if val == 1:
                        newmap[row, col] = oldmap[ipos]
                        ipos += 1
                    else:
                        newmap[row, col] = ""
        self._data["CORE"][mapname] = newmap

    def __contains__(self, itemkey: str) -> bool:
        """Check if item is in the VERA input.

        Parameters
        ----------
        itemkey : str
            Key to check

        Returns
        -------
        bool
            True if the key is in the VERA input, False otherwise
        """
        mydict = self._data
        try:
            for k in itemkey.split("/"):
                mydict = mydict[k]
            return True
        except (KeyError, TypeError):
            return False

    def __getitem__(self, itemkey: str) -> Any:
        """Get item from the VERA input.

        Parameters
        ----------
        itemkey : str
            Key to get

        Returns
        -------
        Any
            Value of the key
        """
        mydict = self._data
        try:
            for k in itemkey.split("/"):
                mydict = mydict[k]
            return mydict
        except (KeyError, TypeError) as exc:
            raise exc.__class__(f"Could not find key '{itemkey}' in the VERA input") from exc

    def get(self, itemkey: str, default: Optional[Any] = None) -> Any:
        """Get item from the VERA input with default value.

        Parameters
        ----------
        itemkey : str
            Key to get
        default : Any, optional
            Default value to return if key is not found

        Returns
        -------
        Any
            Value of the key or default value if key is not found
        """
        try:
            return self[itemkey]
        except (KeyError, TypeError):
            return default

    def keys(self) -> list:
        """Get keys of the VERA input."""
        return list(self._data.keys())

    def items(self):
        """Get items of the VERA input."""
        return self._data.items()

    # pylint: disable=attribute-defined-outside-init
    def __iter__(self):
        """Iterate over the VERA input."""
        return iter(self._data)

    def get_dict(self) -> dict:
        """Get the VERA input as a dictionary."""
        return self._data

    # pylint: disable=attribute-defined-outside-init
    def set_dict(self, data: dict) -> None:
        """Set the VERA input from a dictionary."""
        self._data = data

    def export(self, filename: str, filetype: Optional[str] = None) -> None:
        """Export VERA input to file.

        Parameters
        ----------
        filename : str
            Output filename
        filetype : str, optional
            Output file type. If not provided, uses the filetype from initialization.
        """
        if filetype is None:
            filetype = self.filetype

        filetype = filetype.upper()

        if filetype in ["VERA", "INP"]:
            self.exportASCII(filename)
        elif filetype == "JSON":
            self.exportJSON(filename)
        elif filetype == "H5":
            self.exportHDF5(filename)
        else:
            raise TypeError(f"Export not supported for filetype: {filetype}")

    @classmethod
    def is_ascii_supported(cls) -> bool:
        """Check if ASCII VERA input file support is available.

        Returns
        -------
        bool
            True if verain package is available and ASCII files can be processed, False otherwise.
        """
        return _VERAIN_AVAILABLE

    @classmethod
    def check_verain_available(cls) -> None:
        """Check if verain is available and raise exception if not."""
        if not _VERAIN_AVAILABLE:
            raise ImportError(
                "The 'verain' package is required for ASCII VERA input file support but is not available. "
                "Please install 'verain' or use XML, JSON, or H5 formats instead."
            )
