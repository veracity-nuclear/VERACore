import os
from enum import Enum
from typing import Any

import h5py
import numpy as np

from . import VERAinput as verain


class bc(Enum):
    """Boundary condition type enumeration."""

    none = 0
    rotational = 1
    mirror = 2

    def __str__(self) -> str:
        return ["None", "Rotational", "mirror"][self.value]


class VERAout:
    """Class to represent a VERA output file.

    Parameters
    ----------
    filename : str
        The path to the VERA output file.
    allow_dataset_caching : bool, optional
        Flag to allow caching of datasets, by default False
    """

    def __init__(self, filename: str, allow_dataset_caching: bool = False):
        self._cache_datasets = allow_dataset_caching
        self.cached_datasets = {}

        print(f"Reading VERA Output file {filename:s}")
        if not os.path.isfile(filename):  # Check if file exists
            print(f"The file {filename:s} does not exist.")

        self.filename = filename  # save the filename
        self.h5f = h5py.File(filename, "r", locking=False)  # open HDF5 file for reading

        # Get VERA input from the VERA out file
        if "/INPUT/CASEID" in self.h5f:
            self._verain = verain.VERAinput(filename, "H5", path="/INPUT/CASEID", print_status=False)
        else:
            print(f"XML input path not in {filename:s}")
            self._verain = None

        # Get the geometry information from CORE
        self.core_map = self.GetData("/CORE/core_map")
        self.axial_mesh = self.GetData("/CORE/axial_mesh")
        self.core_sym = self.GetData("/CORE/core_sym")
        self.core_sym_type = bc.rotational
        if self.Exists("/CORE/bc_sym"):
            bc_sym = "".join(self.GetData("/CORE/bc_sym").decode("utf-8"))
            if bc_sym == "rotational":
                self.core_sym_type = bc.rotational
            elif bc_sym == "mirror":
                self.core_sym_type = bc.mirror
            else:
                print(f"Unknown boundary condition type: {bc_sym:s}")
                self.core_sym_type = bc.none

        # get the core dimensions
        if self.Exists("/CORE/size"):
            self.size = self.GetData("/CORE/size")
        else:
            self.size = self.core_map.shape[0]

        # number of pins across an assembly
        if self.Exists("/CORE/num_pins"):
            self.num_pins = self.GetData("/CORE/num_pins")
        else:
            junk = self.__getsampledata__()
            self.num_pins = junk.shape[0]

        # number of assemblies in the calculated geometry
        if self.Exists("/CORE/num_assys"):
            self.num_assys = self.GetData("/CORE/num_assys")
        else:
            junk = self.__getsampledata__()
            self.num_assys = junk.shape[3]

        # number of axial planes
        if self.Exists("/CORE/num_axials"):
            self.num_axials = self.GetData("/CORE/num_axials")
        else:
            self.num_axials = len(self.axial_mesh) - 1

        # Axial mesh heights
        self.axial_heights = np.zeros((self.num_axials))
        for i in range(self.num_axials):
            self.axial_heights[i] = self.axial_mesh[i + 1] - self.axial_mesh[i]

        # number of channels
        self.num_channels = self.num_pins + 1

        # set assembly loop bounds (assumes x/y symmetry)
        self.alo = 0
        self.ahi = self.size
        if self.core_sym == 4:
            if self.size % 2 == 0:  # even
                self.alo = int(self.size / 2)
            else:  # odd
                self.alo = int((self.size - 1) / 2)

        # set pin loop bounds by assembly
        self.pxlo = np.empty([self.num_assys], dtype=int)  # create arrays for loop bounds
        self.pylo = np.empty([self.num_assys], dtype=int)  # in case of quarter symmetry
        self.pxlo.fill(0)  # full symmetry
        self.pylo.fill(0)
        self.pxhi = self.num_pins
        self.pyhi = self.num_pins
        if self.core_sym == 4 and self.size % 2 == 1:  # if quarter symmetry and odd assemblies
            for i in range(self.alo, self.ahi):
                if self.num_pins % 2 == 1:  # if odd
                    self.pxlo[self.core_map[i, self.alo] - 1] = (self.num_pins - 1) / 2
                    self.pylo[self.core_map[self.alo, i] - 1] = (self.num_pins - 1) / 2
                else:  # even
                    self.pxlo[self.core_map[i, self.alo] - 1] = self.num_pins / 2
                    self.pylo[self.core_map[self.alo, i] - 1] = self.num_pins / 2

        # set channel loop bounds by assembly
        self.cxlo = np.empty([self.num_assys], dtype=int)  # create arrays for loop bounds
        self.cylo = np.empty([self.num_assys], dtype=int)  # in case of quarter symmetry
        self.cxlo.fill(0)  # full symmetry
        self.cylo.fill(0)
        self.cxhi = self.num_channels
        self.cyhi = self.num_channels
        if self.core_sym == 4 and self.size % 2 == 1:  # if quarter symmetry and odd assemblies
            for i in range(self.alo, self.ahi):
                if self.num_channels % 2 == 1:  # if odd
                    self.cxlo[self.core_map[i, self.alo] - 1] = (self.num_channels - 1) / 2
                    self.cylo[self.core_map[self.alo, i] - 1] = (self.num_channels - 1) / 2
                else:  # even
                    self.cxlo[self.core_map[i, self.alo] - 1] = self.num_channels / 2
                    self.cylo[self.core_map[self.alo, i] - 1] = self.num_channels / 2

        # fix pin volumes if exists
        # TODO: getPinIndex() assumes the data is aleady unfolded for the assemblies on the line
        #  of symmetry, so we need to keep the pin_volumes as is but multiply the pins on the axis of symmetry by 2.

        if self.Exists("/CORE/pin_volumes"):
            self.pin_volumes = self.GetData("/CORE/pin_volumes")
            if (
                self.core_sym == 4 and self.size % 2 == 1 and self.num_pins % 2 == 1
            ):  # if quarter symmetry and odd assemblies and pins
                for i in range(self.alo, self.ahi):
                    self.pin_volumes[
                        self.pylo[self.core_map[self.alo, i] - 1], :, :, self.core_map[self.alo, i] - 1
                    ] *= 2  # double the volumes on the axis
                    self.pin_volumes[:, self.pxlo[self.core_map[i, self.alo] - 1], :, self.core_map[i, self.alo] - 1] *= 2
        else:
            self.pin_volumes = None

        # calculate the pin_factors and channel factors
        self.__calculatepinfactors__()
        self.__calculatechannelfactors__()

        # load initial mass
        if self.Exists("/CORE/initial_mass"):
            self.initial_mass = self.h5f["CORE"]["initial_mass"][()]
            np.place(self.initial_mass, self.pin_factors == 0, 0.0)

        # get core labels if exists (disables no-member b/c pylint thinks this is an h5py.Group when it is np.ndarray)
        if self.Exists("/CORE/xlabel"):
            self.xlabel = self.h5f["/CORE/xlabel"][()].tolist()  # pylint:disable=no-member
            self.xlabel = [x.decode("utf-8").strip() for x in self.xlabel]
        else:
            self.xlabel = ["R", "P", "N", "M", "L", "K", "J", "H", "G", "F", "E", "D", "C", "B", "A"]

        if self.Exists("/CORE/ylabel"):
            self.ylabel = self.h5f["/CORE/ylabel"][()].tolist()  # pylint:disable=no-member
            self.ylabel = [y.decode("utf-8").strip() for y in self.ylabel]
        else:
            self.ylabel = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]

        # get detector map is exists
        self.__getdetectormap__()

        # load States
        self.states = []
        for s in self.h5f:
            if s.startswith("STATE_"):
                self.states.append(Statepoint(self, s))

    def __del__(self):
        try:
            self.h5f.close()
            print("HDF5 File Closed!")
        except AttributeError as e:
            print(f"Failed to close HDF5 file: {e}")

    def __getsampledata__(self):
        try:
            return self.h5f["CORE"]["pin_factors"]
        except KeyError:
            print("Key 'CORE/pin_factors' not found.")

        try:
            return self.h5f["STATE_0001"]["pin_powers"]
        except KeyError:
            print("Key 'STATE_0001/pin_powers' not found.")

        try:
            return self.h5f["CORE"]["initial_mass"]
        except KeyError:
            print("Key 'CORE/initial_mass' not found.")

        print("Cannot determine the geometry")
        return None

    def __calculatepinfactors__(self):
        try:
            self.pin_factors = self.h5f["CORE"]["pin_factors"][()]
        # want to catch any exception here
        # pylint: disable=broad-except
        except Exception:
            junk = self.__getsampledata__()
            self.pin_factors = np.zeros(junk.shape)  # create array for factors
            if self.core_sym == 1:  # full symmetry
                np.place(self.pin_factors, junk[:] > 0.0, 1.0)  # put 1.0 anywhere there is a value
            else:
                for l in range(self.num_assys):  # same except only in valid locations for the geometry
                    np.place(
                        self.pin_factors[self.pylo[l] : self.pyhi, self.pxlo[l] : self.pxhi, :, l],
                        junk[self.pylo[l] : self.pyhi, self.pxlo[l] : self.pxhi, :, l] > 0.0,
                        1.0,
                    )

                if self.size % 2 == 1 and self.num_pins % 2 == 1:  # if odd pins and odd assemblies
                    for ai in range(self.alo, self.ahi):  # for quarter symmetry
                        l = self.core_map[self.alo, ai] - 1  # cut the pin weights by
                        if l < 0:
                            continue  # half on the line of symmetry
                        self.pin_factors[self.pylo[l], :, :, l] *= 0.5

                    for aj in range(self.alo, self.ahi):  # for quarter symmetry
                        l = self.core_map[aj, self.alo] - 1  # cut the pin weights by
                        if l < 0:
                            continue  # half on the line of symmetry
                        self.pin_factors[:, self.pxlo[l], :, l] *= 0.5

            for k in range(self.num_axials):  # multiply weights by axial mesh size
                self.pin_factors[:, :, k, :] *= self.axial_mesh[k + 1] - self.axial_mesh[k]

        # calculate derived weights
        self.axial_factors = np.sum(self.pin_factors, axis=(0, 1, 3))
        self.radial_factors = np.sum(self.pin_factors, axis=2)
        self.assy_factors = np.sum(self.pin_factors, axis=(0, 1))
        self.radial_assy_factors = np.sum(self.pin_factors, axis=(0, 1, 2))

        # calculate the axial mesh size
        self.axial_mesh_size = self.axial_mesh[1:] - self.axial_mesh[:-1]

    def __calculatepinfactorsfornodes__(self):
        junk = self.__getsampledata__()
        result = np.zeros(junk.shape)  # create array for factors
        if self.core_sym == 1:  # full symmetry
            np.place(result, junk[:] > 0.0, 1.0)  # put 1.0 anywhere there is a value
        else:
            for l in range(self.num_assys):  # same except only in valid locations for the geometry
                np.place(
                    result[self.pylo[l] : self.pyhi, self.pxlo[l] : self.pxhi, :, l],
                    junk[self.pylo[l] : self.pyhi, self.pxlo[l] : self.pxhi, :, l] > 0.0,
                    1.0,
                )

        for k in range(self.num_axials):  # multiply weights by axial mesh size
            result[:, :, k, :] *= self.axial_mesh[k + 1] - self.axial_mesh[k]

        return result

    def __calculatechannelfactors__(self):
        self.channel_factors = np.zeros(
            (self.num_channels, self.num_channels, self.num_axials, self.num_assys)
        )  # create array for factors
        if self.core_sym == 0:
            self.channel_factors = 1.0
        else:
            for l in range(self.num_assys):  # same except only in valid locations for the geometry
                self.channel_factors[self.cylo[l] : self.cyhi, self.cxlo[l] : self.cxhi, :, l] = 1.0

            if self.size % 2 == 1 and self.num_channels % 2 == 1:  # if odd channels and odd assemblies
                for ai in range(self.alo, self.ahi):  # for quarter symmetry
                    l = self.core_map[self.alo, ai] - 1  # cut the channel weights by
                    if l < 0:
                        continue  # half on the line of symmetry
                    self.channel_factors[self.cylo[l], :, :, l] *= 0.5

                for aj in range(self.alo, self.ahi):  # for quarter symmetry
                    l = self.core_map[aj, self.alo] - 1  # cut the channel weights by
                    if l < 0:
                        continue  # half on the line of symmetry
                    self.channel_factors[:, self.pxlo[l], :, l] *= 0.5

        if self.Exists("/CORE/channel_cell_height"):
            cch = self.h5f["/CORE/channel_cell_height"][()]
            for k in range(self.num_axials):
                self.channel_factors[:, :, k, :] *= cch[k]  # maybe a better way to do this

        if self.Exists("/CORE/channel_flow_area"):
            cfa = self.h5f["/CORE/channel_flow_area"][()]

            # need to fix up the current flow area by adjusting the values on the assembly periphery
            for aj in range(self.alo, self.ahi):
                for ai in range(self.alo, self.ahi):
                    l = self.core_map[aj, ai] - 1
                    if l < 0:
                        continue
                    for i in range(self.num_channels):
                        cfa[0, i, l] *= 0.5  # top row
                        cfa[i, 0, l] *= 0.5  # left row
                    if ai < self.size - 1 and self.core_map[aj, ai + 1] > 0:
                        for i in range(self.num_channels):
                            cfa[i, -1, l] *= 0.5  # right side
                    if aj < self.size - 1 and self.core_map[aj + 1, ai] > 0:
                        for i in range(self.num_channels):
                            cfa[-1, i, l] *= 0.5  # bottom row
            # ------------------------------------------------------------------------------------
            for k in range(self.num_axials):
                self.channel_factors[:, :, k, :] *= cfa[:, :, :]  # maybe a better way to do this

        # calculate derived weights for channels
        self.radial_channel_factors = np.sum(self.channel_factors, axis=2)

    def __calculateaxialoffset__(self, axial_data):
        pbot = 0.0
        ptop = 0.0

        baf = 0
        taf = -1
        while self.axial_factors[baf] == 0:
            baf += 1
        while self.axial_factors[taf] == 0:
            taf -= 1

        hmid = (self.axial_mesh[baf] + self.axial_mesh[taf]) * 0.5

        for k in range(self.num_axials):
            if self.axial_mesh[k + 1] <= hmid:
                pbot += axial_data[k] * self.axial_mesh_size[k]
            elif self.axial_mesh[k] < hmid:
                pbot += axial_data[k] * (hmid - self.axial_mesh[k])
                ptop += axial_data[k] * (self.axial_mesh[k + 1] - hmid)
            else:
                ptop += axial_data[k] * self.axial_mesh_size[k]

        return (ptop - pbot) / (ptop + pbot) * 100.0

    def __getdetectormap__(self):
        if self.Exists("/CORE/detector_map"):
            self.detector_map = self.h5f["/CORE/detector_map"][()]
        elif self.Exists("/INPUT/CASEID/CORE/det_map"):
            det_map = self.GetDataIfExists("/INPUT/CASEID/CORE/det_map")
            if det_map is not None:
                self.detector_map = np.zeros(self.core_map.shape, dtype=np.int)
                n = -1
                cnt = 0
                for row in range(self.size):
                    for col in range(self.size):
                        if self.core_map[row, col] > 0:
                            n += 1
                            if not det_map[n] == "-":
                                cnt += 1
                                self.detector_map[row, col] = cnt

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.states[key]
        return self.GetDataIfExists(key, None, True)

    def __setitem__(self, key, item):
        return self.SetData(key, item)

    def getStateNames(self) -> list[str]:
        """Returns the state names in the VERA output file.

        Returns
        -------
        list[str]
            The list of state names
        """
        names = []
        for s in self.h5f:
            if s.startswith("STATE_"):
                names.append(s)
        return names

    def getAssemblyIndex(self, assembly_loc: tuple[int, int]) -> int:
        """Returns the assembly index

        Parameters
        ----------
        assembly_loc : Tuple[int, int]
            The assembly location (row, col)

        Returns
        -------
        asy_index : int
            The assembly 1D index, indexed left to right then top to bottom
        """
        return self.core_map[assembly_loc[0], assembly_loc[1]]  # VERAOut is indexes (row, col)

    def getPinIndex(self, assembly_loc: tuple[int, int], pin_loc: tuple[int, int]) -> tuple[int, int]:
        """Returns the assembly index

        Parameters
        ----------
        assembly_loc : tuple[int, int]
            The assembly location (row, col)
        pin_loc : tuple[int, int]
            The pin location in the assembly (row, col)

        Returns
        -------
        pin_index : tuple[int, int]
            The pin location (row, col)
        """
        # No translation needed for full core VERAout files
        if self.core_sym_type == bc.none or self.core_sym == 1:
            return pin_loc

        row, col = assembly_loc
        rowp, colp = pin_loc
        px, py = 0, 0
        isOdd = self.size % 2 == 1  # even/odd core size changes quarter core symmetry operations
        assert self.core_sym == 4

        # All assemblies on the symmetry line are already unfolded correctly
        if col >= self.alo:
            if row >= self.alo:
                # in SE quadrant, process normally
                py, px = pin_loc
            else:
                if col == self.alo and isOdd:
                    # on W symmetry line
                    px = self.num_pins - 1 - colp
                    py = self.num_pins - 1 - rowp
                else:
                    # in NE quadrant
                    if self.core_sym_type == bc.rotational:
                        px = self.num_pins - 1 - rowp
                        py = colp
                    elif self.core_sym_type == bc.mirror:
                        px = colp
                        py = self.num_pins - 1 - rowp
        else:
            if row > self.alo:
                # in SW quadrant
                if self.core_sym_type == bc.rotational:
                    px = rowp
                    py = self.num_pins - 1 - colp
                elif self.core_sym_type == bc.mirror:
                    px = self.num_pins - 1 - colp
                    py = rowp
            elif row == self.alo and isOdd:
                # is on symmetry line
                px = self.num_pins - 1 - colp
                py = self.num_pins - 1 - rowp
            else:
                # in NW quadrant
                px = self.num_pins - 1 - colp
                py = self.num_pins - 1 - rowp
        return py, px

    def Average(self, data: np.ndarray) -> float:
        """Calculate the average of the data.

        Parameters
        ----------
        data : np.ndarray
            The data to average

        Returns
        -------
        float
            The average of the data
        """
        if len(data.shape) == 4:
            if (
                data.shape[0] == self.num_pins
                and data.shape[1] == self.num_pins
                and data.shape[2] == self.num_axials
                and data.shape[3] == self.num_assys
            ):
                return np.average(data, axis=None, weights=self.pin_factors)
            if (
                data.shape[0] == self.num_channels
                and data.shape[1] == self.num_channels
                and data.shape[2] == self.num_axials
                and data.shape[3] == self.num_assys
            ):
                return np.average(data, axis=None, weights=self.channel_factors)
            return np.average(data)
        if len(data.shape) == 3:
            if data.shape[0] == self.num_pins and data.shape[1] == self.num_pins and data.shape[2] == self.num_assys:
                return np.average(data, axis=None, weights=self.radial_factors)
            if data.shape[0] == self.num_channels and data.shape[1] == self.num_channels and data.shape[2] == self.num_assys:
                return np.average(data, axis=None, weights=self.radial_channel_factors)
            return np.average(data)
        if len(data.shape) == 2:
            if data.shape[0] == self.num_axials and data.shape[1] == self.num_assys:
                return np.average(data, axis=None, weights=self.assy_factors)
            return np.average(data)
        if len(data.shape) == 1 and data.shape[0] == self.num_axials:
            return np.average(data, axis=None, weights=self.axial_factors)
        if len(data.shape) == 1 and data.shape[0] == self.num_assys:
            return np.average(data, axis=None, weights=self.radial_assy_factors)
        return np.average(data)

    def Axial(self, data: np.ndarray) -> np.ndarray:
        """Calculate the axial average of the data.

        Parameters
        ----------
        data : np.ndarray
            The data to average

        Returns
        -------
        np.ndarray
            The axial average of the data
        """
        if len(data.shape) == 4:
            if (
                data.shape[0] == self.num_pins
                and data.shape[1] == self.num_pins
                and data.shape[2] == self.num_axials
                and data.shape[3] == self.num_assys
            ):
                olderr = np.seterr(invalid="ignore")
                ax = np.sum(data * self.pin_factors, axis=(0, 1, 3)) / self.axial_factors
                np.seterr(**olderr)
                np.place(ax, np.isnan(ax[:]), 0.0)  # put 0.0 anywhere there is nan
                return ax
            if (
                data.shape[0] == self.num_channels
                and data.shape[1] == self.num_channels
                and data.shape[2] == self.num_axials
                and data.shape[3] == self.num_assys
            ):
                olderr = np.seterr(invalid="ignore")
                ax = np.sum(data * self.channel_factors, axis=(0, 1, 3)) / self.axial_factors
                np.seterr(**olderr)
                np.place(ax, np.isnan(ax[:]), 0.0)  # put 0.0 anywhere there is nan
                return ax
        if len(data.shape) == 2:
            if data.shape[0] == self.num_axials and data.shape[1] == self.num_assys:
                return np.average(data, axis=1, weights=self.assy_factors)
        return None

    def Radial(self, data: np.ndarray) -> np.ndarray:
        """Calculate the radial average of the data.

        Parameters
        ----------
        data : np.ndarray
            The data to average

        Returns
        -------
        np.ndarray
            The radial average of the data
        """
        if len(data.shape) == 4:
            if (
                data.shape[0] == self.num_pins
                and data.shape[1] == self.num_pins
                and data.shape[2] == self.num_axials
                and data.shape[3] == self.num_assys
            ):
                olderr = np.seterr(invalid="ignore")
                rad = np.sum(data * self.pin_factors, axis=2) / self.radial_factors
                np.seterr(**olderr)
                np.place(rad, np.isnan(rad[:]), 0.0)  # put 0.0 anywhere there is nan
                return rad
            if (
                data.shape[0] == self.num_channels
                and data.shape[1] == self.num_channels
                and data.shape[2] == self.num_axials
                and data.shape[3] == self.num_assys
            ):
                olderr = np.seterr(invalid="ignore")
                rad = np.sum(data * self.channel_factors, axis=2) / self.radial_channel_factors
                np.seterr(**olderr)
                np.place(rad, np.isnan(rad[:]), 0.0)  # put 0.0 anywhere there is nan
                return rad
        return None

    def Assembly(self, data: np.ndarray) -> np.ndarray:
        """Calculate the assembly average of the data.

        Parameters
        ----------
        data : np.ndarray
            The data to average

        Returns
        -------
        np.ndarray
            The assembly average of the data
        """
        if len(data.shape) == 4:
            if (
                data.shape[0] == self.num_pins
                and data.shape[1] == self.num_pins
                and data.shape[2] == self.num_axials
                and data.shape[3] == self.num_assys
            ):
                return np.sum(data * self.pin_factors, axis=(0, 1)) / self.assy_factors
            if (
                data.shape[0] == self.num_channels
                and data.shape[1] == self.num_channels
                and data.shape[2] == self.num_axials
                and data.shape[3] == self.num_assys
            ):
                return np.sum(data * self.channel_factors, axis=(0, 1)) / self.assy_factors
        return None

    def Node(self, data: np.ndarray) -> np.ndarray:
        """Calculate the node results from the data.

        Parameters
        ----------
        data : np.ndarray
            Relevant node data

        Returns
        -------
        np.ndarray
            The node results
        """
        npin = self.num_pins
        asy_node_factors = np.zeros((4, npin, npin))
        mid = npin >> 1

        if npin % 2 == 1:  # odd
            asy_node_factors[0, 0 : mid + 1, 0 : mid + 1] = 1
            asy_node_factors[1, 0 : mid + 1, mid:npin] = 1
            asy_node_factors[2, mid:npin, 0 : mid + 1] = 1
            asy_node_factors[3, mid:npin, mid:npin] = 1

            asy_node_factors[:, 0:npin, mid] *= 0.5
            asy_node_factors[:, mid, 0:npin] *= 0.5

        else:
            asy_node_factors[0, 0:mid, 0:mid] = 1
            asy_node_factors[1, 0:mid, mid:npin] = 1
            asy_node_factors[2, mid:npin, 0:mid] = 1
            asy_node_factors[3, mid:npin, mid:npin] = 1

        pin_factors = self.__calculatepinfactorsfornodes__()

        if len(data.shape) == 4:
            if (
                data.shape[0] == self.num_pins
                and data.shape[1] == self.num_pins
                and data.shape[2] == self.num_axials
                and data.shape[3] == self.num_assys
            ):
                result = np.zeros((4, self.num_axials, self.num_assys))
                for l in range(self.num_assys):
                    for k in range(self.num_axials):
                        for n in range(4):
                            result[n, k, l] = np.sum(data[:, :, k, l] * pin_factors[:, :, k, l] * asy_node_factors[n, :, :])
                            denom = np.sum(pin_factors[:, :, k, l] * asy_node_factors[n, :, :])
                            if denom == 0.0:
                                result[n, k, l] = 0.0
                            else:
                                result[n, k, l] /= denom

                if self.core_sym != 1:
                    if self.size % 2 == 1 and self.num_pins % 2 == 1:  # if odd pins and odd assemblies
                        for ai in range(self.alo, self.ahi):  # for quarter symmetry
                            l = self.core_map[self.alo, ai] - 1  # cut the pin weights by
                            if l < 0:
                                continue  # half on the line of symmetry
                            result[0, :, l] = 0.0
                            result[1, :, l] = 0.0

                        for aj in range(self.alo, self.ahi):  # for quarter symmetry
                            l = self.core_map[aj, self.alo] - 1  # cut the pin weights by
                            if l < 0:
                                continue  # half on the line of symmetry
                            result[0, :, l] = 0.0
                            result[2, :, l] = 0.0

                return result

        return None

    def Radial_Assembly(self, data: np.ndarray) -> np.ndarray:
        """Calculate the radial assembly average of the data.

        Parameters
        ----------
        data : np.ndarray
            The data to average

        Returns
        -------
        np.ndarray
            The radial assembly average of the data
        """
        if len(data.shape) == 4:
            if (
                data.shape[0] == self.num_pins
                and data.shape[1] == self.num_pins
                and data.shape[2] == self.num_axials
                and data.shape[3] == self.num_assys
            ):
                return np.sum(data * self.pin_factors, axis=(0, 1, 2)) / self.radial_assy_factors
            if (
                data.shape[0] == self.num_channels
                and data.shape[1] == self.num_channels
                and data.shape[2] == self.num_axials
                and data.shape[3] == self.num_assys
            ):
                return np.sum(data * self.channel_factors, axis=(0, 1, 2)) / self.radial_assy_factors
            return 0

        if len(data.shape) == 2:
            if data.shape[0] == self.num_axials and data.shape[1] == self.num_assys:
                return np.average(data, axis=0, weights=self.assy_factors)

        return None

    def Exists(self, key: str) -> bool:
        """Check if the key exists in the VERA output file.

        Parameters
        ----------
        key : str
            The key to check for

        Returns
        -------
        bool
            True if the key exists, False otherwise
        """
        if key in self.h5f:
            return True

        return "/CORE/" + key in self.h5f

    def GetData(self, key: str) -> np.ndarray:
        """Get the data from the VERA output file.

        Parameters
        ----------
        key : str
            The key to get the data for

        Returns
        -------
        np.ndarray
            The data
        """
        if key in self.cached_datasets:
            return self.cached_datasets[key]

        if "/CORE/" + key in self.h5f:
            x = self.h5f["/CORE/" + key]
        elif key in self.h5f:
            x = self.h5f[key]

        if isinstance(x, h5py.Dataset):
            x = x[()]
            if len(x.shape) == 1 and x.shape[0] == 1:
                x = x[0]

        if self._cache_datasets:
            print(f"Caching dataset {key:s}")
            self.cached_datasets[key] = x

        return x

    def SetData(self, key: str, item: Any, attrs: dict = None) -> h5py.Dataset:
        """Set the data in the VERA output file.

        Parameters
        ----------
        key : str
            The key to set the data for
        item : Any
            The data to set
        attrs : dict, optional
            The attributes to set, by default None

        Returns
        -------
        h5py.Dataset
            The dataset
        """
        if key in self.h5f:
            ds = self.h5f[key]
            ds[...] = item
            if attrs is not None:
                key, value = attrs
                ds.attrs.modify(key, value)
            if key in self.cached_datasets:
                del self.cached_datasets[key]
            return ds

        ds = self.h5f.create_dataset(key, data=item)
        if attrs is not None:
            key, value = attrs
            ds.attrs.create(key, value)
        return ds

    def GetDataIfExists(self, key: str, badvalue: Any = None, allstates: bool = False) -> Any:
        """Get the data if it exists in the VERA output file.

        Parameters
        ----------
        key : str
            The key to get the data for
        badvalue : Any, optional
            The value to return if the key does not exist, by default None
        allstates : bool, optional
            If True, return the data for all states, by default False

        Returns
        -------
        Any
            The data
        """
        if key in self.cached_datasets:
            return self.cached_datasets[key]

        # if exists, return it
        if self.Exists(key):
            return self.GetData(key)

        # try to return an array over all states
        if allstates and key in self.states[0].h5g:
            return self.GetAllData(key)

        return badvalue

    def GetAllData(self, key: str) -> np.ndarray:
        """Get the data for all states.

        Parameters
        ----------
        key : str
            The key to get the data for

        Returns
        -------
        np.ndarray
            The data for all states
        """
        oldshape = self.states[0][key].shape
        rank = len(oldshape)
        N = len(self.states)
        if rank == 0:
            result = np.empty(N)
            for i in range(N):
                result[i] = self.states[i][key]
            return result

        result = np.empty(N, dtype=list)
        for i in range(N):
            result[i] = self.states[i].h5g[key][()][..., np.newaxis]
        return np.concatenate(result, axis=-1)  # there may be a better way to do this

    def PrintCoreMap(self, data: np.ndarray = None) -> None:
        """Print the core map.

        Parameters
        ----------
        data : np.ndarray, optional
            The data to print, by default None
        """
        if data is None:
            data = self.core_map

        max_ = np.amax(data)
        length = len(str(max_))

        if self.xlabel is not None:
            if self.ylabel is not None:
                s = " ".ljust(length)

            for col in range(self.alo, self.ahi):
                s += " " + self.xlabel[col].ljust(length)
            print(s)

        for row in range(self.alo, self.ahi):
            if self.ylabel is not None:
                s = self.ylabel[row].ljust(length)

            for col in range(self.alo, self.ahi):
                l = self.core_map[row, col]
                if l == 0:
                    s += " ".ljust(length + 1)
                else:
                    s += " " + str(data[row, col]).ljust(length)
            print(s)

    def PrintCore(self, core_data: np.ndarray, format_string: str) -> None:
        """Print the core data.

        Parameters
        ----------
        core_data : np.ndarray
            The core data to print
        format_string : str
            The format string
        """
        f = " {0:" + format_string + "}"
        length = int(format_string[0])
        if self.xlabel is not None:
            if self.ylabel is not None:
                s = " ".ljust(length)

            for col in range(self.alo, self.ahi):
                s += " " + self.xlabel[col].ljust(length)
            print(s)

        for row in range(self.alo, self.ahi):
            if self.ylabel is not None:
                s = self.ylabel[row].ljust(length)

            for col in range(self.alo, self.ahi):
                l = self.core_map[row, col] - 1
                if l == -1:
                    s += " ".ljust(length + 1)
                else:
                    s += f.format(core_data[l])
            print(s)

    def PrintLattice(self, pin_data: np.ndarray, format_string: str, assy: int = -1) -> None:
        """Print the lattice data.

        Parameters
        ----------
        pin_data : np.ndarray
            The pin data to print
        format_string : str
            The format string
        assy: int, optional
            The assembly to print, by default -1
        """
        if assy < 0:
            ax = 0
            zx = self.num_pins
            ay = 0
            zy = self.num_pins
        else:
            ax = self.pxlo[assy]
            zx = self.pxhi
            ay = self.pylo[assy]
            zy = self.pyhi

        f = " {0:" + format_string + "}"
        s = ""
        for row in range(ay, zy):
            for col in range(ax, zx):
                s += f.format(pin_data[row, col])
            print(s)

    def GetCoreIndices(self, core_location: str) -> tuple[int, int]:
        """Get the core indices.

        Parameters
        ----------
        core_location : str
            The core location

        Returns
        -------
        tuple[int, int]
            The core indices
        """
        cols = core_location.split("-")
        col = self.xlabel.index(cols[0])
        row = self.ylabel.index(cols[1])
        return col, row

    def GetDetector(self, core_location: str) -> int:
        """Get the detector.

        Parameters
        ----------
        core_location : str
            The core location

        Returns
        -------
        int
            The detector
        """
        if self.detector_map is None:
            return None
        col, row = self.GetCoreIndices(core_location)
        return self.detector_map[col, row]

    def GetCoreLocation(self, ai: int, aj: int) -> str:
        """Get the core location.

        Parameters
        ----------
        ai : int
            The assembly index
        aj : int
            The assembly index

        Returns
        -------
        str
            The core location
        """
        return self.xlabel[ai] + "-" + self.ylabel[aj]

    @property
    def CacheDatasets(self) -> bool:
        """Get the cache datasets flag."""
        return self._cache_datasets

    @CacheDatasets.setter
    def CacheDatasets(self, value) -> None:
        """Set the cache datasets flag."""
        self._cache_datasets = value
        if not value:
            self.cached_datasets.clear()
            for s in self.states:
                s.cached_datasets.clear()


class Statepoint:
    """Class to represent a statepoint in the VERA output file.

    Parameters
    ----------
    rx : VERAOut
        The VERA output file
    key : str
        The key for the statepoint
    """

    def __init__(self, rx: VERAout, key: str) -> None:
        self.parent = rx
        self.key = key
        self.number = int(key[6:])
        self.h5g = rx.h5f[key]
        self.cached_datasets = {}

    def __getitem__(self, key: str) -> Any:
        return self.GetDataIfExists(key, None)

    def __setitem__(self, key: str, item: Any) -> h5py.Dataset:
        return self.SetData(key, item)

    def Exists(self, key: str) -> bool:
        """Check if the key exists in the statepoint.

        Parameters
        ----------
        key : str
            The key to check for

        Returns
        -------
        bool
            True if the key exists, False otherwise
        """
        return key in self.h5g

    def GetData(self, key: str) -> Any:
        """Get the data from the statepoint.

        Parameters
        ----------
        key : str
            The key to get the data for

        Returns
        -------
        Any
            The data
        """
        if key in self.cached_datasets:
            return self.cached_datasets[key]

        y = self.h5g[key][()]

        if len(y.shape) == 1 and y.shape[0] == 1:
            y = y[0]

        if self.parent.CacheDatasets:
            print(f"Caching dataset {key:s}")
            self.cached_datasets[key] = y
        return y

    def SetData(self, key: str, item: Any, attrs: dict = None) -> h5py.Dataset:
        """Set the data in the statepoint.

        Parameters
        ----------
        key : str
            The key to set the data for
        item : Any
            The data to set
        attrs : dict, optional
            The attributes to set, by default None

        Returns
        -------
        h5py.Dataset
            The dataset
        """
        if key in self.h5g:
            ds = self.h5g[key]
            ds[...] = item
            if attrs is not None:
                key, value = attrs
                ds.attrs.modify(key, value)
            if key in self.cached_datasets:
                del self.cached_datasets[key]
            return ds

        ds = self.h5g.create_dataset(key, data=item)
        if attrs is not None:
            key, value = attrs
            ds.attrs.create(key, value)
        return ds

    def GetDataIfExists(self, key: str, badvalue: Any = None) -> Any:
        """Get the data if it exists in the statepoint.

        Parameters
        ----------
        key : str
            The key to get the data for
        badvalue : Any, optional
            The value to return if the key does not exist, by default None

        Returns
        -------
        Any
            The data
        """
        if key in self.cached_datasets:
            return self.cached_datasets[key]

        if key in self.parent.cached_datasets:  # faster here, but should really be AFTER the check to
            return self.parent.cached_datasets[key]  # see if it exists in this statepoint. This is not likely.

        if self.Exists(key):
            return self.GetData(key)

        return self.parent.GetDataIfExists(key, badvalue=badvalue, allstates=False)

    def GetAxialOffset(self) -> float:
        """Get the axial offset.

        Returns
        -------
        float
            The axial offset
        """
        if self.Exists("pin_powers"):
            axpow = self.parent.Axial(self.h5g["pin_powers"][()])
            return self.parent.__calculateaxialoffset__(axpow)
        return None

    def GetInletTH(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the inlet temperature and density.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The inlet temperature and density
        """
        if self.Exists("tinlet"):
            tin = self.GetData("tinlet")
        elif self.Exists("core_inlet_temp"):
            tin = self.GetData("core_inlet_temp")
        elif self.Exists("channel_liquid_temps"):
            tin = self.parent.Average(self.GetData("channel_liquid_temps")[:, :, 0, :])
        elif self.Exists("pin_mod_temp"):
            tin = self.parent.Average(self.GetData("pin_mod_temp")[:, :, 0, :])
        elif self.Exists("pin_cool_temp"):
            tin = self.parent.Average(self.GetData("pin_cool_temp")[:, :, 0, :])
        else:
            tin = None

        if self.Exists("inlet_dens"):
            pin = self.GetData("inlet_dens")
        elif self.Exists("core_inlet_density"):
            pin = self.GetData("core_inlet_density") / 1000.0  # CTF edits in kg/m^3
        elif self.Exists("channel_liquid_density"):
            pin = self.parent.Average(self.GetData("channel_liquid_density")[:, :, 0, :]) / 1000.0  # CTF edits in kg/m^3
        elif self.Exists("pin_mod_dens"):
            pin = self.parent.Average(self.GetData("pin_mod_dens")[:, :, 0, :])
        elif self.Exists("pin_cool_dens"):
            pin = self.parent.Average(self.GetData("pin_cool_dens")[:, :, 0, :])
        elif self.Exists("modden"):
            pin = self.GetData("modden")
        elif self.Exists("coolden"):
            pin = self.GetData("coolden")
        else:
            pin = None

        return tin, pin

    def GetOutletTH(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the outlet temperature and density.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The outlet temperature and density
        """
        if self.Exists("outlet_temp"):
            tout = self.GetData("outlet_temp")
        elif self.Exists("core_outlet_temp"):
            tout = self.GetData("core_outlet_temp")
        elif self.Exists("channel_liquid_temps"):
            tout = self.parent.Average(self.GetData("channel_liquid_temps")[:, :, -1, :])
        elif self.Exists("pin_mod_temp"):
            tout = self.parent.Average(self.GetData("pin_mod_temp")[:, :, -1, :])
        elif self.Exists("pin_cool_temp"):
            tout = self.parent.Average(self.GetData("pin_cool_temp")[:, :, -1, :])
        else:
            tout = None

        if self.Exists("outlet_dens"):
            pout = self.GetData("outlet_dens")
        elif self.Exists("outlet_density"):
            pout = self.GetData("outlet_density")
        elif self.Exists("core_outlet_density"):
            pout = self.GetData("core_outlet_density") / 1000.0  # CTF edits in kg/m^3
        elif self.Exists("channel_liquid_density"):
            pout = self.parent.Average(self.GetData("channel_liquid_density")[:, :, -1, :]) / 1000.0  # CTF edits in kg/m^3
        elif self.Exists("pin_mod_dens"):
            pout = self.parent.Average(self.GetData("pin_mod_dens")[:, :, -1, :])
        elif self.Exists("pin_cool_dens"):
            pout = self.parent.Average(self.GetData("pin_cool_dens")[:, :, -1, :])
        elif self.Exists("modden"):
            pout = self.GetData("modden")
        elif self.Exists("coolden"):
            pout = self.GetData("coolden")
        else:
            pout = None

        return tout, pout

    def GetCoreExposure(self) -> float:
        """Get the core exposure.

        Returns
        -------
        float
            The core exposure
        """
        if self.Exists("pin_exposures"):
            if self.parent.Exists("/CORE/initial_mass"):
                return np.average(
                    self.h5g["pin_exposures"][()], axis=None, weights=self.parent.pin_factors * self.parent.initial_mass
                )  # check this weighting
            print("No initial mass")
        return None

    def GetCoreXenon(self) -> float:
        """Get the core xenon.

        Returns
        -------
        float
            The core xenon
        """
        if self.Exists("isotopes_Xe-635"):
            return self.parent.Average(self.h5g["isotopes_Xe-635"])
        return None

    def GetAssemblyExposures(self) -> np.ndarray:
        """Get the assembly exposures.

        Returns
        -------
        np.ndarray
            The assembly exposures
        """
        if self.Exists("pin_exposures"):
            if self.parent.Exists("/CORE/initial_mass"):
                x = self.h5g["pin_exposures"][()]
                y = np.sum(x * self.parent.pin_factors * self.parent.initial_mass, axis=(0, 1, 2)) / np.sum(
                    self.parent.pin_factors * self.parent.initial_mass, axis=(0, 1, 2)
                )
                return y
            print("No initial mass")
        return None

    def GetPinExposures(self) -> np.ndarray:
        """Get the pin exposures.

        Returns
        -------
        np.ndarray
            The pin exposures
        """
        if self.Exists("pin_exposures"):
            if self.parent.Exists("/CORE/initial_mass"):
                x = self.h5g["pin_exposures"][()]

                olderr = np.seterr(invalid="ignore")
                y = np.sum(x * self.parent.pin_factors * self.parent.initial_mass, axis=2) / np.sum(
                    self.parent.pin_factors * self.parent.initial_mass, axis=2
                )
                np.seterr(**olderr)
                np.place(y, np.isnan(y[:]), 0.0)  # put 0.0 anywhere there is nan
                return y
            print("No initial mass")
        return None

    def GetBank(self, bank: str) -> np.ndarray:
        """Get the bank data.

        Parameters
        ----------
        bank : str
            The bank to get the data for

        Returns
        -------
        np.ndarray
            The bank data
        """
        if self.Exists("bank_labels"):
            labels = [x.strip() for x in self["bank_labels"]]
            try:
                ndx = labels.index(bank)
                return self["bank_pos"][ndx]
            except ValueError:
                return None
        return None
