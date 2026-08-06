from __future__ import annotations

from enum import Flag, auto
from typing import Any, Iterable, List, Tuple, TypeAlias

import numpy as np

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        """Extends the Enum class for strings."""


# Alias for a 2D coordinate in the form (row, column).
CoreLocation: TypeAlias = Tuple[int, int]

angle_eps = 0.01
degree_to_turn = 1.0 / 360.0


class GeomType(Flag):
    """Defines geometry enumerations"""

    invalid = 0
    cartesian = auto()
    hex_pointy_top = auto()
    hex_flat_top = auto()
    hex = hex_pointy_top | hex_flat_top

    def reverse_hex(self):
        """Returns the opposite hex orientation"""

        if self == GeomType.hex_flat_top:
            return GeomType.hex_pointy_top
        if self == GeomType.hex_pointy_top:
            return GeomType.hex_flat_top
        return GeomType.invalid

    @classmethod
    def from_string(cls, value: str) -> GeomType:
        """Create a SymEnum from a string.

        Parameters
        ----------
        value : str
            The string to create the enum from.

        Returns
        -------
        SymEnum
            The resulting enum.
        """
        # pylint: disable=no-member
        return cls._member_map_[value]


class CellType(StrEnum):
    """Defines cell type enumerations"""

    NULL = auto()
    fuel = auto()
    gtube = auto()
    triso_compact = auto()
    large4 = auto()
    large7 = auto()
    heat_pipe = auto()
    insert = auto()
    control = auto()
    coolant_channel = auto()
    other = auto()

    @classmethod
    def has_key(cls, key: str) -> bool:
        """Check if a particular string is a valid CellType.

        Parameters
        ---------
        key : str
            The string to check.

        Returns
        -------
        bool
            Whether or not the string is a valid CellType name.
        """
        # pylint: disable=no-member
        return key in cls._member_names_


class Location:
    """Defines location data, especially for setting up materials

    Parameters
    ----------
    asy : Tuple[int, int], optional
        The assembly coordinates (row, column).  Defaults to None.
    pin : Tuple[int, int], optional
        The pin coordinates (row, column).  Defaults to None.
    minz : float, optional
        The lower boundary of the axial mesh containing the location.  Defaults to None.
    maxz : float, optional
        The upper boundary of the axial mesh containing the location.  Defaults to None.
    origin : Tuple[float, float], optional
        An (x,y) coordinate for this location.  Defaults to (0., 0.).
    fidelity : Enum, optional
        The fidelity associated with this location.  Defaults to None.
    axial_index : int, optional
        The index in the axial mesh for this location.  Defaults to None.
    radial_index : int, optional
        The index in the radial mesh for this location.  Defaults to None.
    geom_type : GeomType, optional
        The type of geometry this location lies in.  Defaults to None.
    cell_name : str, optional
        The name of the cell the location is in.  Defaults to None.
    """

    # A weird thing here is that fidelity is an enum defined in mcVERA so we can't declare the type here
    def __init__(
        self,
        asy: tuple[int, int] = None,
        pin: tuple[int, int] = None,
        minz: float = None,
        maxz: float = None,
        origin: tuple[float, float] = (0.0, 0.0),
        fidelity=None,
        axial_index: int = None,
        radial_index: int = None,
        geom_type: GeomType = None,
        cell_name: str = None,
    ):
        self.asy = asy
        self.pin = pin
        self.minz = minz
        self.maxz = maxz
        self.origin = origin
        self.fidelity = fidelity
        self.axial_index = axial_index
        self.radial_index = radial_index
        self.geom_type = geom_type
        self.cell_name = cell_name

    def __str__(self) -> str:
        """Formats the location object as a str

        Returns
        -------
        str
            The string representation of the object
        """

        if self.asy is not None:
            s = f"Assembly Location: ({self.asy[0]:d},{self.asy[1]:d})\n"
        else:
            s = "Assembly Location: Not Set\n"
        if self.pin is not None:
            s += f"     Pin Location: ({self.pin[0]:d},{self.pin[1]:d})\n"
        else:
            s += "     Pin Location: Not Set\n"
        s += f"  Minimum Axial Z: {'Not Set' if self.minz is None else str(self.minz)}\n"
        s += f"  Maximum Axial Z: {'Not Set' if self.maxz is None else str(self.maxz)}\n"
        s += f"  Origin Location: ({self.origin[0]:g}, {self.origin[1]:g})\n"
        s += f"         Fidelity: {'Not Set' if self.fidelity is None else str(self.fidelity)}\n"
        s += f"      Axial Index: {'Not Set' if self.axial_index is None else str(self.axial_index)}\n"
        s += f"     Radial Index: {'Not Set' if self.radial_index is None else str(self.radial_index)}\n"
        s += f"    Geometry Type: {'Not Set' if self.geom_type is None else str(self.geom_type)}\n"
        s += f"        Cell Name: {'Not Set' if self.cell_name is None else str(self.cell_name)}\n"
        return s

    def copy(
        self,
        asy: tuple[int, int] = None,
        pin: tuple[int, int] = None,
        minz: float = None,
        maxz: float = None,
        origin: tuple[float, float] = None,
        fidelity: Enum = None,
        axial_index: int = None,
        radial_index: int = None,
        geom_type: GeomType = None,
        cell_name: str = None,
    ):
        """Generates a deep copy of a Location object; other arguments are used to override values copied from source

        Parameters
        ----------
        asy : Tuple[int, int], default=None
            The assembly coordinates (row, column).
        pin : Tuple[int, int], default=None
            The pin coordinates (row, column).
        minz : float, default=None
            The lower boundary of the axial mesh containing the location.
        maxz : float, default=None
            The upper boundary of the axial mesh containing the location.
        origin : Tuple[float, float], default=(0., 0.)
            An (x,y) coordinate for this location.
        fidelity : Enum, default=None
            The fidelity associated with this location.
        axial_index : int, default=None
            The index in the axial mesh for this location.
        radial_index : int, default=None
            The index in the radial mesh for this location.
        geom_type : GeomType, default=None
            The type of geometry this location lies in.
        cell_name : str, default=None
            The name of the cell the location is in.

        Returns
        -------
        result : Location
            The Location object resulting from the copy procedure.
        """

        result = Location(
            self.asy,
            self.pin,
            self.minz,
            self.maxz,
            self.origin,
            self.fidelity,
            self.axial_index,
            self.radial_index,
            self.geom_type,
            self.cell_name,
        )
        if asy is not None:
            result.asy = asy
        if pin is not None:
            result.pin = pin
        if minz is not None:
            result.minz = minz
        if maxz is not None:
            result.maxz = maxz
        if origin is not None:
            result.origin = origin
        if fidelity is not None:
            result.fidelity = fidelity
        if axial_index is not None:
            result.axial_index = axial_index
        if radial_index is not None:
            result.radial_index = radial_index
        if geom_type is not None:
            result.geom_type = geom_type
        if cell_name is not None:
            result.cell_name = cell_name

        return result

    @property
    def elevation(self) -> float:
        """Returns the elevation of the location"""
        if self.minz is not None and self.maxz is not None:
            return 0.5 * (self.minz + self.maxz)
        if self.minz is not None:
            return self.minz
        return self.maxz

    @property
    def delz(self) -> float:
        """Returns the axial width of the location"""
        if self.minz is not None and self.maxz is not None:
            return self.maxz - self.minz
        return 0.0


def bound_angle(angle: float) -> float:
    """Restricts an angle to [0, 360)

    Parameters
    ----------
    angle : float
        The original angle

    Returns
    -------
    float
        The bounded angle in [0, 360)
    """

    return angle % 360.0


def angle_in_arc(angle: float, arc: Tuple[float, float], fuzzy: bool = True) -> bool:
    """Determines if a given angle is inside an arc, accounting for periodicity

    Parameters
    ----------
    angle : float
        The angle to test
    arc : Tuple[float, float]
        The starting and stopping angles for the arc to test against
    fuzzy : bool, optional
        Whether to use fuzzy or exact comparisons.  Defaults to true.

    Returns
    -------
    bool
        Whether angle is inside arc or not.
    """

    assert 0.0 <= angle < 720.0
    assert 0.0 <= arc[0] < 720.0
    assert 0.0 <= arc[1] < 720.0
    assert arc[1] >= arc[0]
    eps = angle_eps if fuzzy else 0.0

    for a in [angle, angle + 360.0]:
        if arc[0] + eps < a <= arc[1] - eps:
            return True

    return False


def arcs_overlap(arc1: Tuple[float, float], arc2: Tuple[float, float], fuzzy: bool = True) -> bool:
    """Determines if any portion of an arc overlaps another arc, accounting for periodicity

    Both arcs must be in the interval [0, 720), and each arc must satisfy arc[1] >= arc[0].

    Parameters
    ----------
    arc1 : Tuple[float, float]
        The starting and stopping angles for the first arc
    arc2 : Tuple[float, float]
        The starting and stopping angles for the second arc
    fuzzy : bool, optional
        Whether to use fuzzy or exact comparisons.  Defaults to true.

    Returns
    -------
    bool
        Whether the two arcs overlap or not.
    """

    assert 0.0 <= arc1[0] < 720.0
    assert 0.0 <= arc1[1] < 720.0
    assert 0.0 <= arc2[0] < 720.0
    assert 0.0 <= arc2[1] < 720.0
    assert arc1[1] >= arc1[0]
    assert arc2[1] >= arc2[0]
    eps = angle_eps if fuzzy else 0.0

    # Easy case: either both arcs cross 0 or neither of them does
    if arc1[0] < arc2[1] - eps and arc1[1] > arc2[0] + eps:
        return True
    # Edge case 1: arc1 crosses 0
    if arc1[0] < arc2[1] + 360.0 - eps and arc1[1] > arc2[0] + 360.0 + eps:
        return True
    # Edge case 2: arc2 crosses 0
    if arc1[0] + 360.0 < arc2[1] - eps and arc1[1] + 360.0 > arc2[0] + eps:
        return True
    return False


def rotate_hex_vector(vec=Tuple[float, float], rotations=float) -> Tuple[float, float]:
    """Rotations a vector, given the number of 60-degree rotations

    Parameters
    ----------
    vec : Tuple[float, float]
        The vector to rotate.
    rotations : float
        The number of 60-degree rotations, positive or negative.

    Returns
    -------
    Tuple[float, float]
        The rotated vector.
    """

    return rotate_vector(vec, rotations * 60.0)


def rotate_vector(vec=Tuple[float, float], angle=float) -> Tuple[float, float]:
    """Rotations a vector, given angle (in degrees) for which it needs to be rotated

    Parameters
    ----------
    vec : Tuple[float, float]
        The vector to rotate.
    angle : float
        The angle by which to rotate the vector, in degrees.

    Returns
    -------
    Tuple[float, float]
        The rotated vector.
    """

    a = bound_angle(angle) / 180.0 * np.pi
    return (np.cos(a) * vec[0] - np.sin(a) * vec[1], np.sin(a) * vec[0] + np.cos(a) * vec[1])


def circle_circumference(r1: float, r2: float = 0.0) -> float:
    """Calculates the circumference of a circle, or an annulus if r2 is provided

    Parameters
    ----------
    r1 : float
        The outer radius of the circle or annulus.
    r2 : float, optional
        The inner radius of the annulus. Defaults to 0.0.

    Returns
    -------
    float
        The circumference of the circle or annulus

    Notes
    -----
    If r2 is provided, the outer and inner circumference of the annulus are added together and returned.
    """

    return 2.0 * np.pi * (r1 + r2)


def circle_area(r1: float, r2: float = 0.0) -> float:
    """Calculates the area of a circle, or an annulus if r2 is provided

    Parameters
    ----------
    r1 : float
        The outer radius of the circle or annulus
    r2 : float, optional
        The inner radius of the annulus. Defaults to 0.0.

    Returns
    -------
    float
        The area of the circle or annulus.
    """

    assert r1 >= 0.0
    assert 0.0 <= r2 <= r1
    return np.pi * (np.power(r1, 2.0) - np.power(r2, 2.0))


def sphere_volume(r1: float, r2: float = 0.0) -> float:
    """Calculates the volume of a sphere, or a shell if r2 is provided.

    Parameters
    ----------
    r1 : float
        The outer radius of the sphere or shell.
    r2 : float, optional
        The inner radius of the shell. Defaults to 0.0.

    Returns
    -------
    float
        The volume of the sphere or shell.
    """

    assert r1 >= 0.0
    assert 0.0 <= r2 <= r1
    return 4.0 / 3.0 * np.pi * (np.power(r1, 3.0) - np.power(r2, 3.0))


def cylinder_volume(z: float, r1: float, r2: float = 0.0) -> float:
    """Calculates the volume of a cylinder, or a cylindrical shell if r2 is provided.

    Parameters
    ----------
    z : float
        The height of the cylinder.
    r1 : float
        The outer radius of the cylinder.
    r2 : float, optional
        The inner radius of the cylinder. Defaults to 0.0.

    Returns
    -------
    float
        The volume of the cylinder.
    """

    assert z >= 0.0
    assert r1 >= 0.0
    assert 0.0 <= r2 <= r1
    return circle_area(r1, r2) * z


def parent_hexagon_area(
    apothem: float, rings: int = 0, inner_apothem: float = 0.0, geom_type: GeomType = None, inner_geom_type: GeomType = None
) -> float:
    """Calculates the area of a parent hexagon that contains a hexagonal array

    Parameters
    ----------
    apothem : float
        The parent hexagon's apothem.
    rings : int, optional
        The number of rings in the hexagonal array contained in the parent hexagon. Defaults to 0.
    inner_apothem : float, optional
        The apothem for the hexagons in the contained hexagonal array. Defaults to 0.0.
    geom_type : GeomType
        The orientation of the parent hexagon. Defaults to None.
    inner_geom_type : GeomType
        The orientation of the hexagons in the contained hexagonal array. Defaults to None.

    Returns
    -------
    float
        The area of the parent hexagon.

    Notes
    -----
    If values are provided for the inner hexagonal array, the area of the array is subtracted from
    the area of the outer hexagon. Clipping of the array by the parent hexagon is accounted for.
    All combinations of orientations are supported.
    """

    assert apothem > 0.0

    # Get the base hexagon area
    area = hexagon_area(apothem)

    # Now we check for clipping...
    if rings > 0 and apothem <= parent_hex_min_pitch(rings, inner_apothem * 2.0):
        assert inner_apothem > 0.0
        assert geom_type in GeomType.hex
        assert inner_geom_type in GeomType.hex

        # Figure out how many interior, edge, and corner elements are in the array
        corner = 0 if rings == 1 else 6
        edge = max((rings - 2) * 6, 0)
        interior = num_units_from_hex_rings(rings) - corner - edge
        # Calculate the areas for each type of element (only 3 unique values because of symmetry)
        corner_area = (
            0.0 if corner == 0 else hexagon_area(inner_apothem, (rings - 1, 0), rings, apothem, inner_geom_type, geom_type)
        )
        edge_area = 0.0 if edge == 0 else hexagon_area(inner_apothem, (rings, 0), rings, apothem, inner_geom_type, geom_type)
        interior_area = hexagon_area(inner_apothem)
        # Decrement the total area by subtracting out the array
        area -= corner * corner_area
        area -= edge * edge_area
        area -= interior * interior_area
    # The easy case with no clipping
    elif rings > 0:
        area -= num_units_from_hex_rings(rings) * hexagon_area(inner_apothem)
    return area


def hexagon_points(
    apothem: float, geom_type: GeomType, centroid: Tuple[float, float] = (0.0, 0.0)
) -> List[Tuple[float, float]]:
    """Returns a list of vertexes that make up a hexagon.

    Parameters
    ----------
    apothem : float
        The hexagon's apothem.
    geom_type : GeomType
        The hexagon's orientation.
    centroid : Tuple[float, float], optional
        The centroid of the hexagon. Defaults to (0.0, 0.0).

    Returns
    -------
    List[Tuple[float, float]]
        The list of vertexes defining the hexagon.

    Notes
    -----
    The lsit always begins with positive x axis and proceeds clockwise. The first point is added
    again as the last point to expilcitly close the polygon, resulting in a length-7 list.
    """

    assert apothem > 0.0
    assert geom_type in GeomType.hex

    side_length = 2.0 / np.sqrt(3) * apothem
    if geom_type == GeomType.hex_flat_top:
        hexagon = [
            (centroid[0] + side_length, centroid[1]),
            (centroid[0] + side_length * 0.5, centroid[1] - apothem),
            (centroid[0] - side_length * 0.5, centroid[1] - apothem),
            (centroid[0] - side_length, centroid[1]),
            (centroid[0] - side_length * 0.5, centroid[1] + apothem),
            (centroid[0] + side_length * 0.5, centroid[1] + apothem),
        ]
    else:
        hexagon = [
            (centroid[0] + apothem, centroid[1] - side_length * 0.5),
            (centroid[0], centroid[1] - side_length),
            (centroid[0] - apothem, centroid[1] - side_length * 0.5),
            (centroid[0] - apothem, centroid[1] + side_length * 0.5),
            (centroid[0], centroid[1] + side_length),
            (centroid[0] + apothem, centroid[1] + side_length * 0.5),
        ]
    hexagon.append((hexagon[0][0], hexagon[0][1]))
    return hexagon


def hex_element_centroid(
    apothem: float, array_coords: Tuple[int, int], rings: int, geom_type: GeomType = GeomType.hex_pointy_top
) -> Tuple[float, float]:
    """Calculates the centroid of an a hexagonal array element.

    Parameters
    ----------
    apothem : float
        The hexagon apothem
    array_coords : Tuple[int, int]
        The coordinates of the hexagon in the hexagonal array.
    rings : int
        The number of rings in the array. 1 ring is a single hexagon.
    geom_type : GeomType, optional
        The orientation of the hexagonal array. Default is GeomType.hex_pointy_top.

    Returns
    -------
    Tuple[float, float]
        The (x,y) centroid of the requested hexagon.

    Notes
    -----
    The center of the center element is (0.0, 0.0). array_coords are (row, column).
    The column index starts at 0 for every row. Row 0 is the top row of the array.
    """

    assert apothem > 0.0
    assert rings > 0
    mid = rings - 1  # The middle row & column index
    assert 0 <= array_coords[0] < 2 * rings - 1
    assert 0 <= array_coords[1] < 2 * rings - 1 - np.abs(mid - array_coords[0])
    assert geom_type in GeomType.hex

    # Calculate x,y for array of pointy-topped hexagons first
    x = (-mid + np.abs(mid - array_coords[0]) * 0.5 + array_coords[1]) * apothem * 2.0
    y = (mid - array_coords[0]) * apothem * 2.0 * np.cos(np.pi / 6.0)
    # If pointy-topped, we just need to rotate 30 degrees
    if geom_type == GeomType.hex_flat_top:
        x, y = rotate_hex_vector((x, y), 0.5)
    return x, y


def line_midpoint(line: Tuple[Tuple[float, float], Tuple[float, float]]) -> Tuple[float, float]:
    """Calculates the midpoint of a line.

    Parameters
    ----------
    line : Tuple[Tuple[float, float], Tuple[float, float]]
        Defines the starting and stopping (x,y) coordinates of the line's endpoints.

    Returns
    -------
    Tuple[float, float]
        The (x,y) coordinates of the line's midpoint
    """

    return (0.5 * (line[0][0] + line[1][0]), 0.5 * (line[0][1] + line[1][1]))


def line_slope_intercept(line: Tuple[Tuple[float, float], Tuple[float, float]]) -> Tuple[float, float]:
    """Calculates the slope m and y-intercept b of a line defined by 2 points.

    Parameters
    ----------
    line : Tuple[Tuple[float, float], Tuple[float, float]]
        Defines the starting and stopping (x,y) coordinates of the line's endpoints.

    Returns
    -------
    Tuple[float, float]
        The slope and y-intercept of the line.

    Notes
    -----
    If the line is vertical, then (None, x-intercept) is returned instead.
    """

    if abs(line[1][0] - line[0][0]) < 1.0e-10:
        m = None
        b = line[0][0]
    else:
        m = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
        b = line[0][1] - m * line[0][0]
    return m, b


def point_on_line(
    point: Tuple[float, float], line: Tuple[Tuple[float, float], Tuple[float, float]], tol: float = 1.0e-6
) -> bool:
    """Determines if a point is on a line.

    Parameters
    ----------
    point : Tuple[float, float]
        The (x,y) coordinates of the point to test.
    line : Tuple[Tuple[float, float], Tuple[float, float]]
        The starting and stopping (x,y) coordinates of the line endpoints.
    tol : float, optional
        The geometric tolerance to use for testing if the point is on the line. Defaults to 1.0e-6.

    Returns
    -------
    bool
        Whether the point is on the line or not.
    """

    m, b = line_slope_intercept(line)
    # Vertical
    if m is None:
        return (abs(point[0] - b) < tol) and (
            (min(line[0][1], line[1][1]) - tol) <= point[1] <= (max(line[0][1], line[1][1]) + tol)
        )
    # Horizontal
    if abs(m) < tol:
        return (abs(point[1] - b) < tol) and min(line[0][0], line[1][0]) - tol <= point[0] <= max(line[0][0], line[1][0]) + tol
    # Other
    return (abs(point[1] - (m * point[0] + b)) < tol) and min(line[0][0], line[1][0]) - tol <= point[0] <= max(
        line[0][0], line[1][0]
    ) + tol


def line_intersection(
    line1: Tuple[Tuple[float, float], Tuple[float, float]], line2: Tuple[Tuple[float, float], Tuple[float, float]]
) -> Tuple[float, float]:
    """Finds the intersection of 2 line segments.

    Parameters
    ----------
    line1 : Tuple[Tuple[float, float], Tuple[float, float]]
        The starting and stopping (x,y) coordinates of the first line's endpoints.
    line2 : Tuple[Tuple[float, float], Tuple[float, float]]
        The starting and stopping (x,y) coordinates of the second line's endpoints.

    Returns
    -------
    Tuple[float, float]
        The (x,y) coordinates of the intersection.

    Notes
    -----
    Returns None if the lines do not intersect or are colinear.
    """

    def det(a, b, c, d):
        return a * d - b * c

    # Calculate the determinants
    denom = det(line1[0][0] - line1[1][0], line1[0][1] - line1[1][1], line2[0][0] - line2[1][0], line2[0][1] - line2[1][1])
    if denom == 0:
        return None  # Lines are parallel or coincident

    # Calculate the intersection point
    det1 = det(line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    det2 = det(line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    x = det(det1, line1[0][0] - line1[1][0], det2, line2[0][0] - line2[1][0]) / denom
    y = det(det1, line1[0][1] - line1[1][1], det2, line2[0][1] - line2[1][1]) / denom

    # Check if the intersection point is within both line segments
    if point_on_line((x, y), line1) and point_on_line((x, y), line2):
        return (x, y)

    return None  # No intersection within the line segments


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]], tol: float = 1e-6) -> int:
    """Determines if a point is inside a polygon.

    Parameters
    ----------
    point : Tuple[float, float]
        The (x,y) coordinates of the point to test.
    polygon : List[Tuple[float, float]]
        The list of points defining the polygon.
    tol : float, optional
        The geometric tolerance to use for testing whether the point is. Defaults to 1.0e-6.

    Returns
    -------
    int
        -1 (point inside polygon), 1 (point outside polygon), or 0 (point on boundary)

    Notes
    -----
    Polygon points must be ordered clockwise. The first point can be repeated at the end of the list,
    but does not have to be.

    Algorithm based on https://github.com/mikolalysenko/robust-point-in-polygon/blob/master/robust-pnp.js
    and rewritten with ChatGPT.
    """

    assert len(polygon) > 3  # At least 4 points required for a closed triangle
    assert tol >= 0.0

    def orient(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float]) -> float:
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    n = len(polygon)
    inside = 1
    lim = n
    if abs(polygon[0][0] - polygon[-1][0]) < tol and abs(polygon[0][1] - polygon[-1][1]) < tol:
        n -= 1
        lim -= 1

    i = 0
    j = n - 1
    while i < lim:
        yj = polygon[j][1]
        yi = polygon[i][1]

        if yj < yi - tol:
            if yj + tol < point[1] < yi - tol:
                s = orient(polygon[i], polygon[j], point)
                if abs(s) < tol:
                    return 0  # on boundary
                inside ^= 0 < s - tol
            elif abs(point[1] - yi) < tol:
                yk = polygon[(i + 1) % n][1]
                if yi < yk - tol:
                    s = orient(polygon[i], polygon[j], point)
                    if abs(s) < tol:
                        return 0  # on boundary
                    inside ^= 0 < s - tol
        elif yi < yj - tol:
            if yi + tol < point[1] < yj - tol:
                s = orient(polygon[i], polygon[j], point)
                if abs(s) < tol:
                    return 0  # on boundary
                inside ^= s < 0 - tol
            elif abs(point[1] - yi) < tol:
                yk = polygon[(i + 1) % n][1]
                if yk < yi - tol:
                    s = orient(polygon[i], polygon[j], point)
                    if abs(s) < tol:
                        return 0  # on boundary
                    inside ^= s < 0 - tol
        elif abs(point[1] - yi) < tol:
            x0 = min(polygon[i][0], polygon[j][0])
            x1 = max(polygon[i][0], polygon[j][0])

            if i == 0:
                while j > 0:
                    k = (j + n - 1) % n
                    if abs(polygon[k][1] - point[1]) >= tol:
                        break
                    px = polygon[k][0]
                    x0 = min(x0, px)
                    x1 = max(x1, px)
                    j = k
                if j == 0:
                    if x0 <= point[0] <= x1:
                        return 0  # on boundary
                    return -1  # inside
                lim = j + 1

            y0 = polygon[(j + n - 1) % n][1]
            while i + 1 < lim:
                if abs(polygon[i + 1][1] - point[1]) >= tol:
                    break
                x0 = min(x0, polygon[i + 1][0])
                x1 = max(x1, polygon[i + 1][0])
                i += 1
            if x0 <= point[0] <= x1:
                return 0  # on boundary
            if point[0] < x0 - tol and (y0 < point[1] - tol != polygon[(i + 1) % n][1] < point[1] - tol):
                inside ^= 1

        j = i
        i += 1

    return 2 * inside - 1


def polygon_intersection(
    subject: List[Tuple[float, float]], clipper: List[Tuple[float, float]], tol: float = 1.0e-6
) -> List[Tuple[float, float]]:
    """Calculates the intersection of 2 arbitrary polygons.

    Parameters
    ----------
    subject : List[Tuple[float, float]]
        The points defining the subject polygon.
    clipper : List[Tuple[float, float]]
        The points defining the clipping polygon.
    tol : float, optional
        The tolerance to use for testing points. Defaults to 1.0e-6.

    Returns
    -------
    List[Tuple[float, float]]
        The points defining the area inside both subject and clipper.

    Notes
    -----
    Assumptions:
       1. Both polygons are ordered clockwise
       2. The first vertex of each polygon is the same as the last (explicitly close the polygon)

    If the polygons do not intersect, the subject polygon is returned.
    """

    # Based on the Weiler-Atherton clipping algorithm; Step 1 was to get all the vertexes (provide via input)
    # Step 2: Label each vertex of the clipping polygon (large hex) as begin
    # inside (True) or outside (False) the target polygon (small hex)
    clipper_inside_subject = []
    for point in clipper:
        clipper_inside_subject.append(point_in_polygon(point, subject))

    # Step 3: Find all polygon intersections and insert them into both lists, linking the lists at the intersections
    subj_index = 1
    while subj_index < len(subject):
        sp1 = subject[subj_index - 1]
        sp2 = subject[subj_index]
        clip_index = 1
        while clip_index < len(clipper):
            lp1 = clipper[clip_index - 1]
            lp2 = clipper[clip_index]
            intersect = line_intersection((sp1, sp2), (lp1, lp2))
            if intersect:
                if not any(abs(intersect[0] - point[0]) < tol and abs(intersect[1] - point[1]) < tol for point in subject):
                    subject.insert(subj_index, intersect)
                if not any(abs(intersect[0] - point[0]) < tol and abs(intersect[1] - point[1]) < tol for point in clipper):
                    clipper.insert(clip_index, intersect)
                    clipper_inside_subject.insert(clip_index, 0)
                    clip_index += 1  # Increment an extra time; the same 2 line segments cannot cross twice
            clip_index += 1
        subj_index += 1

    # Step 4: Generate a list of "inbound" intersections
    inbound = [False for _ in range(len(clipper))]
    for clip_index, c in enumerate(clipper):
        if clipper_inside_subject[clip_index - 1] == 0:
            # 2 points in a row are on the boundary, so check the midpoint
            if clipper_inside_subject[clip_index] == 0:
                midpoint = line_midpoint((clipper[clip_index - 1], c))
                if point_in_polygon(midpoint, subject, tol=tol):
                    inbound[clip_index - 1] = True
            # Next point is inside the polygon, so set to True
            elif clipper_inside_subject[clip_index] == -1:
                inbound[clip_index - 1] = True
    inbound[-1] = inbound[0]

    # Before proceeding, if nothing is inbound there is no intersection
    # and we should just return subject
    if not any(inbound):
        # No intersections; just return subject
        intersection = list(subject)
        return intersection

    # Step 5: Follow each intersection clockwise around the linked lists until the start position is found
    # First, find a point on subject that is also an inbound point on clipper
    start_subj_index = -1
    for subj_index, s in enumerate(subject):
        clip_index = -1
        for k, c in enumerate(clipper):
            if abs(s[0] - c[0]) < tol and abs(s[1] - c[1]) < tol:
                clip_index = k
                break
        if clip_index > -1:
            if inbound[clip_index]:
                start_subj_index = subj_index
                break
    # Now follow the clipper inside subject and subject inside the clipper
    solved = False
    subj_index = start_subj_index
    intersection = []
    n1 = len(subject) - 1
    n2 = len(clipper) - 1
    clipper_points_added = []
    while subj_index != start_subj_index or not solved:
        intersection.append(subject[subj_index])
        clip_index = -1
        # See if this subject point is also in the clipper
        for k, c in enumerate(clipper):
            if abs(subject[subj_index][0] - c[0]) < tol and abs(subject[subj_index][1] - c[1]) < tol:
                clip_index = k
                break
        # If this point is also in the clipper, append it to the list of added points
        if clip_index != -1 and clip_index not in clipper_points_added:
            clipper_points_added.append(clip_index)
            # If this point is inbound, then part of the clipper is inside the subject
            # so we need to follow the clipper for a bit
            if inbound[clip_index]:
                while True:
                    clip_index += 1
                    if clip_index == len(clipper):
                        clip_index = 0
                    intersection.append(clipper[clip_index])
                    clipper_points_added.append(clip_index)
                    # If we've reached a non-inbound point that lies on the subject,
                    # then exit this loop
                    if not inbound[clip_index]:
                        for k, s in enumerate(subject):
                            if abs(s[0] - clipper[clip_index][0]) < tol and abs(s[1] - clipper[clip_index][1]) < tol:
                                subj_index = k
                                k = -1
                                break
                        if k == -1:
                            break
        subj_index += 1
        if subj_index == len(subject):
            subj_index = 0
        solved = True
        # Check for inifinite loop; exit condition is if every segment of the
        # clipping polygon clipped every segment of the subject polygon; this is
        # certainly an overestimate of what's possible, but it's sufficient to prevent
        # an infinite loop
        if len(intersection) > n1 + n2 + n1 * n2:
            raise ValueError("Infinite loop encountered during polygon intersection!")

    # Because the loop had to start in the middle of subject, the start & end
    # points of intersection may not be at the start & end, so reorder:
    itsct_index = -1
    for i, p in enumerate(intersection):
        if abs(p[0] - intersection[i - 1][0]) < tol and abs(p[1] - intersection[i - 1][1]) < tol:
            itsct_index = i
            break

    # Found consecutive points that are the same and not at the start of the shape
    if itsct_index > 0:
        start_subj_index = 0
        intersection = intersection[itsct_index:] + intersection[:itsct_index]
    # If we didn't start with the first index, we might need to "close the loop" by appending
    # the first element again
    elif start_subj_index != 0:
        intersection.append(intersection[0])

    return intersection


def polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """Calculates the area of a polygon using the shoelace theorem.

    Parameters
    ----------
    polygon : List[Tuple[float, float]]
        The list of points defining the polygon.

    Returns
    -------
    float
        The area of the polygon

    Notes
    -----
    Assumptions:
        1. The vertexes are ordered around the perimeter (clockwise vs. counter-clockwise does not matter)
        2. The first vertex of the polygon is the same as the last one (explicitly close the polygon)
    """

    area = 0.0
    for i in range(1, len(polygon)):
        x = polygon[1][0] if i == len(polygon) - 1 else polygon[i + 1][0]
        area += polygon[i][1] * (polygon[i - 1][0] - x)
    return abs(area) * 0.5


def hexagon_area(
    apothem: float,
    array_coords: Tuple[int, int] = None,
    rings: int = 0,
    outer_apothem: float = 0.0,
    geom_type: GeomType = GeomType.hex_pointy_top,
    outer_geom_type: GeomType = GeomType.hex_flat_top,
) -> float:
    """Calculates the area of a hexagon.

    Parameters
    ----------
    apothem : float
        The hexagon apothem.
    array_coords : Tuple[int, int], optional
        The location of the hexagon in an array. Defaults to None.
    rings : int, optional
        The number of rings in the hexagonal array. Defaults to 0.
    outer_apothem : float, optional
        The apothem of the parent hexagon containing the hexagonal array.
    geom_type : GeomType, optional
        The orientation of the hexagon. Defaults to GeomType.hex_pointy_top.
    outer_geom_type : GeomType, optional
        The orientation of the parent hexagon containing the hexagonal array. Defaults to GeomType.hex_flat_top.

    Returns
    -------
    float
        The area of the hexagon.

    Notes
    -----
    If rings, outer_apothem, and geom_type are specified, then clipping of the hexagon
    by a larger parent hexagon is considered. Otherwise, the basic geometric formula
    for a hexagon's area is employed.
    """

    if not array_coords or outer_apothem >= parent_hex_min_pitch(rings, apothem * 2.0) * 0.5:
        return 2.0 * np.sqrt(3) * np.power(apothem, 2.0)
    assert rings > 0
    assert outer_apothem > 0.0
    assert geom_type in GeomType.hex

    # Get the vertexes for each hexagon and use them to calculate the area
    centroid = hex_element_centroid(apothem, array_coords, rings, geom_type)
    small_hex = hexagon_points(apothem, geom_type, centroid)
    large_hex = hexagon_points(outer_apothem, outer_geom_type)
    intersection = polygon_intersection(small_hex, large_hex)
    return polygon_area(intersection)


def num_rings_from_hex_map(map_: List[List[Any]]) -> int:
    """
    Calculates the number of rings from a hexagonal map.

    Parameters
    ----------
    map_ : List[List[Any]]
        The hexagonal map to check.

    Returns
    -------
    rings : int
        The number of rings in the map.

    Notes
    -----
    Also does some error checking to ensure the map has a valid shape.
    """
    assert len(map_) % 2 == 1
    rings = round((len(map_) + 1) / 2)
    for i in range(rings):
        assert len(map_[i]) == rings + i, f"row index {i}, length = {len(map_[i])}, should be {rings + i}"
        assert len(map_[-1 - i]) == rings + i, f"row index {-1-i}, length = {len(map_[-1-i])}, should be {rings + i}"
    return rings


def num_units_in_hex_ring(ring: int) -> int:
    """
    Calculates the number of units in a particular ring of a hexagonal array.

    Parameters
    ----------
    ring : int
        The ring index to count.

    Returns
    -------
    int
        The number of units in the ring.
    """
    assert ring > 0
    return 1 if ring == 1 else 6 * (ring - 1)


def num_units_from_hex_rings(rings: int) -> int:
    """Calculates the total number of hexagonal array elements given a number of rings.

    Parameters
    ----------
    rings : int
        The number of rings in the hexagonal array.

    Returns
    -------
    int
        The total number of hexagons in the array.

    Notes
    -----
    A single hexagon is considered to be an array of 1 ring.
    """

    assert rings > 0
    return 3 * rings * (rings - 1) + 1


def parent_hex_min_pitch(
    rings: int, pitch: float, parent_geom: GeomType = GeomType.hex_flat_top, geom: GeomType = GeomType.hex_pointy_top
) -> float:
    """Calculates the minimum pitch of a hexagonal bounding box to avoid clipping a hexagonal array

    Parameters
    ----------
    rings : int
        The number of rings in the array.
    pitch : float
        The pitch (flat-to-flat distance) of the hexagons.
    parent_geom : GeomType, optional
        The orientation of the parent hexagon containing the array. Defaults to GeomType.hex_flat_top.
    geom : GeomType, optional
        The orientation of the hexagonal array. Defaults to GeomType.hex_pointy_top.

    Returns
    -------
    float
        The minimum pitch of the parent hexagon to avoid clipping the array.

    Notes
    -----
    A single hexagon is considered an array of 1 ring.
    """

    assert rings > 0
    assert pitch > 0.0
    point_to_point = pitch * 2.0 / np.sqrt(3.0)
    if parent_geom == geom:
        parent_pitch = (2 * rings - 1) * pitch
    else:
        # Add the innermost hex first
        parent_pitch = point_to_point
        for i in range(1, rings):
            # Odd number rings: add the side length for the top & bottom of the ring
            if i % 2:
                parent_pitch += point_to_point  # really it's point_to_point/2 * 2
                # If this is also last ring, add the height of the jagged edge for the top & bottom of the ring
                if i == rings - 1:
                    parent_pitch += point_to_point * 0.5  # really point_to_point/4 * 2
            # Even number rings: add the point-to-point for the top & bottom of the ring
            else:
                parent_pitch += point_to_point * 2.0
                # If this is the last ring, we've reached the uppermost point, so nothing else is needed

    return parent_pitch


def hex_circle_max_radius(rings: int, pitch: float) -> float:
    """Calculates the maximum circle radius that fits completely inside a hexagonal array.

    Parameters
    ----------
    rings : int
        The number of rings in the hexagonal array
    pitch : float
        The pitch (flat-to-flat distance) of the hexagons.

    Returns
    -------
    float
        The maximum radius of a circle that does not clip outside the array.

    Notes
    -----
    The center of the circle is always assumed to be at the center of the center hexagon.

    A single hexagon is considered an array of 1 ring.
    """

    side_length = pitch / np.sqrt(3.0)
    if rings % 2 == 0:
        return ((rings / 2 - 1) * 3 + 2) * side_length
    return np.sqrt(np.power(((((rings + 1) / 2) - 1) * 3 + 0.5) * side_length, 2.0) + np.power(side_length, 2.0))


def hex_nrings_fill_circle(radius: float, pitch: float, rings_init: int = 1) -> int:
    """Calculates the maximum number of rings in a hexagonal array without clipping a circle of a given radius

    Parameters
    ----------
    radius : float
        The radius of the circle to fill.
    pitch : float
        The pitch (flat-to-flat distance) of the hexagons.
    rings_init : int, optional
        The initial guess for the number of rings. Default is 1.

    Returns
    -------
    int
        The maximum number of hexagonal array rings that can fit inside the circle without clipping it.

    Notes
    -----
    A single hexagon is considered an array of 1 ring.

    If a solution is not found, a RuntimeError is raised.
    """
    assert radius > 0.0
    assert pitch > 0.0
    assert rings_init > 0
    for rings in range(rings_init, 1000):
        check_rad = hex_circle_max_radius(rings, pitch)
        if check_rad >= radius:
            return rings

    raise RuntimeError(
        "Could not find a number of rings (pitch={0:g}) to fill circle (radius={1:g})! Only 1000 rings are searched!"
    )


def hex_circle_min_radius(rings: int, pitch: float) -> float:
    """Calculates the minimum circle radius that does not clip a hexagonal array.

    Parameters
    ----------
    rings : int
        The number of rings in the hexagonal array.
    pitch : float
        The pitch (flat-to-flat distance) of the hexagons.

    Returns
    -------
    float
        The minimum circle radius that is completely outside the hexagonal array without clipping it.

    Notes
    -----
    A single hexagon is considered an array of 1 ring.
    """
    x = pitch / np.sqrt(3)
    y = (2 * rings - 1) * pitch * 0.5
    return np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))


def count_rings(length: int) -> int:
    """Counts the number of rings in a hexagonal array given a total number of elements

    Parameters
    ----------
    length : int
        The total number of elements in the hexagonal array.

    Returns
    -------
    int
        The number of rings in the array.

    Notes
    -----
    A single hexagon is considered an array of 1 ring.

    Arrays with more than 24 rings cannot be tested due to integer overflow.

    If a value cannot be calculated, a ValueError is raised.
    """

    old = (-1, -1)
    new = (0, 0)
    for i in range(24):  # overflow for i > 24
        old = new
        new = (i + 1, num_units_from_hex_rings(i + 1))
        if length == new[1]:
            return i + 1
        if old[1] < length < new[1]:
            break
    raise ValueError(
        f"The input hex map of length {length} could not have its number of rings determined!  Between {old} and {new}"
    )


def reshape_hex_map(map_: Iterable, dtype: np.dtype = int) -> np.array:
    """Reshapes a 1D hexagonal array into a 2D hexagonal array.

    Parameters
    ----------
    map_ : List[dtype]
        The list of hexagonal array elements.
    dtype : np.dtype, optional
        The data type of the elements of the list. Defaults to int.

    Returns
    -------
    np.array
        The 2D array containing the hexagonal elements.
    """

    rings = count_rings(len(map_))

    map_index = 0
    new_map = np.zeros([2 * rings - 1, 2 * rings - 1], dtype=dtype)
    for ring in range(rings):
        for index in range(rings + ring):
            new_map[ring, index] = map_[map_index]
            map_index += 1
    for ring in range(rings, 2 * rings - 1):
        for index in range(3 * rings - ring - 2):
            new_map[ring, index] = map_[map_index]
            map_index += 1

    return new_map


def add_hex_ring(map_: List[Any], new_ring: List[Any]) -> List[Any]:
    """Adds a new ring around the outside of a 1D hexagonal array.

    Parameters
    ----------
    map_ : List[Any]
        The list of hexagonal array elements
    new_ring : List[Any]
        The list of array elements to add around the outside of the prevoius array.

    Returns
    -------
    List[Any]
        The new list of hexagonal array elements with the extra ring added.

    Notes
    -----
    An example is given below:

    list = [a1, a2,
          b1, b2, b3,
            c1, c2]
    new_ring = [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12]
    result = [n1, n2, n3,
           n12, a1, a2, n4,
         n11, b1, b2, b3, n5,
           n10, c1, c2, n6,
            n9, n8, n7]
    """

    old_rings = count_rings(len(map_))
    new_rings = count_rings(len(map_) + len(new_ring))
    assert new_rings == old_rings + 1
    new_map = []
    new_ring_local = list(new_ring)

    # Add the top row first
    for _ in range(new_rings):
        new_map.append(new_ring_local.pop(0))
    # Add top half & northeast edge
    map_index = 0
    for ring in range(old_rings):
        new_map.append(new_ring_local.pop(-1))
        for _ in range(old_rings + ring):
            new_map.append(map_[map_index])
            map_index += 1
        new_map.append(new_ring_local.pop(0))
    # Add bottom half & southeast edge
    for ring in range(old_rings, 2 * old_rings - 1):
        new_map.append(new_ring_local.pop(-1))
        for _ in range(3 * old_rings - ring - 2):
            new_map.append(map_[map_index])
            map_index += 1
        new_map.append(new_ring_local.pop(0))
    # Add bottom row
    for _ in range(new_rings):
        new_map.append(new_ring_local.pop(-1))

    return new_map


def location_in_hex_array(
    point: Tuple[float, float], rings: int, apothem: float, geom_type: GeomType, centroid: Tuple[float, float] = (0.0, 0.0)
) -> Tuple[int, int]:
    """Returns the array coordinates of the hexagon containing a point.

    Parameters
    ----------
    point : Tuple[float, float]
        The (x,y) coordinates of the point to test.
    rings : int
        The number of rings in the hexagonal array.
    apothem : float
        The apothem of the hexagons.
    geom_type : GeomType
        The orientation of the hexagons.
    centroid : Tuple[float, float], optional
        The (x,y) location of the hexagonal array centroid. Default is (0.0, 0.0).

    Returns
    -------
    Tuple[int, int]
        The (row, column) indexes of the hexagon containing the point.

    Notes
    -----
    A single hexagon is considered to be an array of 1 ring.
    """
    assert rings > 0
    assert apothem > 0.0

    old_centroid = centroid
    hexagon = hexagon_points(apothem, geom_type, centroid)
    for row in range(2 * rings - 1):
        for col in range(2 * rings - 1 - abs(row + 1 - rings)):
            hex_centroid = hex_element_centroid(apothem, (row, col), rings, geom_type)
            new_centroid = (hex_centroid[0] + centroid[0], hex_centroid[1] + centroid[1])
            hexagon = [(p[0] - old_centroid[0] + new_centroid[0], p[1] - old_centroid[1] + new_centroid[1]) for p in hexagon]
            if point_in_polygon(point, hexagon) < 1:  # Accept both on the boundary & inside
                return (row, col)
            old_centroid = new_centroid
    return -1, -1


def nrings_from_2d_map(map_: List[List[Any]]) -> int:
    """Calculate the number of rings in a 2D hexagonal array map.

    Parameters
    ----------
    map_ : List[List[str]]
        2D map array with string identifiers for the hexagonal array.

    Returns
    -------
    int
        Number of rings in the hexagonal array.

    Notes
    -----
    A single hexagon is considered to be an array of 1 ring.
    """
    if len(map_) == 0:
        return 0
    if len(map_) == 1:
        return 1
    return (len(map_) - 1) // 2


def add_rings_to_2d_map(map_: List[List[Any]], id_: Any, n_new_rings: int) -> List[List[Any]]:
    """Add a ring to a 2D hexagonal array map.

    Parameters
    ----------
    map_ : List[List[str]]
        2D map array with string identifiers for the hexagonal array.
    id_ : str
        The element to repeatedly add as new rings around the array.
    n_new_rings : int
        Number of new rings to add to the map.

    Returns
    -------
    List[List[str]]
        Updated hexagonal array map.
    """
    new_map = map_.copy()
    for _ in range(n_new_rings):
        new_row = [id_] * (len(new_map[0]) + 1)
        for row in new_map:
            row.insert(0, id_)
            row.append(id_)
        new_map.insert(0, new_row)
        new_map.append(new_row.copy())

    return new_map
