## ---------------------------------------------------------------- ##
## SPACE.PY
## ---------------------------------------------------------------- ##
## Contains some useful primitives to work in 3D space.
## ---------------------------------------------------------------- ##

__all__ = ['Volume', 'Point', 'Sphere', 'distance', 'sphere_volume']

from ..basic import SPyNEObject

class Volume(SPyNEObject):
    """A representations of a 3D volume in space"""
    def __init__(self, width=0, height=0, depth=0):
        self.width  = width
        self.height = height
        self.depth  = depth


## --- POINT ---------------------------------------------------------

def distance(p1, p2):
    """
    Returns the Euclidean distance between two 3D points.
    """
    if p1 is not None and p2 is not None:
        return sqrt((p1.x - p2.x) ** 2 +
                    (p1.y - p2.y) ** 2 +
                    (p2.z - p2.z) ** 2)

def point_sum(p1, p2):
    """Returns the sum of two points (or vectors)"""
    if p1 is not None and p2 is not None:
        return Point(p1.x + p2.x,
                     p1.y + p2.y,
                     p1.z + p2.z)

def point_diff(p1, p2):
    """Returns the difference of two points (or vectors)"""
    if p1 is not None and p2 is not None:
        return Point(p1.x - p2.x,
                     p1.y - p2.y,
                     p1.z - p2.z)

class Point(SPyNEObject):
    """
    An object representing a point in 3D space. Points can be seen as
    locations in a coordinate system or a vector in a 3D space.
    As vectors, points have a number of possible arithmetic operations
    that are defined, such as addition, subtration, and
    multiplication.
    """
    def __init__(self, x=0, y=0, z=0):
        SPyNEObject.__init__(self)
        self.x = x
        self.y = y
        self.z = z
        self.DistanceFrom = distance

    def __repr__(self):
        """Visual representation of a Point as 'P{x, y, z}'"""
        return "P{%s, %s, %s}" % (self.x, self.y, self.z)

    def __str__(self):
        """Visual representation of a Point as 'P{x, y, z}'"""
        return self.__repr__()

    def __eq__(self, other):
        """Checks whether two points are equal"""
        if other is None:
            return False
        if  other.__class__ != self.__class__:
            return False
        if  other.x == self.x and \
            other.y == self.y and \
            other.z == self.z:
            return True
        else:
            return False

    def __ne__(self, other):
        """Checks whether two points are equal"""
        return not self.__eq__(other)


    def __add__(self, other):
        """Sums two points"""
        return point_sum(self, other)

    def __sub__(self, other):
        """Subtracts two points"""
        return point_diff(self, other)

    def __mul__(self, other):
        """Multiply two points (as vectors)"""
        return Point(self.y*other.z - self.z*other.y,
                     self.z*other.x - self.x*other.z,
                     self.x*other.y - self.y*other.x)

    def Translate(self, x, y, z):
        """Translates a point in 3D space"""
        self.x += x
        self.y += y
        self.z += z
        self.Notify()

    def Update(self, x, y, z):
        """Updates the position of a point in 3D space"""
        self.x = x
        self.y = y
        self.z = y
        self.Notify()

class Sphere(Point):
    """
    An object representing a sphere in 3D space
    A sphere is defined by the 3D coordinates of
    its center (x, y, z) and its radius 'r'
    """
    def __init__(self, x=0, y=0, z=0, r=0):
        Point.__init__(self, x, y, z)
        self.r = r
        self.Volume = sphere_volume
        

def sphere_volume(sphere):
    """
    Returns the volume of a sphere 
    """
    return (pi * r**3) * 4/3
