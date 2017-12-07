#!/usr/bin/env python

import os
import sys
import time
import math
import numpy
import struct
import numbers
import argparse
import platform as plat
import subprocess

from collections import namedtuple
from pyquaternion import Quaternion

try:
    from itertools import zip_longest as ziplong
except ImportError:
    from itertools import izip_longest as ziplong

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except:
    print(''' Error PyOpenGL not installed properly !!''')
    sys.exit(  )


def float_fmt(val):
    """
    Returns a short, clean floating point string representation.
    Unnecessary trailing zeroes and decimal points are trimmed off.
    """
    s = "{0:.6f}".format(val).rstrip('0').rstrip('.')
    return s if s != '-0' else '0'


class Vector(object):
    """Class to represent an N dimentional vector."""

    def __init__(self, *args):
        self._values = []
        if len(args) == 1:
            val = args[0]
            if isinstance(val, numbers.Real):
                self._values = [val]
                return
            elif isinstance(val, numbers.Complex):
                self._values = [val.real, val.imag]
                return
        else:
            val = args
        try:
            for x in val:
                if not isinstance(x, numbers.Real):
                    raise TypeError('Expected sequence of real numbers.')
                self._values.append(x)
        except:
            pass

    def __iter__(self):
        """Iterator generator for vector values."""
        for idx in self._values:
            yield idx

    def __len__(self):
        return len(self._values)

    def __getitem__(self, idx):
        """Given a vertex number, returns a vertex coordinate vector."""
        return self._values[idx]

    def __hash__(self):
        """Returns hash value for vector coords"""
        return hash(tuple(self._values))

    def __eq__(self, other):
        """Equality comparison for points."""
        return self._values == other._values

    def __cmp__(self, other):
        """Compare points for sort ordering in an arbitrary heirarchy."""
        longzip = ziplong(self._values, other, fillvalue=0.0)
        for v1, v2 in reversed(list(longzip)):
            val = cmp(v1, v2)
            if val != 0:
                return val
        return 0

    def __sub__(self, v):
        return Vector(i - j for i, j in zip(self._values, v))

    def __rsub__(self, v):
        return Vector(i - j for i, j in zip(v, self._values))

    def __add__(self, v):
        return Vector(i + j for i, j in zip(self._values, v))

    def __radd__(self, v):
        return Vector(i + j for i, j in zip(v, self._values))

    def __div__(self, s):
        """Divide each element in a vector by a scalar."""
        return Vector(x / (s+0.0) for x in self._values)

    def __mul__(self, s):
        """Multiplies each element in a vector by a scalar."""
        return Vector(x * s for x in self._values)

    def __format__(self, fmt):
        vals = [float_fmt(x) for x in self._values]
        if "a" in fmt:
            return "[{0}]".format(", ".join(vals))
        if "s" in fmt:
            return " ".join(vals)
        if "b" in fmt:
            return struct.pack('<{0:d}f'.format(len(self._values)), *self._values)
        return "({0})".format(", ".join(vals))

    def __repr__(self):
        return "<Vector: {0}>".format(self)

    def __str__(self):
        """Returns a standard array syntax string of the coordinates."""
        return "{0:a}".format(self)

    def dot(self, v):
        """Dot (scalar) product of two vectors."""
        return sum(p*q for p, q in zip(self, v))

    def cross(self, v):
        """
        Cross (vector) product against another 3D Vector.
        Returned 3D Vector will be perpendicular to both original 3D Vectors.
        """
        return Vector(
            self._values[1]*v[2] - self._values[2]*v[1],
            self._values[2]*v[0] - self._values[0]*v[2],
            self._values[0]*v[1] - self._values[1]*v[0]
        )

    def length(self):
        """Returns the length of the vector."""
        return math.sqrt(sum(x*x for x in self._values))

    def normalize(self):
        """Normalizes the given vector to be unit-length."""
        return self / self.length()

    def angle(self, other):
        """Returns angle in radians between this and another vector."""
        return math.acos(self.dot(other) / (self.length() * other.length()))


class Point3D(object):
    """Class to represent a 3D Point."""

    def __init__(self, *args):
        self._values = [0.0, 0.0, 0.0]
        if len(args) == 1:
            val = args[0]
            if isinstance(val, numbers.Real):
                self._values = [val, 0.0, 0.0]
                return
            elif isinstance(val, numbers.Complex):
                self._values = [val.real, val.imag, 0.0]
                return
        else:
            val = args
        try:
            for i, x in enumerate(val):
                if not isinstance(x, numbers.Real):
                    raise TypeError('Expected sequence of real numbers.')
                self._values[i] = x
        except:
            pass

    def __iter__(self):
        """Iterator generator for point values."""
        for idx in range(3):
            yield self[idx]

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        """Given a vertex number, returns a vertex coordinate vector."""
        if idx >= len(self._values):
            return 0.0
        return self._values[idx]

    def __hash__(self):
        """Returns hash value for point coords"""
        return hash(tuple(self._values))

    def __cmp__(self, p):
        """Compare points for sort ordering in an arbitrary heirarchy."""
        longzip = ziplong(self._values, p, fillvalue=0.0)
        for v1, v2 in reversed(list(longzip)):
            val = v1 - v2
            if val != 0:
                val /= abs(val)
                return val
        return 0

    def __eq__(self, other):
        """Equality comparison for points."""
        return self._values == other._values

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

    def __sub__(self, v):
        return Point3D(self[i] - v[i] for i in range(3))

    def __rsub__(self, v):
        return Point3D(v[i] - self[i] for i in range(3))

    def __add__(self, v):
        return Vector(i + j for i, j in zip(self._values, v))

    def __radd__(self, v):
        return Vector(i + j for i, j in zip(v, self._values))

    def __div__(self, s):
        """Divide each element in a vector by a scalar."""
        return Vector(x / s for x in self._values)

    def __format__(self, fmt):
        vals = [float_fmt(x) for x in self._values]
        if "a" in fmt:
            return "[{0}]".format(", ".join(vals))
        if "s" in fmt:
            return " ".join(vals)
        if "b" in fmt:
            return struct.pack('<3f', *self._values)
        return "({0})".format(", ".join(vals))

    def __repr__(self):
        return "<Point3D: {0}>".format(self)

    def __str__(self):
        """Returns a standard array syntax string of the coordinates."""
        return "{0:a}".format(self)

    def distFromPoint(self, v):
        """Returns the distance from another point."""
        return math.sqrt(sum(math.pow(x1-x2, 2.0) for x1, x2 in zip(v, self)))

    def distFromLine(self, pt, line):
        """
        Returns the distance of a 3d point from a line defined by a sequence
        of two 3d points.
        """
        w = Vector(pt - line[0])
        v = Vector(line[1]-line[0])
        return v.normalize().cross(w).length()


class Point3DCache(object):
    """Cache class for 3D Points."""

    def __init__(self):
        """Initialize as an empty cache."""
        self.point_hash = {}
        self.minx = 9e99
        self.miny = 9e99
        self.minz = 9e99
        self.maxx = -9e99
        self.maxy = -9e99
        self.maxz = -9e99

    def __len__(self):
        """Length of sequence."""
        return len(self.point_hash)

    def _update_volume(self, p):
        """Update the volume cube that contains all the points."""
        if p[0] < self.minx:
            self.minx = p[0]
        if p[0] > self.maxx:
            self.maxx = p[0]
        if p[1] < self.miny:
            self.miny = p[1]
        if p[1] > self.maxy:
            self.maxy = p[1]
        if p[2] < self.minz:
            self.minz = p[2]
        if p[2] > self.maxz:
            self.maxz = p[2]

    def get_volume(self):
        """Returns the 3D volume that contains all the points in the cache."""
        return (
            self.minx, self.miny, self.minz,
            self.maxx, self.maxy, self.maxz
        )

    def add(self, x, y, z):
        """Given XYZ coords, returns the (new or cached) Point3D instance."""
        key = tuple(round(n, 4) for n in [x, y, z])
        if key in self.point_hash:
            return self.point_hash[key]
        pt = Point3D(key)
        self.point_hash[key] = pt
        self._update_volume(pt)
        return pt

    def __iter__(self):
        """Creates an iterator for the points in the cache."""
        for pt in self.point_hash.values():
            yield pt


class LineSegment3D(object):
    """A class to represent a 3D line segment."""

    def __init__(self, p1, p2):
        """Initialize with twwo endpoints."""
        if p1 > p2:
            p1, p2 = (p2, p1)
        self.p1 = p1
        self.p2 = p2
        self.count = 1

    def __len__(self):
        """Line segment always has two endpoints."""
        return 2

    def __iter__(self):
        """Iterator generator for endpoints."""
        yield self.p1
        yield self.p2

    def __getitem__(self, idx):
        """Given a vertex number, returns a vertex coordinate vector."""
        if idx == 0:
            return self.p1
        if idx == 1:
            return self.p2
        raise LookupError()

    def __hash__(self):
        """Returns hash value for endpoints"""
        return hash((self.p1, self.p2))

    def __cmp__(self, p):
        """Compare points for sort ordering in an arbitrary heirarchy."""
        val = cmp(self[0], p[0])
        if val != 0:
            return val
        return cmp(self[1], p[1])

    def __format__(self, fmt):
        """Provides .format() support."""
        pfx = ""
        sep = " - "
        sfx = ""
        if "a" in fmt:
            pfx = "["
            sep = ", "
            sfx = "]"
        elif "s" in fmt:
            pfx = ""
            sep = " "
            sfx = ""
        p1 = self.p1.__format__(fmt)
        p2 = self.p2.__format__(fmt)
        return pfx + p1 + sep + p2 + sfx

    def __repr__(self):
        """Standard string representation."""
        return "<LineSegment3D: {0}>".format(self)

    def __str__(self):
        """Returns a human readable coordinate string."""
        return "{0:a}".format(self)

    def length(self):
        """Returns the length of the line."""
        return self.p1.distFromPoint(self.p2)


class LineSegment3DCache(object):
    """Cache class for 3D Line Segments."""

    def __init__(self):
        """Initialize as an empty cache."""
        self.endhash = {}
        self.seghash = {}

    def _add_endpoint(self, p, seg):
        if p not in self.endhash:
            self.endhash[p] = []
        self.endhash[p].append(seg)

    def endpoint_segments(self, p):
        if p not in self.endhash:
            return []
        return self.endhash[p]

    def get(self, p1, p2):
        """Given 2 endpoints, return the cached LineSegment3D inst, if any."""
        key = (p1, p2) if p1 < p2 else (p2, p1)
        if key not in self.seghash:
            return None
        return self.seghash[key]

    def add(self, p1, p2):
        """Given 2 endpoints, return the (new or cached) LineSegment3D inst."""
        key = (p1, p2) if p1 < p2 else (p2, p1)
        if key in self.seghash:
            seg = self.seghash[key]
            seg.count += 1
            return seg
        seg = LineSegment3D(p1, p2)
        self.seghash[key] = seg
        self._add_endpoint(p1, seg)
        self._add_endpoint(p2, seg)
        return seg

    def __iter__(self):
        """Creates an iterator for the line segments in the cache."""
        for pt in self.seghash.values():
            yield pt

    def __len__(self):
        """Length of sequence."""
        return len(self.seghash)


class Facet3D(object):
    """Class to represent a 3D triangular face."""

    def __init__(self, v1, v2, v3, norm):
        for x in [v1, v2, v3, norm]:
            try:
                n = len(x)
            except:
                n = 0
            if n != 3:
                raise TypeError('Expected 3D vector.')
            for y in x:
                if not isinstance(y, numbers.Real):
                    raise TypeError('Expected 3D vector.')
        verts = [
            Point3D(v1),
            Point3D(v2),
            Point3D(v3)
        ]
        # Re-order vertices in a normalized order.
        while verts[0] > verts[1] or verts[0] > verts[2]:
            verts = verts[1:] + verts[:1]
        self.vertices = verts
        self.norm = Vector(norm)
        self.count = 1
        self.fixup_normal()

    def __len__(self):
        """Length of sequence.  Three vertices and a normal."""
        return 4

    def __getitem__(self, idx):
        """Get vertices and normal by index."""
        lst = self.vertices + [self.norm]
        return lst[idx]

    def __hash__(self):
        """Returns hash value for facet"""
        return hash((self.verts, self.norm))

    def __cmp__(self, other):
        """Compare faces for sorting in an arbitrary heirarchy."""
        cl1 = [sorted(v[i] for v in self.vertices) for i in range(3)]
        cl2 = [sorted(v[i] for v in other.vertices) for i in range(3)]
        for i in reversed(range(3)):
            for c1, c2 in ziplong(cl1[i], cl2[i]):
                if c1 is None:
                    return -1
                val = cmp(c1, c2)
                if val != 0:
                    return val
        return 0

    def __format__(self, fmt):
        """Provides .format() support."""
        pfx = ""
        sep = " - "
        sfx = ""
        if "a" in fmt:
            pfx = "["
            sep = ", "
            sfx = "]"
        elif "s" in fmt:
            pfx = ""
            sep = " "
            sfx = ""
        ifx = sep.join(n.__format__(fmt) for n in list(self)[0:3])
        return pfx + ifx + sfx

    def is_clockwise(self):
        """
        Returns true if the three vertices of the face are in clockwise
        order with respect to the normal vector.
        """
        v1 = Vector(self.vertices[1]-self.vertices[0])
        v2 = Vector(self.vertices[2]-self.vertices[0])
        return self.norm.dot(v1.cross(v2)) < 0

    def fixup_normal(self):
        if self.norm.length() > 0:
            # Make sure vertex ordering is counter-clockwise,
            # relative to the outward facing normal.
            if self.is_clockwise():
                self.vertices = [
                    self.vertices[0],
                    self.vertices[2],
                    self.vertices[1]
                ]
        else:
            # If no normal was specified, we should calculate it, relative
            # to the counter-clockwise vertices (as seen from outside).
            v1 = Vector(self.vertices[2] - self.vertices[0])
            v2 = Vector(self.vertices[1] - self.vertices[0])
            self.norm = v1.cross(v2)
            if self.norm.length() > 1e-6:
                self.norm = self.norm.normalize()


class Facet3DCache(object):
    """Cache class for 3D Facets."""

    def __init__(self):
        """Initialize as an empty cache."""
        self.vertex_hash = {}
        self.edge_hash = {}
        self.facet_hash = {}

    def _add_vertex(self, pt, facet):
        """Remember that a given vertex touches a given facet."""
        if pt not in self.vertex_hash:
            self.vertex_hash[pt] = []
        self.vertex_hash[pt].append(facet)

    def _add_edge(self, p1, p2, facet):
        """Remember that a given edge touches a given facet."""
        if p1 > p2:
            edge = (p1, p2)
        else:
            edge = (p2, p1)
        if edge not in self.edge_hash:
            self.edge_hash[edge] = []
        self.edge_hash[edge].append(facet)

    def vertex_facets(self, pt):
        """Returns the facets that have a given facet."""
        if pt not in self.vertex_hash:
            return []
        return self.vertex_hash[pt]

    def edge_facets(self, p1, p2):
        """Returns the facets that have a given edge."""
        if p1 > p2:
            edge = (p1, p2)
        else:
            edge = (p2, p1)
        if edge not in self.edge_hash:
            return []
        return self.edge_hash[edge]

    def get(self, p1, p2, p3):
        """Given 3 vertices, return the cached Facet3D instance, if any."""
        key = (p1, p2, p3)
        if key not in self.facet_hash:
            return None
        return self.facet_hash[key]

    def add(self, p1, p2, p3, norm):
        """
        Given 3 vertices and a norm, return the (new or cached) Facet3d inst.
        """
        key = (p1, p2, p3)
        if key in self.facet_hash:
            facet = self.facet_hash[key]
            facet.count += 1
            return facet
        facet = Facet3D(p1, p2, p3, norm)
        self.facet_hash[key] = facet
        self._add_edge(p1, p2, facet)
        self._add_edge(p2, p3, facet)
        self._add_edge(p3, p1, facet)
        self._add_vertex(p1, facet)
        self._add_vertex(p2, facet)
        self._add_vertex(p3, facet)
        return facet

    def sorted(self):
        """Returns a sorted iterator."""
        vals = self.facet_hash.values()
        for pt in sorted(vals):
            yield pt

    def __iter__(self):
        """Creates an iterator for the facets in the cache."""
        for pt in self.facet_hash.values():
            yield pt

    def __len__(self):
        """Length of sequence."""
        return len(self.facet_hash)


class StlEndOfFileException(Exception):
    """Exception class for reaching the end of the STL file while reading."""
    pass


class StlMalformedLineException(Exception):
    """Exception class for malformed lines in the STL file being read."""
    pass


class StlData(object):
    """Class to read, write, and validate STL file data."""

    def __init__(self):
        """Initialize with empty data set."""
        self.points = Point3DCache()
        self.edges = LineSegment3DCache()
        self.facets = Facet3DCache()
        self.filename = ""
        self.dupe_faces = []
        self.dupe_edges = []
        self.hole_edges = []
        self.wireframe = False
        self.show_facets = True
        self.perspective = True
        self.boundsrad = 1.0
        self.cx, self.cy, self.cz = 0.0, 0.0, 0.0
        self.width, self.height = 800, 600
        self._xstart, self._ystart = 0, 0
        self._model_list = None
        self._errs_list = None
        self._grid_list = None
        self._mouse_btn = GLUT_LEFT_BUTTON
        self._mouse_state = GLUT_UP
        self._action = None
        self.reset_view()

    def reset_view(self):
        self._view_q = Quaternion(axis=[0, 0, 1], degrees=25)
        self._view_q *= Quaternion(axis=[1, 0, 0], degrees=55)
        self._xtrans, self._ytrans= 0.0, 0.0
        self._zoom = 1.0

    def _read_ascii_line(self, f, watchwords=None):
        line = f.readline(1024).decode('utf-8')
        if line == "":
            raise StlEndOfFileException()
        words = line.strip(' \t\n\r').lower().split()
        if not words:
            return []
        if words[0] == 'endsolid':
            raise StlEndOfFileException()
        argstart = 0
        if watchwords:
            watchwords = watchwords.lower().split()
            argstart = len(watchwords)
            for i in range(argstart):
                if words[i] != watchwords[i]:
                    raise StlMalformedLineException()
        return [float(val) for val in words[argstart:]]

    def _read_ascii_vertex(self, f):
        point = self._read_ascii_line(f, watchwords='vertex')
        return self.points.add(*point)

    def _read_ascii_facet(self, f):
        while True:
            try:
                normal = self._read_ascii_line(f, watchwords='facet normal')
                self._read_ascii_line(f, watchwords='outer loop')
                vertex1 = self._read_ascii_vertex(f)
                vertex2 = self._read_ascii_vertex(f)
                vertex3 = self._read_ascii_vertex(f)
                self._read_ascii_line(f, watchwords='endloop')
                self._read_ascii_line(f, watchwords='endfacet')
                if vertex1 == vertex2:
                    continue  # zero area facet.  Skip to next facet.
                if vertex2 == vertex3:
                    continue  # zero area facet.  Skip to next facet.
                if vertex3 == vertex1:
                    continue  # zero area facet.  Skip to next facet.
            except StlEndOfFileException:
                return None
            except StlMalformedLineException:
                continue  # Skip to next facet.
            self.edges.add(vertex1, vertex2)
            self.edges.add(vertex2, vertex3)
            self.edges.add(vertex3, vertex1)
            return self.facets.add(vertex1, vertex2, vertex3, normal)

    def _read_binary_facet(self, f):
        data = struct.unpack('<3f 3f 3f 3f H', f.read(4*4*3+2))
        normal = data[0:3]
        vertex1 = data[3:6]
        vertex2 = data[6:9]
        vertex3 = data[9:12]
        v1 = self.points.add(*vertex1)
        v2 = self.points.add(*vertex2)
        v3 = self.points.add(*vertex3)
        self.edges.add(v1, v2)
        self.edges.add(v2, v3)
        self.edges.add(v3, v1)
        return self.facets.add(v1, v2, v3, normal)

    def read_file(self, filename):
        self.filename = filename
        with open(filename, 'rb') as f:
            line = f.readline(80)
            if line == "":
                return  # End of file.
            if line[0:6].lower() == b"solid ":
                # Reading ASCII STL file.
                while self._read_ascii_facet(f) is not None:
                    pass
            else:
                # Reading Binary STL file.
                chunk = f.read(4)
                facets = struct.unpack('<I', chunk)[0]
                for n in range(facets):
                    if self._read_binary_facet(f) is None:
                        break

    def _write_ascii_file(self, filename):
        with open(filename, 'wb') as f:
            f.write("solid Model\n")
            for facet in self.facets.sorted():
                f.write(
                    "  facet normal {norm:s}\n"
                    "    outer loop\n"
                    "      vertex {v0:s}\n"
                    "      vertex {v1:s}\n"
                    "      vertex {v2:s}\n"
                    "    endloop\n"
                    "  endfacet\n"
                    .format(
                        v0=facet[0],
                        v1=facet[1],
                        v2=facet[2],
                        norm=facet.norm
                    )
                )
            f.write("endsolid Model\n")

    def _write_binary_file(self, filename):
        with open(filename, 'wb') as f:
            f.write('{0:-80s}'.format('Binary STL Model'))
            f.write(struct.pack('<I', len(self.facets)))
            for facet in self.facets.sorted():
                f.write(struct.pack(
                    '<3f 3f 3f 3f H',
                    facet.norm[0], facet.norm[1], facet.norm[2],
                    facet[0][0], facet[0][1], facet[0][2],
                    facet[1][0], facet[1][1], facet[1][2],
                    facet[2][0], facet[2][1], facet[2][2],
                    0
                ))

    def write_file(self, filename, binary=False):
        if binary:
            self._write_binary_file(filename)
        else:
            self._write_ascii_file(filename)

    def _gl_set_color(self, side, color, shininess=0.33):
        glMaterialfv(side, GL_AMBIENT_AND_DIFFUSE, color)
        glMaterialfv(side, GL_SPECULAR, color)
        glMaterialf(side, GL_SHININESS, int(127*shininess))
        glColor4fv(color)

    def _gl_regenerate_backdrop_if_needed(self):
        if self._grid_list:
            return

        xcm = int(math.ceil((self.points.maxx - self.points.minx) / 10.0)+2)
        ycm = int(math.ceil((self.points.maxy - self.points.miny) / 10.0)+2)
        zcm = int(math.ceil((self.points.maxz - self.points.minz) / 10.0)+2)
        zmin = self.cx - ((self.points.maxz - self.points.minz) / 2.0)
        ox = self.cx - xcm/2.0 * 10
        oy = self.cy - ycm/2.0 * 10

        self._grid_list = glGenLists(1)
        glNewList(self._grid_list, GL_COMPILE)

        glDisable(GL_CULL_FACE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        # Draw reference axes
        glLineWidth(2.0)
        self._gl_set_color(GL_FRONT_AND_BACK, [0.0, 0.0, 0.0, 1.0], shininess=0.0)
        glBegin(GL_LINES)
        glVertex3fv([ox, oy, zmin])
        glVertex3fv([ox + 10, oy, zmin])
        glEnd()
        glBegin(GL_LINES)
        glVertex3fv([ox, oy, zmin])
        glVertex3fv([ox, oy + 10, zmin])
        glEnd()
        glBegin(GL_LINES)
        glVertex3fv([ox, oy, zmin])
        glVertex3fv([ox, oy, zmin+10])
        glEnd()

        # Draw dark squares of 1cm build plate grid
        self._gl_set_color(GL_FRONT_AND_BACK, [0.2, 0.2, 0.7, 0.3], shininess=0.75)
        for gx in range(xcm):
            for gy in range(ycm):
                if (gx + gy) % 2 == 0:
                    continue
                x1 = ox + gx * 10
                y1 = oy + gy * 10
                x2 = x1 + 10
                y2 = y1 + 10
                glBegin(GL_POLYGON)
                glVertex3fv([x1, y1, zmin-0.1])
                glVertex3fv([x2, y1, zmin-0.1])
                glVertex3fv([x2, y2, zmin-0.1])
                glVertex3fv([x1, y2, zmin-0.1])
                glEnd()

        # Draw light squares of 1cm build plate grid
        self._gl_set_color(GL_FRONT_AND_BACK, [0.5, 0.5, 0.9, 0.3], shininess=0.75)
        for gx in range(xcm):
            for gy in range(ycm):
                if (gx + gy) % 2 != 0:
                    continue
                x1 = ox + gx * 10
                y1 = oy + gy * 10
                x2 = x1 + 10
                y2 = y1 + 10
                glBegin(GL_POLYGON)
                glVertex3fv([x1, y1, zmin-0.1])
                glVertex3fv([x2, y1, zmin-0.1])
                glVertex3fv([x2, y2, zmin-0.1])
                glVertex3fv([x1, y2, zmin-0.1])
                glEnd()

        glEndList()

    def _gl_regenerate_model_if_needed(self):
        if self._model_list:
            return

        # Draw model facets.
        cp = (self.cx, self.cy, self.cz)
        self._model_list = glGenLists(1)
        glNewList(self._model_list, GL_COMPILE)
        for facet in self.facets:
            if facet.count == 1:
                glBegin(GL_POLYGON)
                glNormal3fv(tuple(facet.norm))
                for vertex in facet.vertices:
                    glVertex3fv(tuple(vertex-cp))
                glEnd()
        glEndList()

    def _gl_regenerate_errors_if_needed(self):
        if self._errs_list:
            return

        self._errs_list = glGenLists(1)
        glNewList(self._errs_list, GL_COMPILE)

        # Draw error facets.
        cp = (self.cx, self.cy, self.cz)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self._gl_set_color(GL_FRONT_AND_BACK, [1.0, 0.5, 0.5, 1.0], shininess=0.0)
        for facet in self.facets:
            if facet.count != 1:
                glBegin(GL_POLYGON)
                glNormal3fv(tuple(facet.norm))
                for vertex in facet.vertices:
                    glVertex3fv(tuple(vertex-cp))
                glEnd()
        for facet in self.facets:
            if facet.count != 1:
                glBegin(GL_POLYGON)
                glNormal3fv([-x for x in facet.norm])
                for vertex in reversed(facet.vertices):
                    glVertex3fv(tuple(vertex-cp))
                glEnd()

        # draw error facet edges.
        glLineWidth(4.0)
        self._gl_set_color(GL_FRONT_AND_BACK, [0.8, 0.0, 0.0, 1.0], shininess=0.0)
        for edge in self.edges:
            if edge.count != 2:
                # Draw bad edges in a highlighted color.
                glBegin(GL_LINES)
                for vertex in edge:
                    glVertex3fv(tuple(vertex-cp))
                glEnd()
        glEndList()

    def _gl_display(self):
        #gluLookAt(0, 0, 4.0 * self.boundsrad / self._zoom,  0, 0, 0,  0, 1, 0)

        glClearColor(0.6, 0.6, 0.6, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPushMatrix()
        glTranslate(self._xtrans, self._ytrans, 0)
        scalefac = self._zoom * (1.0 if self.perspective else 1.333)
        glScale(scalefac, scalefac, scalefac)
        glMultMatrixd(self._view_q.transformation_matrix.reshape(16))

        self._gl_regenerate_model_if_needed()
        self._gl_regenerate_errors_if_needed()
        self._gl_regenerate_backdrop_if_needed()

        if not self.wireframe:
            glLineWidth(1.0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            self._gl_set_color(GL_FRONT, [0.2, 0.5, 0.2, 1.0], shininess=0.333)
            self._gl_set_color(GL_BACK, [0.7, 0.4, 0.4, 1.0], shininess=0.1)
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
            glCallList(self._model_list)
            glCullFace(GL_FRONT)
            glCallList(self._model_list)
        if self.wireframe or self.show_facets:
            glLineWidth(1.0)
            glDisable(GL_CULL_FACE)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-1.0, 1.0);
            self._gl_set_color(GL_FRONT_AND_BACK, [0.0, 0.0, 0.0, 1.0], shininess=0.0)
            glCallList(self._model_list)
            glDisable(GL_POLYGON_OFFSET_FILL)
        glCallList(self._errs_list)
        glCallList(self._grid_list)  # draw last because transparent

        glPopMatrix()
        glFlush()
        glutSwapBuffers()
        glutPostRedisplay()

    def _gl_reshape(self, width, height):
        """window reshape callback."""
        glViewport(0, 0, width, height)

        xspan = self.points.maxx - self.points.minx
        yspan = self.points.maxy - self.points.miny
        zspan = self.points.maxz - self.points.minz

        self.boundsrad = r = 0.5 * max(xspan, yspan, zspan) * math.sqrt(2.0)
        self.cx = (self.points.minx + self.points.maxx) / 2.0
        self.cy = (self.points.miny + self.points.maxy) / 2.0
        self.cz = (self.points.minz + self.points.maxz) / 2.0

        winrad = .5 * min(width, height) / r
        w, h = width / winrad, height / winrad

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if self.perspective:
            gluPerspective(40.0, width/float(height), 1.0, min(1000.0, 10.0 * r))
        else:
            glOrtho(-w, w, -h, h, 1.0, min(1000.0, 10.0 * r))

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 4.0 * self.boundsrad,  0, 0, 0,  0, 1, 0)

        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 1.0*self.boundsrad, 4.0*self.boundsrad, 0.0])
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)

    def _gl_keypressed(self, key, x, y):
        if key == b'\033':
            sys.exit()
        elif key == b'r':
            self.reset_view()

        elif key == b'\x17':
            self._ytrans += 10.0
        elif key == b'\x13':
            self._ytrans -= 10.0
        elif key == b'\x04':
            self._xtrans += 10.0
        elif key == b'\x01':
            self._xtrans -= 10.0

        elif key == b'w':
            q = Quaternion(axis=[1, 0, 0], degrees=5)
            self._view_q = (self._view_q * q).unit
        elif key == b's':
            q = Quaternion(axis=[1, 0, 0], degrees=-5)
            self._view_q = (self._view_q * q).unit
        elif key == b'd':
            q = Quaternion(axis=[0, 1, 0], degrees=-5)
            self._view_q = (self._view_q * q).unit
        elif key == b'a':
            q = Quaternion(axis=[0, 1, 0], degrees=5)
            self._view_q = (self._view_q * q).unit

        elif key == b'q':
            q = Quaternion(axis=[0, 0, 1], degrees=-5)
            self._view_q = (self._view_q * q).unit
        elif key == b'e':
            q = Quaternion(axis=[0, 0, 1], degrees=5)
            self._view_q = (self._view_q * q).unit

        elif key == b'=':
            self._zoom *= 1.05
            self._zoom = min(10.0,max(0.1,self._zoom))
        elif key == b'-':
            self._zoom /= 1.05
            self._zoom = min(10.0,max(0.1,self._zoom))

        elif key == b'1':
            self.wireframe = True
            self.show_facets = True
        elif key == b'2':
            self.wireframe = False
            self.show_facets = True
        elif key == b'3':
            self.wireframe = False
            self.show_facets = False

        elif key == b'4':
            self._view_q = Quaternion(axis=[1, 0, 0], degrees=0)
        elif key == b'5':
            self._view_q = Quaternion(axis=[1, 0, 0], degrees=180)
        elif key == b'6':
            self._view_q = Quaternion(axis=[1, 0, 0], degrees=-90)
            self._view_q *= Quaternion(axis=[0, 0, 1], degrees=180)
        elif key == b'7':
            self._view_q = Quaternion(axis=[1, 0, 0], degrees=90)
        elif key == b'8':
            self._view_q = Quaternion(axis=[0, 0, 1], degrees=90)
            self._view_q *= Quaternion(axis=[1, 0, 0], degrees=90)
        elif key == b'9':
            self._view_q = Quaternion(axis=[0, 0, 1], degrees=-90)
            self._view_q *= Quaternion(axis=[1, 0, 0], degrees=90)
        elif key == b'0':
            self._view_q = Quaternion(axis=[0, 0, 1], degrees=25)
            self._view_q *= Quaternion(axis=[1, 0, 0], degrees=55)

        elif key == b'p':
            self.perspective = not self.perspective
            self._gl_reshape(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT))
        glutPostRedisplay()

    def _gl_mousebutton(self, button, state, x, y):
        w, h = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        cx, cy = w/2.0, h/2.0
        self._mouse_state = state
        if state == GLUT_DOWN:
            self._xstart = x
            self._ystart = y
            self._mouse_btn = button
            if button == GLUT_LEFT_BUTTON:
                if glutGetModifiers() == GLUT_ACTIVE_SHIFT:
                    self._action = "ZOOM"
                elif glutGetModifiers() == GLUT_ACTIVE_CTRL:
                    self._action = "TRANS"
                elif math.hypot(cx-x, cy-y) > min(w,h)/2:
                    self._action = "ZROT"
                else:
                    self._action = "XYROT"
            elif button == GLUT_MIDDLE_BUTTON:
                self._action = "ZOOM"
            elif button == GLUT_RIGHT_BUTTON:
                self._action = "TRANS"
            elif button == 3:
                self._zoom *= 1.01
                self._zoom = min(10.0,max(0.1,self._zoom))
            elif button == 4:
                self._zoom /= 1.01
                self._zoom = min(10.0,max(0.1,self._zoom))
        else:
            self._action = None
        glutPostRedisplay()

    def _gl_mousemotion(self, x, y):
        w, h = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        cx, cy = w/2.0, h/2.0
        dx = x - self._xstart
        dy = y - self._ystart
        r = 5.0 * self.boundsrad / min(w, h)
        if self._action == "TRANS":
            self._xtrans += dx * r
            self._ytrans -= dy * r
        elif self._action == "ZOOM":
            if dy >= 0:
                self._zoom *= (1.0 + dy/100.0)
            else:
                self._zoom /= (1.0 - dy/100.0)
            self._zoom = min(10.0,max(0.1,self._zoom))
        elif self._action == "XYROT":
            qx = Quaternion(axis=[0, 1, 0], degrees=-dx*360.0/min(w,h))
            qy = Quaternion(axis=[1, 0, 0], degrees=-dy*360.0/min(w,h))
            self._view_q = self._view_q * qx * qy
            self._view_q = self._view_q.unit
        elif self._action == "ZROT":
            oldang = math.atan2(self._ystart-cy, self._xstart-cx)
            newang = math.atan2(y-cy, x-cx)
            dang = newang - oldang
            qz = Quaternion(axis=[0, 0, 1], radians=dang)
            self._view_q = self._view_q * qz
            self._view_q = self._view_q.unit
        self._xstart = x
        self._ystart = y
        glutPostRedisplay()

    def gui_show(self, wireframe=False, show_facets=True):
        self.wireframe = wireframe
        self.show_facets = show_facets

        glutInit(sys.argv)
        glutInitWindowSize(self.width, self.height)
        glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA)
        glutCreateWindow("STL Show")
        glutDisplayFunc(self._gl_display)
        glutKeyboardFunc(self._gl_keypressed)
        glutMouseFunc(self._gl_mousebutton)
        glutMotionFunc(self._gl_mousemotion)
        glutReshapeFunc(self._gl_reshape)

        # Use depth buffering for hidden surface elimination.
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        # cheap-assed Anti-aliasing
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_POLYGON_SMOOTH)
        glEnable(GL_LINE_SMOOTH)

        # Setup the view of the cube.
        self._gl_reshape(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT))

        if plat.system() == "Darwin":
            os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

        glutMainLoop()

    def _check_manifold_duplicate_faces(self):
        return [facet for facet in self.facets if facet.count != 1]

    def _check_manifold_hole_edges(self):
        return [edge for edge in self.edges if edge.count == 1]

    def _check_manifold_excess_edges(self):
        return [edge for edge in self.edges if edge.count > 2]

    def check_manifold(self, verbose=False):
        is_manifold = True

        self.dupe_faces = self._check_manifold_duplicate_faces()
        for face in self.dupe_faces:
            is_manifold = False
            print("NON-MANIFOLD DUPLICATE FACE! {0}: {1}"
                  .format(self.filename, face))

        self.hole_edges = self._check_manifold_hole_edges()
        for edge in self.hole_edges:
            is_manifold = False
            print("NON-MANIFOLD HOLE EDGE! {0}: {1}"
                  .format(self.filename, edge))

        self.dupe_edges = self._check_manifold_excess_edges()
        for edge in self.dupe_edges:
            is_manifold = False
            print("NON-MANIFOLD DUPLICATE EDGE! {0}: {1}"
                  .format(self.filename, edge))

        return is_manifold


def main():
    parser = argparse.ArgumentParser(prog='myprogram')
    parser.add_argument('-v', '--verbose',
                        help='Show verbose output.',
                        action="store_true")
    parser.add_argument('-c', '--check-manifold',
                        help='Perform manifold validation of model.',
                        action="store_true")
    parser.add_argument('-b', '--write-binary',
                        help='Use binary STL format for output.',
                        action="store_true")
    parser.add_argument('-o', '--outfile',
                        help='Write normalized STL to file.')
    parser.add_argument('-g', '--gui-display',
                        help='Show non-manifold edges in GUI.',
                        action="store_true")
    parser.add_argument('-f', '--show-facets',
                        help='Show facet edges in GUI.',
                        action="store_true")
    parser.add_argument('-w', '--wireframe-only',
                        help='Display wireframe only in GUI.',
                        action="store_true")
    parser.add_argument('infile', help='Input STL filename.')
    args = parser.parse_args()

    stl = StlData()
    stl.read_file(args.infile)
    if args.verbose:
        print("Read {0} ({1:.1f} x {2:.1f} x {3:.1f})".format(
            args.infile,
            stl.points.maxx - stl.points.minx,
            stl.points.maxy - stl.points.miny,
            stl.points.maxz - stl.points.minz,
        ))

    manifold = True
    if args.check_manifold:
        manifold = stl.check_manifold(verbose=args.verbose)
        if manifold and (args.verbose or args.gui_display):
            print("{0} is manifold.".format(args.infile))
    if args.gui_display:
        stl.gui_show(wireframe=args.wireframe_only, show_facets=args.show_facets)
    if not manifold:
        sys.exit(-1)

    if args.outfile:
        stl.write_file(args.outfile, binary=args.write_binary)
        if args.verbose:
            print("Wrote {0} ({1})".format(
                args.outfile,
                ("binary" if args.write_binary else "ASCII"),
            ))

    sys.exit(0)


if __name__ == "__main__":
    main()


# vim: expandtab tabstop=4 shiftwidth=4 softtabstop=4 nowrap
