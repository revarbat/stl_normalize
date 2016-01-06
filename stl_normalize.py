#!/usr/bin/env python

import os
import os.path
import sys
import math
import time
import struct
import numbers
import argparse
import platform
import itertools
import subprocess


# Template OpenSCAD code for displaying non-manifold issues in GUI mode.
guiscad_template = """\
module showlines(clr, lines) {{
    for (line = lines) {{
        delta = line[1]-line[0];
        dist = norm(delta);
        theta = atan2(delta[1],delta[0]);
        phi = atan2(delta[2],norm([delta[0],delta[1]]));
        translate(line[0]) {{
            rotate([0, 90-phi, theta]) {{
                color(clr) cylinder(d=0.5, h=dist);
            }}
        }}
    }}
}}
module showfaces(clr, faces) {{
    color(clr) {{
        for (face = faces) {{
            polyhedron(points=face, faces=[[0, 1, 2], [0, 2, 1]], convexity=2);
        }}
    }}
}}
showlines([1.0, 0.0, 1.0], [
{dupe_edges}
]);
showlines([1.0, 0.0, 0.0], [
{hole_edges}
]);
showfaces([1.0, 0.0, 1.0], [
{dupe_faces}
]);
color([0.0, 1.0, 0.0, 0.2]) import("{filename}", convexity=100);

"""


def float_fmt(val):
    """
    Returns a short, clean floating point string representation.
    Uneccessary trailing zeroes and decimal points are trimmed off.
    """
    s = "%.4f" % val
    while len(s) > 1 and s[-1:] in '0.':
        if s[-1:] == '.':
            s = s[:-1]
            break
        s = s[:-1]
    if (s == '-0'):
        s = '0'
    return s


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
        longzip = itertools.izip_longest(self._values, other, fillvalue=0.0)
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
            return struct.pack('<%df' % len(self._values), *self._values)
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
        longzip = itertools.izip_longest(self._values, p, fillvalue=0.0)
        for v1, v2 in reversed(list(longzip)):
            val = cmp(v1, v2)
            if val != 0:
                return val
        return 0

    def __eq__(self, other):
        """Equality comparison for points."""
        return self._values == other._values

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
        return math.sqrt(sum(pow(x1-x2, 2.0) for x1, x2 in zip(v, self)))

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
            for c1, c2 in itertools.izip_longest(cl1[i], cl2[i]):
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

    def _read_ascii_line(self, f, watchwords=None):
        line = f.readline(1024)
        if line == "":
            raise StlEndOfFileException()
        words = line.strip(' \t\n\r').lower().split()
        if words[0] == 'endsolid':
            raise StlEndOfFileException()
        argstart = 0
        if watchwords:
            watchwords = watchwords.lower().split()
            argstart = len(watchwords)
            for i in xrange(argstart):
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
            if line[0:6].lower() == "solid ":
                # Reading ASCII STL file.
                while self._read_ascii_facet(f) is not None:
                    pass
            else:
                # Reading Binary STL file.
                chunk = f.read(4)
                facets = struct.unpack('<I', chunk)[0]
                for n in xrange(facets):
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
            f.write('%-80s' % 'Binary STL Model')
            f.write(struct.pack('<I', len(self.facets)))
            for facet in self.facets.sorted():
                f.write(
                    '{norm:b}{v0:b}{v1:b}{v2:b}'.format(
                        v0=facet[0],
                        v1=facet[1],
                        v2=facet[2],
                        norm=facet.norm
                    )
                )
                f.write(struct.pack('<H', 0))

    def write_file(self, filename, binary=False):
        if binary:
            self._write_binary_file(filename)
        else:
            self._write_ascii_file(filename)

    def _openscad_face_list(self, faces):
        return ",\n".join("  {0:a}".format(x) for x in faces)

    def _openscad_edge_list(self, edges):
        return ",\n".join("  {0:a}".format(x) for x in edges)

    def _gui_display_manifold(self, hole_edges, dupe_edges, dupe_faces):
        global guiscad_template
        modulename = os.path.basename(self.filename)
        if modulename.endswith('.stl'):
            modulename = modulename[:-4]
        tmpfile = "mani-{0}.scad".format(modulename)
        with open(tmpfile, 'w') as f:
            f.write(
                guiscad_template.format(
                    hole_edges=hole_edges,
                    dupe_edges=dupe_edges,
                    dupe_faces=dupe_faces,
                    modulename=modulename,
                    filename=self.filename,
                )
            )
        if platform.system() == 'Darwin':
            subprocess.call(['open', tmpfile])
            time.sleep(5)
        else:
            subprocess.call(['openscad', tmpfile])
            time.sleep(5)
        os.remove(tmpfile)

    def _check_manifold_duplicate_faces(self):
        return [facet for facet in self.facets if facet.count != 1]

    def _check_manifold_hole_edges(self):
        return [edge for edge in self.edges if edge.count == 1]

    def _check_manifold_excess_edges(self):
        return [edge for edge in self.edges if edge.count > 2]

    def check_manifold(self, verbose=False, gui=False):
        is_manifold = True

        faces = self._check_manifold_duplicate_faces()
        dupe_faces = self._openscad_face_list(faces)
        for face in faces:
            is_manifold = False
            print("NON-MANIFOLD DUPLICATE FACE! {0}: {1}"
                  .format(self.filename, face))

        edges = self._check_manifold_hole_edges()
        hole_edges = self._openscad_edge_list(edges)
        for edge in edges:
            is_manifold = False
            print("NON-MANIFOLD HOLE EDGE! {0}: {1}"
                  .format(self.filename, edge))

        edges = self._check_manifold_excess_edges()
        dupe_edges = self._openscad_edge_list(edges)
        for edge in edges:
            is_manifold = False
            print("NON-MANIFOLD DUPLICATE EDGE! {0}: {1}"
                  .format(self.filename, edge))

        if is_manifold:
            if gui or verbose:
                print("%s is manifold." % self.filename)
        elif gui:
                self._gui_display_manifold(hole_edges, dupe_edges, dupe_faces)
        return is_manifold


def main():
    parser = argparse.ArgumentParser(prog='myprogram')
    parser.add_argument('-v', '--verbose',
                        help='Show verbose output.',
                        action="store_true")
    parser.add_argument('-c', '--check-manifold',
                        help='Perform manifold validation of model.',
                        action="store_true")
    parser.add_argument('-g', '--gui-display',
                        help='Show non-manifold edges in GUI.',
                        action="store_true")
    parser.add_argument('-b', '--write-binary',
                        help='Use binary STL format for output.',
                        action="store_true")
    parser.add_argument('-o', '--outfile',
                        help='Write normalized STL to file.')
    parser.add_argument('infile', help='Input STL filename.')
    args = parser.parse_args()

    stl = StlData()
    stl.read_file(args.infile)
    if args.verbose:
        print("Read {0} ({1:.1f} x {2:.1f} x {3:.1f})".format(
            args.infile,
            (stl.points.maxx-stl.points.minx),
            (stl.points.maxy-stl.points.miny),
            (stl.points.maxz-stl.points.minz),
        ))

    if args.check_manifold or args.gui_display:
        if not stl.check_manifold(verbose=args.verbose, gui=args.gui_display):
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
