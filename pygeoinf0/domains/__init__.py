"""Domains in R^2 and R^3: boxes, disks, cylinders and balls; and closed manifolds: circles, spheres, flat tori.

Each domain has natural coordinates in which Lebesgue measure is a product of one-dimensional weights:

    Box       (x, y[, z])          dx dy dz
    Disk      (r, phi)             r dr dphi
    Cylinder  (r, phi, z)          r dr dphi dz
    Ball      (r, theta, phi)      r^2 sin(theta) dr dtheta dphi
    Shell     (r, theta, phi)      r^2 sin(theta) dr dtheta dphi (inner <= r <= outer; layers of a LayeredBall)
    Circle    (phi)                R dphi                       (arc length; a curve in R^2)
    Sphere    (theta, phi)         R^2 sin(theta) dtheta dphi   (surface area; a surface in R^3)
    Torus     (x, y[, z])          dx dy (dz)                   (flat; periodic in every coordinate)
    Segment   (s)                  ds                           (arc length; also Polyline, and Arc on a sphere)
    Patch     (s, t)               sin(angle) ds dt             (area of a flat parallelogram in R^3; in R^2 a
                                                                 parallelogram of full dimension)
    Curve     (s)                  ds                           (arc length of a curve given by a callable)

and two sets that are supports of measures but not domains of function spaces: Simplicial (unions of simplices, with
an optional chart onto a sphere) and Surface (a parametric surface in R^3); see EXP-031 DESIGN 4.

so one tensor-product backend serves all of them (realisations/series/tensor.py). Functions are always given and
evaluated in Cartesian coordinates (x, y[, z]); the natural coordinates are available as elements (M.r, ...).
"""
from .solids import Ball
from .base import JACOBIANS, Domain
from .solids import Box
from .curves import Curve, Surface
from .solids import Cylinder
from .solids import Disk
from .flat import Arc, Flat, Patch, Polyline, Segment, clip
from .layered import Crossing, LayeredBall, Shell
from .manifolds import Circle, Sphere, Torus
from .partition import Partition
from .simplicial import Simplex, Simplicial, inside_closure

__all__ = ["Domain", "Box", "Disk", "Cylinder", "Ball", "Shell", "LayeredBall", "Crossing", "Circle", "Sphere", "Torus", "Flat", "Segment", "Patch", "Polyline", "Arc", "clip", "JACOBIANS", "Curve", "Surface", "Simplex",
           "Simplicial", "inside_closure", "Partition"]
