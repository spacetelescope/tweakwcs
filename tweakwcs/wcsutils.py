# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A module that provides utility functions for WCS transformations.

:Authors: Mihai Cara (contact: help@stsci.edu)

"""
# NOTES:
#
# Currently this module implements some general algorithms from
# jwst.transforms.tpcorr module so that tweakwcs would not have a
# hard dependency on jwst pipeline.

import math
import numpy as np

from . import __version__, __version_date__  # noqa: F401


__all__ = ['cartesian_to_spherical', 'spherical_to_cartesian', 'planar_rot_3d']

__author__ = 'Mihai Cara'


def planar_rot_3d(angle, axis):
    """
    Create a 3D rotation matrix that performs a rotation *in a plane*
    perpendicular to the specified ``axis``.

    """
    if axis not in range(3):
        raise ValueError("'axis' must be either 0, 1, or 2.")
    axis = int(axis)
    cs = math.cos(angle)
    sn = math.sin(angle)
    axisv = np.array(axis * [0.0] + [1.0] + (2 - axis) * [0.0],
                     dtype=np.float)
    mat_2d = np.array([[cs, sn], [-sn, cs]], dtype=np.float)
    return np.insert(np.insert(mat_2d, axis, [0.0, 0.0], 1), axis, axisv, 0)


def cartesian_to_spherical(x, y, z):
    """ Convert cartesian coordinates to spherical coordinates (in deg). """
    h = np.hypot(x, y)
    alpha = np.rad2deg(np.arctan2(y, x))
    delta = np.rad2deg(np.arctan2(z, h))
    return alpha, delta


def spherical_to_cartesian(alpha, delta):
    """ Convert spherical coordinates (in deg) to cartesian. """
    alpha = np.deg2rad(alpha)
    delta = np.deg2rad(delta)
    x = np.cos(alpha) * np.cos(delta)
    y = np.cos(delta) * np.sin(alpha)
    z = np.sin(delta)
    return x, y, z
