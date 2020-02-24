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

from . import __version__  # noqa: F401


__all__ = ['planar_rot_3d']

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
                     dtype=np.double)
    mat_2d = np.array([[cs, sn], [-sn, cs]], dtype=np.double)
    return np.insert(np.insert(mat_2d, axis, [0.0, 0.0], 1), axis, axisv, 0)
