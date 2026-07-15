"""
A module that provides utility functions for WCS transformations.

:Authors: Mihai Cara

:License: :doc:`LICENSE`

"""

# NOTES:
#
# Currently this module implements some general algorithms from
# jwst.transforms.tpcorr module so that tweakwcs would not have a
# hard dependency on jwst pipeline.
from __future__ import annotations

import math
from functools import wraps
from typing import NamedTuple

import numpy as np

from astropy.utils.masked import get_data_and_mask

from . import __version__  # noqa: F401

__all__ = ["masked", "planar_rot_3d", "unmask_args"]

__author__ = "Mihai Cara"


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
    axisv = np.array(axis * [0.0] + [1.0] + (2 - axis) * [0.0], dtype=np.double)
    mat_2d = np.array([[cs, sn], [-sn, cs]], dtype=np.double)
    return np.insert(np.insert(mat_2d, axis, [0.0, 0.0], 1), axis, axisv, 0)


class Unmasked(NamedTuple):
    """The filtered arrays after applying a mask"""

    args: tuple[np.ndarray[tuple[int, ...], np.float64], ...]
    mask: np.ndarray[tuple[int, ...], np.bool_] | None


def unmask_args(*args: np.ndarray[tuple[int, ...], np.float64]) -> Unmasked:
    """
    Unmask the input arrays and return the unmasked data along with the mask.

    Parameters
    ----------
    *args : np.ndarray[tuple[int, ...], np.float64]
        The input arrays to be filtered.

    """
    data = []
    mask = None
    for arg in args:
        arg_data, arg_mask = get_data_and_mask(arg)
        if arg_mask is not None:
            if mask is None:
                mask = arg_mask
            else:
                mask |= arg_mask
        data.append(arg_data)

    # If anything is masked we perform the filtering
    if np.any(mask):
        return Unmasked(args=tuple(arg[~mask] for arg in data), mask=mask)

    # Otherwise we just return the arrays
    return Unmasked(args=tuple(data), mask=None)


def masked(func, mask=True):
    """
    Decorator for a function that uses an astropy model to properly handle masked arrays
        This decorator operates by extracting the unmasked elements from the input arrays,
        pass them to the decorated function, and then reconstructs the masked arrays to
        their original masked form.

    Parameters
    ----------
    func : callable
        The function to be decorated.
    mask : bool, optional
        Whether or not to reapply the mask after running the decorated function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        unmasked = unmask_args(*args)
        outputs = func(self, *unmasked.args, **kwargs)

        # Bail out early if there is no mask or we are not re applying the mask
        if unmasked.mask is None or not mask:
            return outputs

        # Make sure we have a tuple of outputs
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Iterate over the outputs filling out an array of the original shape but
        #    masked with the original mask
        masked_outputs = []
        for output in outputs:
            masked_output = np.ma.zeros_like(unmasked.mask, dtype=output.dtype)
            masked_output[~unmasked.mask] = output
            masked_output.mask = unmasked.mask
            masked_outputs.append(masked_output)

        # Return a single output if there was just one
        if len(masked_outputs) == 1:
            return masked_outputs[0]

        # Return a tuple of masked outputs if there were multiple outputs
        return tuple(masked_outputs)

    return wrapper
