# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides general purpose and/or specialized linear algebra
routines.

:Authors: Mihai Cara (contact: help@stsci.edu)

:License: :doc:`../LICENSE`

"""
# STDLIB
import logging

# THIRD-PARTY
import numpy as np

# LOCAL
from . import __version__  # noqa: F401

__author__ = 'Mihai Cara'

__all__ = ['inv']

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def _is_longdouble_lte_flt_type(flt_type=np.double):
    if np.longdouble is flt_type:
        return True

    fi1 = np.finfo(np.longdouble)
    fi2 = np.finfo(flt_type)

    lte_flt = all(
        [
            fi1.eps >= fi2.eps,
            fi1.bits <= fi2.bits,
            fi1.epsneg >= fi2.epsneg,
            fi1.iexp <= fi2.iexp,
            fi1.max <= fi2.max,
            fi1.maxexp <= fi2.maxexp,
            fi1.min >= fi2.min,
            fi1.minexp >= fi2.minexp,
            fi1.negep >= fi2.negep,
            fi1.nexp <= fi1.nexp,
            fi1.nmant <= fi2.nmant,
            fi1.precision <= fi2.precision,
            fi1.resolution >= fi2.resolution,
            fi1.tiny >= fi2.tiny,
        ]
    )

    return lte_flt


def _find_max_linalg_type():
    max_type = None
    for np_type in np.sctypes['float'][::-1]:  # pragma: no branch
        try:
            r = np.linalg.inv(np.identity(2, dtype=np_type))
            max_type = np_type
            eps = 100 * np.finfo(max_type).eps
            if np.max(np.abs(r - np.identity(2, dtype=np_type))) > eps:  # pragma: no branch
                log.warning('Loss of accuracy during matrix inversion.')
            break
        except TypeError:
            continue


_USE_NUMPY_LINALG_INV = _is_longdouble_lte_flt_type(flt_type=np.double)
_MAX_LINALG_TYPE = _find_max_linalg_type()
if _MAX_LINALG_TYPE is None:
    _USE_NUMPY_LINALG_INV = False  # pragma: no cover


def inv(m):
    """ This function computes inverse matrix using Gauss-Jordan elimination
    with full pivoting. Computations are performed using ``numpy.longdouble``
    precision. On systems on which ``numpy.longdouble`` is equivalent to
    ``numpy.double`` this function reverts to `numpy.linalg.inv` for
    performance reasons.

    Parameters
    ----------
    m: numpy.ndarray
        A 2D *square* matrix of type `numpy.ndarray`.

    Returns
    -------
    invm: numpy.ndarray
        Inverse matrix of the input matrix ``m``: a 2D *square* `numpy.ndarray`
        of type ``numpy.longdouble`` on systems on which it is more accurate
        than ``numpy.double``.

    """
    # check that matrix is square:
    if _USE_NUMPY_LINALG_INV:
        invm = np.linalg.inv(np.array(m, dtype=_MAX_LINALG_TYPE))
        # detect singularity:
        if not np.all(np.isfinite(invm)):
            raise np.linalg.LinAlgError('Singular matrix.')
        return invm

    m = np.array(m, dtype=np.longdouble)
    if len(m.shape) != 2 or m.shape[0] != m.shape[1]:
        raise np.linalg.LinAlgError("Input matrix must be a square matrix.")
    order = m.shape[0]

    # create permutation matrices:
    qt = np.eye(order, dtype=np.int)

    eps = np.finfo(np.double).tiny

    # initial inverse matrix:
    invm = np.eye(order, dtype=np.longdouble)

    # forward Gauss elimination with full pivoting:
    for k in range(order):
        # find pivot:
        im, jm = np.unravel_index(np.argmax(np.abs(m[k:, k:])),
                                  (order - k, order - k))
        im += k
        jm += k

        pv = m[im, jm]

        # detect singularity:
        if np.abs(pv) < eps:
            raise np.linalg.LinAlgError('Singular matrix.')

        if im != k or jm != k:
            # swap rows & columns:
            q = np.eye(order, dtype=np.int)

            # swap rows in the m and invm matrix:
            m[k, :], m[im, :] = m[im, :], m[k, :].copy()
            invm[k, :], invm[im, :] = invm[im, :], invm[k, :].copy()

            # swap rows in the Q matrix:
            q[k, :], q[jm, :] = q[jm, :], q[k, :].copy()

            # swap columns:
            m = np.dot(m, q)
            invm = np.dot(invm, q)
            qt = np.dot(qt, q)

        m[k, k:] /= pv
        r = m[k, (k + 1):]

        invm[k, :] /= pv
        w = invm[k, :]

        for l in range(k + 1, order):  # noqa: E741
            pv2 = m[l, k]
            m[l, (k + 1):] -= pv2 * r
            m[l, k] = 0.0
            invm[l, :] -= pv2 * w

    # inverse Jordan elimination:
    for k1 in range(order - 1, 0, -1):
        for k2 in range(k1 - 1, -1, -1):
            invm[k2, :] -= m[k2, k1] * invm[k1, :]

    # detect singularity:
    if not np.all(np.isfinite(invm)):
        raise np.linalg.LinAlgError('Singular matrix.')

    # undo permutations:
    invm = np.dot(qt, np.dot(invm, qt.T))

    return invm
