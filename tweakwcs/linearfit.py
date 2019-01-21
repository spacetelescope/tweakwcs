# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A module that provides algorithms for performing linear fit between
sets of 2D points.

:Authors: Mihai Cara, Warren Hack (contact: help@stsci.edu)

:License: :doc:`../LICENSE`

"""
import logging
import numpy as np

from .linalg import inv
from . import __version__, __version_date__

__author__ = 'Mihai Cara, Warren Hack'

__all__ = ['iter_linear_fit', 'build_fit_matrix']


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class SingularMatrixError(Exception):
    """ An error class used to report when a singular matrix is encountered."""
    pass


class NotEnoughPointsError(Exception):
    """
    An error class used to report when there are not enough points to
    find parameters of a linear transformation.
    """
    pass


def iter_linear_fit(xy, uv, wxy=None, wuv=None,
                    xyindx=None, uvindx=None, xyorig=None, uvorig=None,
                    fitgeom='general', nclip=3, sigma=3.0, center=None,
                    verbose=True):
    """
    Compute iteratively using sigma-clipping linear transformation parameters
    that fit `xy` sources to `uv` sources.

    """
    minobj_per_fitgeom = {'shift': 1, 'rscale': 2, 'general': 3}
    minobj = minobj_per_fitgeom[fitgeom]

    xy = np.array(xy)
    uv = np.array(uv)

    mask = np.ones(len(xy), dtype=np.bool_)

    if wxy is not None:
        wxy = np.asarray(wxy)
        mask *= wxy > 0.0

    if wuv is not None:
        wuv = np.asarray(wuv)
        mask *= wxy > 0.0

    if xy.shape[0] < nclip:
        log.warning("The number of sources for the fit < number of clipping "
                    "iterations.")
        log.warning("Resetting number of clipping iterations to 0.")
        nclip = 0

    if center is None:
        center = uv.mean(axis=0, dtype=np.longdouble)

    xy -= center
    uv -= center

    if fitgeom == 'general':
        linear_fit = fit_general
    elif fitgeom == 'rscale':
        linear_fit = fit_rscale
    elif fitgeom == 'shift':
        linear_fit = fit_shifts
    else:
        raise ValueError("Unsupported 'fitgeom' value: '{}'".format(fitgeom))

    if verbose:
        log.info("Performing '{:s}' fit".format(fitgeom))

    # initial fit:
    fit = linear_fit(xy, uv, wxy, wuv)

    if nclip is None:
        nclip = 0
    effective_nclip = 0

    # clipping iterations:
    for n in range(nclip):
        resids = fit['resids']

        # redefine what pixels will be included in next iteration
        cutoff = sigma * fit['rmse']

        goodpix = np.linalg.norm(resids, axis=1) < cutoff
        if n == 0:
            goodpix *= mask
        ngoodpix = np.count_nonzero(goodpix)

        if ngoodpix < minobj:
            break

        effective_nclip += 1

        xy = xy[goodpix]
        uv = uv[goodpix]

        if xyindx is not None:
            xyindx = xyindx[goodpix]
        if uvindx is not None:
            uvindx = uvindx[goodpix]

        if xyorig is not None:
            xyorig = xyorig[goodpix]
        if uvorig is not None:
            uvorig = uvorig[goodpix]

        if wxy is not None:
            wxy = wxy[goodpix]
        if wuv is not None:
            wuv = wuv[goodpix]

        fit = linear_fit(xy, uv, wxy, wuv)

    fit['xy_coords'] = xy
    fit['uv_coords'] = uv
    fit['xy_indx'] = xyindx
    fit['uv_indx'] = uvindx
    fit['xy_orig_xy'] = xyorig
    fit['uv_orig_xy'] = uvorig
    fit['eff_nclip'] = effective_nclip
    return fit


def fit_shifts(xy, uv, wxy=None, wuv=None):
    """ Performs a simple fit for the shift only between
        matched lists of positions 'xy' and 'uv'.

        =================================
        DEVELOPMENT NOTE:
            Checks need to be put in place to verify that
            enough objects are available for a fit.
        =================================

        Output:
           (Xo,Yo),Rot,(Scale,Sx,Sy)
           where
                Xo,Yo:  offset,
                Rot:    rotation,
                Scale:  average scale change, and
                Sx,Sy:  scale changes in X and Y separately.

        Algorithm and nomenclature provided by: Colin Cox (11 Nov 2004)

    """
    if len(xy) < 1:
        raise NotEnoughPointsError(
            "At least one point is required to find shifts."
        )

    diff_pts = xy - uv

    if wxy is None and wuv is None:
        # no weighting
        meanx = (diff_pts[:, 0].mean(dtype=np.longdouble)).astype(np.float64)
        meany = (diff_pts[:, 1].mean(dtype=np.longdouble)).astype(np.float64)

    else:
        if wxy is None:
            w = np.array(wuv, dtype=np.longdouble)
        elif wuv is None:
            w = np.array(wxy, dtype=np.longdouble)
        else:
            # 1/w = sigma**2 = sigma_xy**2 + sigma_uv**2 = 1/wxy + 1/wuv
            wuv = np.array(wuv, dtype=np.longdouble)
            wxy = np.array(wxy, dtype=np.longdouble)
            m = np.logical_and(wuv > 0, wxy > 0)
            w = np.zeros_like(wuv)
            w[m] = wxy[m] * wuv[m] / (wxy[m] + wuv[m])

        if np.any(w < 0.0):
            raise ValueError("Invalid weights: weights must be non-negative.")

        if np.sum(w > 0) < 1:
            raise ValueError("Not enough valid data for 'shift' fit: "
                             "too many weights are zero!")

        w /= np.sum(w, dtype=np.longdouble)

        meanx = np.dot(w, diff_pts[:, 0]).astype(np.float64)
        meany = np.dot(w, diff_pts[:, 1]).astype(np.float64)

    Pcoeffs = np.array([1.0, 0.0, meanx])
    Qcoeffs = np.array([0.0, 1.0, meany])

    fit = build_fit(Pcoeffs, Qcoeffs, 'shift')
    resids = diff_pts - fit['offset']
    fit['resids'] = resids
    if wxy is None and wuv is None:
        fit['rmse'] = float(np.sqrt(np.mean(2 * resids**2)))
        fit['mae'] = float(np.mean(np.linalg.norm(resids, axis=1)))
    else:
        fit['rmse'] = float(np.sqrt(np.sum(np.dot(w, resids**2))))
        fit['mae'] = float(np.dot(w, np.linalg.norm(resids, axis=1)))

    return fit


# Implementation of geomap 'rscale' fitting based on 'lib/geofit.x'
# by Warren Hack. Support for axis flips added by Mihai Cara.
def fit_rscale(xy, uv, wxy=None, wuv=None):
    """
    Set up the products used for computing the fit derived using the code from
    lib/geofit.x for the function 'geo_fmagnify()'. Comparisons with results
    from geomap (no additional clipping) were made and produced the same
    results out to 5 decimal places.

    Output
    ------
    fit: dict
        Dictionary containing full solution for fit.
    """
    if len(xy) < 2:
        raise NotEnoughPointsError(
            "At least two points are required to find shifts, rotation, and "
            "scale."
        )

    x = np.array(xy[:, 0], dtype=np.longdouble)
    y = np.array(xy[:, 1], dtype=np.longdouble)
    u = np.array(uv[:, 0], dtype=np.longdouble)
    v = np.array(uv[:, 1], dtype=np.longdouble)

    if wxy is None and wuv is None:
        # no weighting
        xm = np.mean(x)
        ym = np.mean(y)
        um = np.mean(u)
        vm = np.mean(v)

        x -= xm
        y -= ym
        u -= um
        v -= vm

        Su2 = np.dot(u, u)
        Sv2 = np.dot(v, v)
        Sxv = np.dot(x, v)
        Syu = np.dot(y, u)
        Sxu = np.dot(x, u)
        Syv = np.dot(y, v)
        Su2v2 = Su2 + Sv2

    else:
        if wxy is None:
            w = np.array(wuv, dtype=np.longdouble)
        elif wuv is None:
            w = np.array(wxy, dtype=np.longdouble)
        else:
            # 1/w = sigma**2 = sigma_xy**2 + sigma_uv**2 = 1/wxy + 1/wuv
            wuv = np.array(wuv, dtype=np.longdouble)
            wxy = np.array(wxy, dtype=np.longdouble)
            m = np.logical_and(wuv > 0, wxy > 0)
            w = np.zeros_like(wuv)
            w[m] = wxy[m] * wuv[m] / (wxy[m] + wuv[m])

        if np.any(w < 0.0):
            raise ValueError("Invalid weights: weights must be non-negative.")

        if np.sum(w > 0) < 2:
            raise ValueError("Not enough valid data for 'rscale' fit: "
                             "too many weights are zero!")

        w /= np.sum(w, dtype=np.longdouble)
        xm = np.dot(w, x)
        ym = np.dot(w, y)
        um = np.dot(w, u)
        vm = np.dot(w, v)

        x -= xm
        y -= ym
        u -= um
        v -= vm

        Su2 = np.dot(w, u**2)
        Sv2 = np.dot(w, v**2)
        Sxv = np.dot(w, x * v)
        Syu = np.dot(w, y * u)
        Sxu = np.dot(w, x * u)
        Syv = np.dot(w, y * v)
        Su2v2 = Su2 + Sv2

    det = Sxu * Syv - Sxv * Syu
    if (det < 0):
        rot_num = Sxv + Syu
        rot_denom = Sxu - Syv
    else:
        rot_num = Sxv - Syu
        rot_denom = Sxu + Syv

    if rot_num == rot_denom:
        theta = 0.0
    else:
        theta = np.rad2deg(np.arctan2(rot_num, rot_denom))
        if theta < 0:
            theta += 360.0

    ctheta = np.cos(np.deg2rad(theta))
    stheta = np.sin(np.deg2rad(theta))
    s_num = rot_denom * ctheta + rot_num * stheta

    if Su2v2 > 0.0:
        mag = s_num / Su2v2
    else:
        raise SingularMatrixError(
            "Singular matrix: suspected colinear points."
        )

    if det < 0:
        # "flip" y-axis (reflection about x-axis *after* rotation)
        # NOTE: keep in mind that 'fit_matrix'
        #       is the transposed rotation matrix.
        sthetax = -mag * stheta
        cthetay = -mag * ctheta
    else:
        sthetax = mag * stheta
        cthetay = mag * ctheta

    cthetax = mag * ctheta
    sthetay = mag * stheta

    sdet = np.sign(det)
    xshift = xm - um * cthetax - sdet * vm * sthetax
    yshift = ym + sdet * um * sthetay - vm * cthetay

    P = np.array([cthetax, sthetay, xshift], dtype=np.float64)
    Q = np.array([-sthetax, cthetay, yshift], dtype=np.float64)

    # Return the shift, rotation, and scale changes
    fit = build_fit(P, Q, fitgeom='rscale')
    resids = xy - np.dot(uv, fit['fit_matrix']) - fit['offset']
    fit['resids'] = resids
    if wxy is None and wuv is None:
        fit['rmse'] = float(np.sqrt(np.mean(2 * resids**2)))
        fit['mae'] = float(np.mean(np.linalg.norm(resids, axis=1)))
    else:
        fit['rmse'] = float(np.sqrt(np.sum(np.dot(w, resids**2))))
        fit['mae'] = float(np.dot(w, np.linalg.norm(resids, axis=1)))

    return fit


def fit_general(xy, uv, wxy=None, wuv=None):
    """ Performs a simple fit for the shift only between
        matched lists of positions 'xy' and 'uv'.

        =================================
        DEVELOPMENT NOTE:
            Checks need to be put in place to verify that
            enough objects are available for a fit.
        =================================

        Output:
           (Xo,Yo),Rot,(Scale,Sx,Sy)
           where
                Xo,Yo:  offset,
                Rot:    rotation,
                Scale:  average scale change, and
                Sx,Sy:  scale changes in X and Y separately.

        Algorithm and nomenclature provided by: Colin Cox (11 Nov 2004)

    """
    if len(xy) < 3:
        raise NotEnoughPointsError(
            "At least three points are required to find 6-parameter linear "
            "affine transformations."
        )

    x = np.array(xy[:, 0], dtype=np.longdouble)
    y = np.array(xy[:, 1], dtype=np.longdouble)
    u = np.array(uv[:, 0], dtype=np.longdouble)
    v = np.array(uv[:, 1], dtype=np.longdouble)

    if wxy is None and wuv is None:
        # no weighting

        # Set up products used for computing the fit
        Sw = float(x.size)
        Sx = x.sum()
        Sy = y.sum()
        Su = u.sum()
        Sv = v.sum()

        Sxu = np.dot(x, u)
        Syu = np.dot(y, u)
        Sxv = np.dot(x, v)
        Syv = np.dot(y, v)
        Suu = np.dot(u, u)
        Svv = np.dot(v, v)
        Suv = np.dot(u, v)

    else:
        if wxy is None:
            w = np.array(wuv, dtype=np.longdouble)
        elif wuv is None:
            w = np.array(wxy, dtype=np.longdouble)
        else:
            # 1/w = sigma**2 = sigma_xy**2 + sigma_uv**2 = 1/wxy + 1/wuv
            wuv = np.array(wuv, dtype=np.longdouble)
            wxy = np.array(wxy, dtype=np.longdouble)
            m = np.logical_and(wuv > 0, wxy > 0)
            w = np.zeros_like(wuv)
            w[m] = wxy[m] * wuv[m] / (wxy[m] + wuv[m])

        if np.any(w < 0.0):
            raise ValueError("Invalid weights: weights must be non-negative.")

        if np.sum(w > 0) < 3:
            raise ValueError("Not enough valid data for 'general' fit: "
                             "too many weights are zero!")

        # Set up products used for computing the fit
        Sw = np.sum(w, dtype=np.longdouble)
        Sx = np.dot(w, x)
        Sy = np.dot(w, y)
        Su = np.dot(w, u)
        Sv = np.dot(w, v)

        Sxu = np.dot(w, x * u)
        Syu = np.dot(w, y * u)
        Sxv = np.dot(w, x * v)
        Syv = np.dot(w, y * v)
        Suu = np.dot(w, u * u)
        Svv = np.dot(w, v * v)
        Suv = np.dot(w, u * v)

    M = np.array([[Su, Sv, Sw], [Suu, Suv, Su], [Suv, Svv, Sv]])
    U = np.array([Sx, Sxu, Sxv])
    V = np.array([Sy, Syu, Syv])

    try:
        invM = inv(M)
    except ArithmeticError:
        raise SingularMatrixError(
            "Singular matrix: suspected colinear points."
        )

    P = np.dot(invM, U).astype(np.float64)
    Q = np.dot(invM, V).astype(np.float64)
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(Q))):
        raise SingularMatrixError(
            "Singular matrix: suspected colinear points."
        )

    # Return the shift, rotation, and scale changes
    fit = build_fit(P, Q, 'general')
    resids = xy - np.dot(uv, fit['fit_matrix']) - fit['offset']
    fit['resids'] = resids
    if wxy is None and wuv is None:
        fit['rmse'] = float(np.sqrt(np.mean(2 * resids**2)))
        fit['mae'] = float(np.mean(np.linalg.norm(resids, axis=1)))
    else:
        fit['rmse'] = float(np.sqrt(np.sum(np.dot(w / Sw, resids**2))))
        fit['mae'] = float(np.dot(w / Sw, np.linalg.norm(resids, axis=1)))

    return fit


def build_fit(P, Q, fitgeom):
    # Build fit matrix:
    fit_matrix = np.dstack((P[:2], Q[:2]))[0]

    # determinant of the transformation
    det = P[0] * Q[1] - P[1] * Q[0]
    sdet = np.sign(det)
    proper = sdet >= 0

    # Create a working copy (no reflections) for computing transformation
    # parameters (scale, rotation angle, skew):
    wfit = fit_matrix.copy()

    # Default skew:
    skew = 0.0

    if fitgeom == 'shift':
        return {'offset': (P[2], Q[2]),
                'fit_matrix': fit_matrix,
                'rot': 0.0,
                'rotxy': (0.0, 0.0, 0.0, skew),
                'scale': (1.0, 1.0, 1.0),
                'coeffs': (P, Q),
                'skew': skew,
                'proper': proper,
                'fitgeom': fitgeom}

    # Compute average scale:
    s = np.sqrt(np.abs(det))
    # Compute scales for each axis:
    if fitgeom == 'general':
        sx = np.sqrt(P[0]**2 + Q[0]**2)
        sy = np.sqrt(P[1]**2 + Q[1]**2)
    else:
        sx = s
        sy = s

    # Remove scale from the transformation matrix:
    wfit[0, :] /= sx
    wfit[1, :] /= sy

    # Compute rotation angle as if we have a proper rotation.
    # This will also act as *some sort* of "average rotation" even for
    # transformations with different rot_x and rot_y:
    prop_rot = np.rad2deg(np.arctan2(wfit[1, 0] - sdet * wfit[0, 1],
                                     wfit[0, 0] + sdet * wfit[1, 1])) % 360.0

    if proper and fitgeom == 'rscale':
        rotx = prop_rot
        roty = prop_rot
        rot = prop_rot
        skew = 0.0
    else:
        rotx = np.rad2deg(np.arctan2(-wfit[0, 1], wfit[0, 0])) % 360.0
        roty = np.rad2deg(np.arctan2(wfit[1, 0], wfit[1, 1])) % 360.0
        rot = 0.5 * (rotx + roty)
        skew = roty - rotx

    return {'offset': (P[2], Q[2]),
            'fit_matrix': fit_matrix,
            'rot': prop_rot,
            'rotxy': (rotx, roty, rot, skew),
            'scale': (s, sx, sy),
            'coeffs': (P, Q),
            'skew': skew,
            'proper': proper,
            'fitgeom': fitgeom}


def build_fit_matrix(rot, scale=1):
    """
    Create an affine transformation matrix (2x2) from the provided rotation
    and scale transformations.

    Parameters
    ----------
    rot : tuple, float, optional
        Rotation angle in degrees. Two values (one for each axis) can be
        provided as a tuple.

    scale : tuple, float, optional
        Scale of the liniar transformation. Two values (one for each axis)
        can be provided as a tuple.

    Returns
    -------
    matrix : numpy.ndarray
       A 2x2 `numpy.ndarray` containing coefficients of a liniear
       transformation.

    """
    if hasattr(rot, '__iter__'):
        rx = np.deg2rad(rot[0])
        ry = np.deg2rad(rot[1])
    else:
        rx = np.deg2rad(float(rot))
        ry = rx

    if hasattr(scale, '__iter__'):
        sx = scale[0]
        sy = scale[1]
    else:
        sx = float(scale)
        sy = sx

    matrix = np.array([[sx * np.cos(rx), -sx * np.sin(rx)],
                       [sy * np.sin(ry), sy * np.cos(ry)]])

    return matrix
