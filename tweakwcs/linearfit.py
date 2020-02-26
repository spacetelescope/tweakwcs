# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A module that provides algorithms for performing linear fit between
sets of 2D points.

:Authors: Mihai Cara, Warren Hack

:License: :doc:`../LICENSE`

"""
import logging
import numbers
import numpy as np

from .linalg import inv
from . import __version__  # noqa: F401

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
                    fitgeom='general', center=None,
                    nclip=3, sigma=(3.0, 'rmse'), clip_accum=False):
    r"""
    Compute linear transformation parameters that "best" (in the sense of
    minimizing residuals) transform ``uv`` source position to ``xy``
    sources iteratively using sigma-clipping.

    More precisely, this functions attempts to find a ``2x2`` matrix ``F`` and
    a shift vector ``s`` that minimize the residuals between the *transformed*
    reference source coordinates ``uv``

    .. math::
        \mathbf{xy}'_k = \mathbf{F}\cdot(\mathbf{uv}_k-\mathbf{c})+\
        \mathbf{s} + \mathbf{c}
        :label: ilf1

    and the "observed" source positions ``xy``:

    .. math::
        \epsilon^2 = \Sigma_k w_k \|\mathbf{xy}_k-\mathbf{xy}'_k\|^2.
        :label: ilf2

    In the above equations, :math:`\mathbf{F}` is a ``2x2`` matrix while
    :math:`\mathbf{xy}_k` and :math:`\mathbf{uv}_k` are the position
    coordinates of the ``k``-th source (row in input ``xy`` and ``uv`` arrays).

    One of the two catalogs (``xy`` or ``uv``) contains what we refer to as
    "image" source positions and the other one as "reference" source positions.
    The meaning assigned to ``xy`` and ``uv`` parameters are up to the
    caller of this function.

    Parameters
    ----------
    xy: numpy.ndarray
        A ``(N, 2)``-shaped array of source positions (one 2-coordinate
        position per line).

    uv: numpy.ndarray
        A ``(N, 2)``-shaped array of source positions (one 2-coordinate
        position per line). This array *must have* the same length (shape)
        as the ``xy`` array.

    wxy: numpy.ndarray, None, optional
        A 1-dimensional array of weights of the same length (``N``)
        as ``xy`` array indicating how much a given coordinate should be
        weighted in the fit. If not provided or set to `None`, all positions
        will be contribute equally to the fit if ``wuv`` is also set to `None`.
        See ``Notes`` section for more details.

    wuv: numpy.ndarray, None, optional
        A 1-dimensional array of weights of the same length (``N``)
        as ``xy`` array indicating how much a given coordinate should be
        weighted in the fit. If not provided or set to `None`, all positions
        will be contribute equally to the fit if ``wxy`` is also set to `None`.
        See ``Notes`` section for more details.

    fitgeom: {'shift', 'rscale', 'general'}, optional
        The fitting geometry to be used in fitting the matched object lists.
        This parameter is used in fitting the shifts (offsets), rotations
        and/or scale changes from the matched object lists. The 'general'
        fit geometry allows for independent scale and rotation for each axis.

    center: tuple, list, numpy.ndarray, None, optional
        A list-like container with two ``X``- and ``Y``-positions of the center
        (origin) of rotations in the ``uv`` and ``xy`` coordinate frames.
        If not provided, ``center`` is estimated as a (weighted) mean position
        in the ``uv`` frame.

    nclip: int, None, optional
        Number (a non-negative integer) of clipping iterations in fit.
        Clipping will be turned off if ``nclip`` is either `None` or 0.

    sigma: float, tuple of the form (float, str), optional
        When a tuple is provided, first value (a positive number)
        indicates the number of "fit error estimates" to use for clipping.
        The second value (a string) indicates the statistic to be
        used for "fit error estimate". Currently the following values are
        supported: ``'rmse'``, ``'mae'``, and ``'std'``
        - see ``Notes`` section for more details.

        When ``sigma`` is a single number, it must be a positive number and
        the default error estimate ``'rmse'`` is assumed.

        This parameter is ignored when ``nclip`` is either `None` or 0.

    clip_accum: bool, optional
        Indicates whether or not to reset the list of "bad" (clipped out)
        sources after each clipping iteration. When set to `True` the list
        only grows with each iteration as "bad" positions never re-enter the
        pool of available position for the fit. By default the list of
        "bad" source positions is purged at each iteration.

    Returns
    -------
    fit: dict
        - ``'shift'``: A ``numpy.ndarray`` with two components of the
          computed shift.
        - ``'shift_ld'``: A ``numpy.ndarray`` with two components of the
          computed shift of type ``numpy.longdouble``.
        - ``'matrix'``: A ``2x2`` ``numpy.ndarray`` with the computed
          generalized rotation matrix.
        - ``'matrix_ld'``: A ``2x2`` ``numpy.ndarray`` with the computed
          generalized rotation matrix of type ``numpy.longdouble``.
        - ``'proper_rot'``: Rotation angle (degree) as if the rotation is
          proper.
        - ``'rot'``: A tuple of ``(rotx, roty)`` - the rotation angles with
          regard to the ``X`` and ``Y`` axes.
        - ``'<rot>'``: *Arithmetic mean* of the angles of rotation around
          ``X`` and ``Y`` axes.
        - ``'scale'``: A tuple of ``(sx, sy)`` - scale change in the direction
          of the ``X`` and ``Y`` axes.
        - ``'<scale>'``: *Geometric mean* of scales ``sx`` and ``sy``.
        - ``'skew'``: Computed skew.
        - ``'proper'``: a boolean indicating whether the rotation is proper.
        - ``'fitgeom'``: Fit geometry (allowed transformations) used for
          fitting data (to minimize residuals). This is copy of the input
          argument ``fitgeom``.
        - ``'center'``: Center of rotation
        - ``'center_ld'``: Center of rotation as a ``numpy.longdouble``.
        - ``'fitmask'``: A boolean array indicating which source positions
          where used for fitting (`True`) and which were clipped out
          (`False`). **NOTE** For weighted fits, positions with zero
          weights are automatically excluded from the fits.
        - ``'eff_nclip'``: Effective number of clipping iterations
        - ``'rmse'``: Root-Mean-Square Error
        - ``'mae'``: Mean Absolute Error
        - ``'std'``: Standard Deviation of the residuals
        - ``'resids'``: An array of residuals of the fit.
          **NOTE:** Only the residuals for the "valid" points are reported
          here. Therefore the length of this array may be smaller than the
          length of input arrays of positions.

    Notes
    -----
    **Weights**

    Weights can be provided for both "image" source positions and "reference"
    source positions. When no weights are given, all positions are weighted
    equally. When only one set of positions have weights (i.e., either ``wxy``
    or ``wuv`` is not `None`) then weights in :eq:`ilf2` are set to be equal
    to the provided set of weights. When weights for *both* "image" source
    positions and "reference" source positions are provided, then the
    combined weight that is used in :eq:`ilf2` is computed as:

    .. math::
        1/w = 1/w_{xy} + 1/w_{uv}.

    **Statistics for clipping**

    Several statistics are available for clipping iterations and all of them
    are reported in the returned ``fit`` dictionary regardless of the
    setting in ``sigma``:

    .. math::
        \mathrm{RMSE} = \sqrt{\Sigma_k w_k \|\mathbf{r}_k\|^2}

    .. math::
        \mathrm{MAE} = \sqrt{\Sigma_k w_k \|\mathbf{r}_k\|}

    .. math::
        \mathrm{STD} = \sqrt{\Sigma_k w_k \|\mathbf{r}_k - \
                       \mathbf{\overline{r}}\|^2}/(1-V_2)

    where :math:`\mathbf{r}_k=\mathbf{xy}_k-\mathbf{xy}'_k`,
    :math:`\Sigma_k w_k = 1`, and :math:`V_2=\Sigma_k w_k^2`.

    """
    if fitgeom == 'general':
        linear_fit = fit_general
    elif fitgeom == 'rscale':
        linear_fit = fit_rscale
    elif fitgeom == 'shift':
        linear_fit = fit_shifts
    else:
        raise ValueError("Unsupported 'fitgeom' value: '{}'".format(fitgeom))

    minobj_per_fitgeom = {'shift': 1, 'rscale': 2, 'general': 3}
    minobj = minobj_per_fitgeom[fitgeom]

    xy = np.array(xy, dtype=np.longdouble)
    uv = np.array(uv, dtype=np.longdouble)

    if len(xy.shape) != 2 or xy.shape[1] != 2 or uv.shape != xy.shape:
        raise ValueError("Input coordinate arrays 'xy' and 'uv' must be of "
                         "shape (N, 2) where N is the number of coordinate "
                         "points.")

    wmask = np.ones(len(xy), dtype=np.bool_)

    if wxy is not None:
        wxy = np.asarray(wxy)
        if len(wxy.shape) != 1 or wxy.shape[0] != xy.shape[0]:
            raise ValueError("Weights 'wxy' must be a 1-dimensional vector "
                             "of lengths equal to the number of input points.")
        wmask *= wxy > 0.0

    if wuv is not None:
        wuv = np.asarray(wuv)
        if len(wuv.shape) != 1 or wuv.shape[0] != xy.shape[0]:
            raise ValueError("Weights 'wuv' must be a 1-dimensional vector "
                             "of lengths equal to the number of input points.")
        wmask *= wuv > 0.0

    mask = wmask

    if sigma is None and nclip is not None and nclip > 0:
        raise ValueError("Argument 'sigma' cannot be None when 'nclip' is "
                         "a positive number.")

    if isinstance(sigma, numbers.Number):
        sigstat = 'rmse'  # default value
        nsigma = float(sigma)

    elif sigma is not None:
        nsigma = float(sigma[0])
        sigstat = sigma[1]
        if sigstat not in ['rmse', 'mae', 'std']:
            raise ValueError("Unsupported sigma statistics value.")

    if sigma is not None and nsigma <= 0.0:
        raise ValueError("The value of sigma for clipping iterations must be "
                         "positive.")

    if nclip is None:
        nclip = 0
    else:
        if nclip < 0:
            raise ValueError("Argument 'nclip' must be non-negative.")
        nclip = int(nclip)

    if np.count_nonzero(mask) == minobj:
        log.warning("The number of sources for the fit is smaller than the "
                    "minimum number of sources necessary for the requested "
                    "'fitgeom'.")
        log.warning("Resetting number of clipping iterations to 0.")
        nclip = 0

    if center is None:
        center_ld = uv[mask].mean(axis=0, dtype=np.longdouble)
        center = center_ld.astype(np.double)
    else:
        center_ld = np.longdouble(center)

    xy[mask] -= center_ld
    uv[mask] -= center_ld

    log.info("Performing '{:s}' fit".format(fitgeom))

    # initial fit:
    wmxy = None if wxy is None else wxy[mask]
    wmuv = None if wuv is None else wuv[mask]
    fit = linear_fit(xy[mask], uv[mask], wmxy, wmuv)

    # clipping iterations:
    effective_nclip = 0
    for n in range(nclip):
        resids = fit['resids']

        # redefine what pixels will be included in next iteration
        cutoff = nsigma * fit[sigstat]

        nonclipped = np.linalg.norm(resids, axis=1) < cutoff
        if np.count_nonzero(nonclipped) < minobj or nonclipped.all():
            break

        effective_nclip += 1

        prev_mask = mask
        if not clip_accum:
            mask = np.array(wmask)
        mask[prev_mask] *= nonclipped

        wmxy = None if wxy is None else wxy[mask]
        wmuv = None if wuv is None else wuv[mask]
        fit = linear_fit(xy[mask], uv[mask], wmxy, wmuv)

    fit['center'] = center
    fit['center_ld'] = center_ld
    fit['fitmask'] = mask
    fit['eff_nclip'] = effective_nclip
    return fit


def _compute_stat(fit, residuals, weights):
    if weights is None:
        fit['rmse'] = float(np.sqrt(np.mean(2 * residuals**2)))
        fit['mae'] = float(np.mean(np.linalg.norm(residuals, axis=1)))
        fit['std'] = float(np.linalg.norm(residuals.std(axis=0)))
    else:
        # assume all weights > 0 (this should be insured by the caller => no
        # need to repeat the check here)
        npts = len(weights)
        wt = np.sum(weights)
        if npts == 0 or wt == 0.0:
            fit['rmse'] = float('nan')
            fit['mae'] = float('nan')
            fit['std'] = float('nan')
            return

        w = weights / wt
        fit['rmse'] = float(np.sqrt(np.sum(np.dot(w, residuals**2))))
        fit['mae'] = float(np.dot(w, np.linalg.norm(residuals, axis=1)))

        if npts == 1:
            fit['std'] = 0.0
        else:
            # see:
            # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights_2
            wmean = np.dot(w, residuals)
            fit['std'] = float(
                np.sqrt(np.sum(np.dot(w, (residuals - wmean)**2) /
                               (1.0 - np.sum(w**2))))
            )


def fit_shifts(xy, uv, wxy=None, wuv=None):
    """ Fits (non-iteratively and without sigma-clipping) a displacement
    transformation only between input lists of positions ``xy`` and ``uv``.
    When weights are provided, a weighted fit is performed. Parameter
    descriptions and return values are identical to those in `iter_linear_fit`,
    except returned ``fit`` dictionary does not contain the following
    keys irrelevant to this function: ``'center'``, ``'fitmask'``, and
    ``'eff_nclip'``.

    """
    if xy.size == 0:
        raise NotEnoughPointsError(
            "At least one point is required to find shifts."
        )

    diff_pts = np.subtract(xy, uv, dtype=np.longdouble)

    if wxy is None and wuv is None:
        # no weighting
        w = None

        meanx = diff_pts[:, 0].mean(dtype=np.longdouble)
        meany = diff_pts[:, 1].mean(dtype=np.longdouble)

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

        if not np.sum(w > 0, dtype=np.int):
            raise ValueError("Not enough valid data for 'shift' fit: "
                             "too many weights are zero!")

        w /= np.sum(w, dtype=np.longdouble)

        meanx = np.dot(w, diff_pts[:, 0])
        meany = np.dot(w, diff_pts[:, 1])

    p = np.array([1.0, 0.0, meanx], dtype=np.longdouble)
    q = np.array([0.0, 1.0, meany], dtype=np.longdouble)

    fit = _build_fit(p, q, 'shift')
    resids = diff_pts - fit['shift']
    fit['resids'] = resids.astype(np.double)
    _compute_stat(fit, residuals=resids, weights=w)
    return fit


# Implementation of geomap 'rscale' fitting based on 'lib/geofit.x'
# by Warren Hack. Support for axis flips added by Mihai Cara.
def fit_rscale(xy, uv, wxy=None, wuv=None):
    """ Fits (non-iteratively and without sigma-clipping) a displacement,
    rotation and scale transformations between input lists of positions
    ``xy`` and ``uv``. When weights are provided, a weighted fit is performed.
    Parameter descriptions and return values are identical to those
    in `iter_linear_fit`, except returned ``fit`` dictionary does not contain
    the following keys irrelevant to this function: ``'center'``,
    ``'fitmask'``, and ``'eff_nclip'``.

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
        w = None

        xm = np.mean(x)
        ym = np.mean(y)
        um = np.mean(u)
        vm = np.mean(v)

        x -= xm
        y -= ym
        u -= um
        v -= vm

        su2 = np.dot(u, u)
        sv2 = np.dot(v, v)
        sxv = np.dot(x, v)
        syu = np.dot(y, u)
        sxu = np.dot(x, u)
        syv = np.dot(y, v)
        su2v2 = su2 + sv2

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

        su2 = np.dot(w, u**2)
        sv2 = np.dot(w, v**2)
        sxv = np.dot(w, x * v)
        syu = np.dot(w, y * u)
        sxu = np.dot(w, x * u)
        syv = np.dot(w, y * v)
        su2v2 = su2 + sv2

    det = sxu * syv - sxv * syu
    if det < 0:
        rot_num = sxv + syu
        rot_denom = sxu - syv
    else:
        rot_num = sxv - syu
        rot_denom = sxu + syv

    if rot_num == rot_denom:
        theta = 0.0
    else:
        theta = np.rad2deg(np.arctan2(rot_num, rot_denom))
        if theta < 0:
            theta += 360.0

    ctheta = np.cos(np.deg2rad(theta))
    stheta = np.sin(np.deg2rad(theta))
    s_num = rot_denom * ctheta + rot_num * stheta

    if su2v2 > 0.0:
        mag = s_num / su2v2
    else:
        raise SingularMatrixError(
            "Singular matrix: suspected colinear points."
        )

    if det < 0:
        # "flip" y-axis (reflection about x-axis *after* rotation)
        # NOTE: keep in mind that 'matrix' is the transposed rotation matrix.
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

    p = np.array([cthetax, sthetay, xshift], dtype=np.longdouble)
    q = np.array([-sthetax, cthetay, yshift], dtype=np.longdouble)

    # Return the shift, rotation, and scale changes
    fit = _build_fit(p, q, fitgeom='rscale')
    resids = xy - np.dot(uv, fit['matrix_ld'].T) - fit['shift_ld']
    fit['resids'] = resids.astype(np.double)
    _compute_stat(fit, residuals=resids, weights=w)
    return fit


def fit_general(xy, uv, wxy=None, wuv=None):
    """ Fits (non-iteratively and without sigma-clipping) a displacement,
    rotation, scale, and skew transformations (i.e., the full ``2x2``
    transformation matrix) between input lists of positions
    ``xy`` and ``uv``. When weights are provided, a weighted fit is performed.
    Parameter descriptions and return values are identical to those
    in `iter_linear_fit`, except returned ``fit`` dictionary does not contain
    the following keys irrelevant to this function: ``'center'``,
    ``'fitmask'``, and ``'eff_nclip'``.

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
        w = None

        # Set up products used for computing the fit
        sw = float(x.size)
        sx = x.sum()
        sy = y.sum()
        su = u.sum()
        sv = v.sum()

        sxu = np.dot(x, u)
        syu = np.dot(y, u)
        sxv = np.dot(x, v)
        syv = np.dot(y, v)
        suu = np.dot(u, u)
        svv = np.dot(v, v)
        suv = np.dot(u, v)

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
        sw = np.sum(w, dtype=np.longdouble)
        sx = np.dot(w, x)
        sy = np.dot(w, y)
        su = np.dot(w, u)
        sv = np.dot(w, v)

        sxu = np.dot(w, x * u)
        syu = np.dot(w, y * u)
        sxv = np.dot(w, x * v)
        syv = np.dot(w, y * v)
        suu = np.dot(w, u * u)
        svv = np.dot(w, v * v)
        suv = np.dot(w, u * v)

    m = np.array([[su, sv, sw], [suu, suv, su], [suv, svv, sv]],
                 dtype=np.longdouble)
    a = np.array([sx, sxu, sxv], dtype=np.longdouble)
    b = np.array([sy, syu, syv], dtype=np.longdouble)

    try:
        inv_m = inv(m)
    except np.linalg.LinAlgError:
        raise SingularMatrixError(
            "Singular matrix: suspected colinear points."
        )

    p = np.dot(inv_m, a)
    q = np.dot(inv_m, b)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
        raise SingularMatrixError(
            "Singular matrix: suspected colinear points."
        )  # pragma: no cover

    # Return the shift, rotation, and scale changes
    fit = _build_fit(p, q, 'general')
    resids = xy - np.dot(uv, fit['matrix_ld'].T) - fit['shift_ld']
    fit['resids'] = resids.astype(np.double)
    _compute_stat(fit, residuals=resids, weights=w)
    return fit


def _build_fit(p, q, fitgeom):
    # Build fit matrix:
    fit_matrix = np.vstack((p[:2], q[:2]))

    # determinant of the transformation
    det = p[0] * q[1] - p[1] * q[0]
    sdet = np.sign(det)
    proper = sdet >= 0

    # Create a working copy (no reflections) for computing transformation
    # parameters (scale, rotation angle, skew):
    wfit = fit_matrix.copy()

    # Skew is zero for all fitgeom except 'general':
    skew = 0.0

    if fitgeom == 'shift':
        fit = {
            'shift': np.array([p[2], q[2]], dtype=np.double),
            'shift_ld': np.array([p[2], q[2]], dtype=np.longdouble),
            'matrix': np.array(fit_matrix, dtype=np.double),
            'matrix_ld': np.array(fit_matrix, dtype=np.longdouble),
            'proper_rot': 0.0,
            'rot': (0.0, 0.0),
            '<rot>': 0.0,
            'scale': (1.0, 1.0),
            '<scale>': 1.0,
            'skew': 0.0,
            'proper': proper,
            'fitgeom': 'shift'
        }

        return fit

    # Compute average scale:
    s = np.sqrt(np.abs(det))
    # Compute scales for each axis:
    if fitgeom == 'general':
        sx, sy = np.sqrt(p[:2]**2 + q[:2]**2)
    else:
        sx = s
        sy = s

    # Remove scale from the transformation matrix:
    wfit[:, 0] /= sx
    wfit[:, 1] /= sy

    # Compute rotation angle as if we have a proper rotation.
    # This will also act as *some sort* of "average rotation" even for
    # transformations with different rot_x and rot_y:
    prop_rot = np.rad2deg(
        np.arctan2(wfit[0, 1] - sdet * wfit[1, 0],
                   wfit[0, 0] + sdet * wfit[1, 1])
    )

    if proper and fitgeom == 'rscale':
        rotx = prop_rot
        roty = prop_rot
        rot = prop_rot

    else:
        rotx = np.rad2deg(np.arctan2(-wfit[1, 0], wfit[0, 0]))
        roty = np.rad2deg(np.arctan2(wfit[0, 1], wfit[1, 1]))
        rot = 0.5 * (rotx + roty)
        skew = np.mod(roty - rotx - 180.0, 360.0) - 180.0

    fit = {
        'shift': np.array([p[2], q[2]], dtype=np.double),
        'shift_ld': np.array([p[2], q[2]], dtype=np.longdouble),
        'matrix': np.array(fit_matrix, dtype=np.double),
        'matrix_ld': np.array(fit_matrix, dtype=np.longdouble),
        'proper_rot': float(prop_rot),
        'rot': (float(rotx), float(roty)),
        '<rot>': float(rot),
        'scale': (float(sx), float(sy)),
        '<scale>': float(s),
        'skew': float(skew),
        'proper': proper,
        'fitgeom': fitgeom
    }

    return fit


def build_fit_matrix(rot, scale=1):
    r"""
    Create an affine transformation matrix (2x2) from the provided rotation
    angle(s) and scale(s):

    .. math::

        M = \begin{bmatrix}
                s_x \cos(\theta_x) & s_y \sin(\theta_y) \\
                -s_x \sin(\theta_x) & s_y \cos(\theta_y)
            \end{bmatrix}

    Parameters
    ----------
    rot: tuple, float, optional
        Rotation angle in degrees. Two values (one for each axis) can be
        provided as a tuple.

    scale: tuple, float, optional
        Scale of the liniar transformation. Two values (one for each axis)
        can be provided as a tuple.

    Returns
    -------
    matrix: numpy.ndarray
       A 2x2 `numpy.ndarray` containing coefficients of a liniear
       transformation.

    """
    if hasattr(rot, '__iter__'):
        rx, ry = map(np.deg2rad, rot)
    else:
        rx = ry = np.deg2rad(float(rot))

    if hasattr(scale, '__iter__'):
        sx, sy = scale
    else:
        sx = sy = float(scale)

    matrix = np.array([[sx * np.cos(rx), sy * np.sin(ry)],
                       [-sx * np.sin(rx), sy * np.cos(ry)]])

    return matrix
