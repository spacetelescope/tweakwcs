# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A module that provides algorithms matching catalogs and for initial estimation
of shifts based on 2D histograms.

:License: :doc:`../LICENSE`

"""
import logging
from abc import ABC, abstractmethod

import numpy as np
import astropy

from stsci.stimage import xyxymatch

from . import __version__  # noqa: F401

__author__ = 'Mihai Cara'

__all__ = ['MatchCatalogs', 'TPMatch']

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class MatchCatalogs(ABC):
    """ A class that provides common interface for matching catalogs. """

    def __init__(self):
        """
        """

    @abstractmethod
    def __call__(self, refcat, imcat):
        """ Performs catalog matching.

        Parameters
        ----------

        refcat: astropy.table.Table
            A reference source catalog. Reference catalog must contain
            ``'TPx'`` and ``'TPy'`` columns that provide undistorted
            (distortion-correction applied) source coordinates coordinate
            system common (shared) with the image catalog ``imcat``.

        imcat: astropy.table.Table
            Source catalog associated with an image. Image catalog must contain
            ``'TPx'`` and ``'TPy'`` columns that provide undistorted
            (distortion-correction applied) source coordinates coordinate
            system common (shared) with the image catalog ``refcat``.

        Returns
        -------
        (refcat_idx, imcat_idx): tuple of numpy.ndarray
            A tuple of two 1D `numpy.ndarray` containing indices of matched
            sources in the ``refcat`` and ``imcat`` catalogs accordingly.

        """
        pass


class TPMatch(MatchCatalogs):
    """ Catalog source matching in tangent plane. Uses ``xyxymatch``
    algorithm to cross-match sources between this catalog and
    a reference catalog.

    .. note::
        The tangent plane is a plane tangent to the celestial sphere and it
        must be not distorted, that is, if *image* coordinates are distorted,
        then distortion correction must be applied to them before tangent
        plane coordinates are computed. Alternatively, one can think that
        undistorted world coordinates are projected from the sphere onto the
        tangent plane.

    """
    def __init__(self, searchrad=3.0, separation=0.5, use2dhist=True,
                 xoffset=0.0, yoffset=0.0, tolerance=1.0):
        """
        Parameters
        ----------

        searchrad: float, optional
            The search radius for a match (in units of the tangent plane).

        separation: float, optional
            The  minimum  separation in the tangent plane (in units of
            the tangent plane) for sources in the image and reference
            catalogs in order to be considered to be disctinct sources.
            Objects closer together than ``separation`` distance
            are removed from the image and reference coordinate catalogs prior
            to matching. This parameter gets passed directly to
            :py:func:`~stsci.stimage.xyxymatch` for use in matching the object
            lists from each image with the reference catalog's object list.

        use2dhist: bool, optional
            Use 2D histogram to find initial offset?

        xoffset: float, optional
            Initial estimate for the offset in X (in units of the tangent
            plane) between the sources in the image and the reference catalogs.
            This offset will be used for all input images provided.
            This parameter is ignored when ``use2dhist`` is `True`.

        yoffset: float, optional
            Initial estimate for the offset in Y (in units of the tangent
            plane) between the sources in the image and the reference catalogs.
            This offset will be used for all input images provided.
            This parameter is ignored when ``use2dhist`` is `True`.

        tolerance: float, optional
            The matching tolerance (in units of the tangent plane) after
            applying an initial solution derived from the 'triangles'
            algorithm.  This parameter gets passed directly to
            :py:func:`~stsci.stimage.xyxymatch` for use in matching
            the object lists from each image with the reference image's object
            list.

        """
        self._use2dhist = use2dhist

        if searchrad > 0:
            self._searchrad = searchrad
        else:
            raise ValueError("'searchrad' must be a positive number.")

        if separation > 0:
            self._separation = separation
        else:
            raise ValueError("'separation' must be a positive number.")

        if tolerance > 0:
            self._tolerance = tolerance
        else:
            raise ValueError("'tolerance' must be a positive number.")

        self._xoffset = float(xoffset)
        self._yoffset = float(yoffset)

    def __call__(self, refcat, imcat, tp_wcs=None):
        r""" Performs catalog matching.

        Parameters
        ----------

        refcat: astropy.table.Table
            A reference source catalog. When a tangent-plane ``WCS`` is
            provided through ``tp_wcs``, the catalog must contain ``'RA'`` and
            ``'DEC'`` columns which indicate reference source world
            coordinates (in degrees). Alternatively, when ``tp_wcs`` is `None`,
            reference catalog must contain ``'TPx'`` and ``'TPy'`` columns that
            provide undistorted (distortion-correction applied) source
            coordinates in some *tangent plane*. In this case, the ``'RA'``
            and ``'DEC'`` columns in the ``refcat`` catalog will be ignored.

        imcat: astropy.table.Table
            Source catalog associated with an image. Must contain ``'x'`` and
            ``'y'`` columns which indicate source coordinates (in pixels) in
            the associated image. Alternatively, when ``tp_wcs`` is `None`,
            catalog must contain ``'TPx'`` and ``'TPy'`` columns that
            provide undistorted (distortion-correction applied) source
            coordinates in **the same**\ *tangent plane* used to define
            ``refcat``'s tangent plane coordinates. In this case, the ``'x'``
            and ``'y'`` columns in the ``imcat`` catalog will be ignored.

        tp_wcs: TPWCS, None, optional
            A ``WCS`` that defines a tangent plane onto which both
            reference and image catalog sources can be projected. For this
            reason, ``tp_wcs`` is associated with the image in which sources
            from the ``imcat`` catalog were found in the sense that ``tp_wcs``
            must be able to map image coordinates ``'x'`` and ``'y'`` from the
            ``imcat`` catalog to the tangent plane. When ``tp_wcs`` is
            provided, the ``'TPx'`` and ``'TPy'`` columns in both ``imcat`` and
            ``refcat`` catalogs will be ignored (if present).

        Returns
        -------
        (refcat_idx, imcat_idx): tuple of numpy.ndarray
            A tuple of two 1D `numpy.ndarray` containing indices of matched
            sources in the ``refcat`` and ``imcat`` catalogs accordingly.

        """
        # Check catalogs:
        if not isinstance(refcat, astropy.table.Table):
            raise TypeError("'refcat' must be an instance of "
                            "astropy.table.Table")

        if not refcat:
            raise ValueError("Reference catalog must contain at least one "
                             "source.")

        if not isinstance(imcat, astropy.table.Table):
            raise TypeError("'imcat' must be an instance of "
                            "astropy.table.Table")

        if not imcat:
            raise ValueError("Image catalog must contain at least one "
                             "source.")

        if tp_wcs is None:
            if 'TPx' not in refcat.colnames or 'TPy' not in refcat.colnames:
                raise KeyError("When tangent plane WCS is not provided, "
                               "'refcat' must contain both 'TPx' and 'TPy' "
                               "columns.")

            if 'TPx' not in imcat.colnames or 'TPy' not in imcat.colnames:
                raise KeyError("When tangent plane WCS is not provided, "
                               "'imcat' must contain both 'TPx' and 'TPy' "
                               "columns.")

            imxy = np.asarray([imcat['TPx'], imcat['TPy']]).T
            refxy = np.asarray([refcat['TPx'], refcat['TPy']]).T

        else:
            if 'RA' not in refcat.colnames or 'DEC' not in refcat.colnames:
                raise KeyError("When tangent plane WCS is provided,  'refcat' "
                               "must contain both 'RA' and 'DEC' columns.")

            if 'x' not in imcat.colnames or 'y' not in imcat.colnames:
                raise KeyError("When tangent plane WCS is provided,  'imcat' "
                               "must contain both 'x' and 'y' columns.")

            # compute x & y in the tangent plane provided by tp_wcs:
            imxy = np.asarray(
                tp_wcs.det_to_tanp(imcat['x'], imcat['y'])
            ).T

            refxy = np.asarray(
                tp_wcs.world_to_tanp(refcat['RA'], refcat['DEC'])
            ).T

        imcat_name = imcat.meta.get('name', 'Unnamed')
        if imcat_name is None:
            imcat_name = 'Unnamed'

        refcat_name = refcat.meta.get('name', 'Unnamed')
        if refcat_name is None:
            refcat_name = 'Unnamed'

        log.info("Matching sources from '{:s}' catalog with sources from the "
                 "reference '{:s}' catalog."
                 .format(imcat_name, refcat_name))

        ps = 1.0 if tp_wcs is None else tp_wcs.tanp_center_pixel_scale

        if self._use2dhist:
            # Determine xyoff (X,Y offset) and tolerance
            # to be used with xyxymatch:
            zpxoff, zpyoff = _estimate_2dhist_shift(
                imxy / ps,
                refxy / ps,
                searchrad=self._searchrad
            )
            xyoff = (zpxoff * ps, zpyoff * ps)

        else:
            xyoff = (self._xoffset * ps, self._yoffset * ps)

        matches = xyxymatch(
            imxy,
            refxy,
            origin=xyoff,
            tolerance=ps * self._tolerance,
            separation=ps * self._separation
        )

        return matches['ref_idx'], matches['input_idx']


def _xy_2dhist(imgxy, refxy, r):
    # This code replaces the C version (arrxyzero) from carrutils.c
    # It is about 5-8 times slower than the C version.
    dx = np.subtract.outer(imgxy[:, 0], refxy[:, 0]).ravel()
    dy = np.subtract.outer(imgxy[:, 1], refxy[:, 1]).ravel()
    idx = np.where((dx < r + 0.5) & (dx >= -r - 0.5) &
                   (dy < r + 0.5) & (dy >= -r - 0.5))
    r = int(np.ceil(r))
    h = np.histogram2d(dx[idx], dy[idx], 2 * r + 1,
                       [[-r - 0.5, r + 0.5], [-r - 0.5, r + 0.5]])
    return h[0].T


def _estimate_2dhist_shift(imgxy, refxy, searchrad=3.0):
    """ Create a 2D matrix-histogram which contains the delta between each
        XY position and each UV position. Then estimate initial offset
        between catalogs.
    """
    log.info("Computing initial guess for X and Y shifts...")

    # create ZP matrix
    zpmat = _xy_2dhist(imgxy, refxy, r=searchrad)

    nonzeros = np.count_nonzero(zpmat)
    if nonzeros == 0:
        # no matches within search radius. Return (0, 0):
        log.warning("No matches found within a search radius of {:g} pixels."
                    .format(searchrad))
        return 0.0, 0.0

    elif nonzeros == 1:
        # only one non-zero bin:
        yp, xp = np.unravel_index(np.argmax(zpmat), zpmat.shape)
        maxval = zpmat[yp, xp]
        xp -= searchrad
        yp -= searchrad
        log.info("Found initial X and Y shifts of {:.4g}, {:.4g} "
                 "based on a single non-zero bin and {} matches"
                 .format(xp, yp, int(maxval)))
        return xp, yp

    (xp, yp), fit_status, fit_sl = _find_peak(zpmat, peak_fit_box=5,
                                              mask=zpmat > 0)
    if fit_status.startswith('ERROR'):
        log.warning("No valid shift found within a search radius of {:g} "
                    "pixels.".format(searchrad))
        return 0.0, 0.0

    xp -= searchrad
    yp -= searchrad

    if fit_status == 'WARNING:EDGE':
        log.info("Found peak in the 2D histogram lies at the edge of the "
                 "histogram. Try increasing 'searchrad' for improved results.")

    flux = int(zpmat[fit_sl].sum())

    # Attempt to estimate "significance of detection":
    maxval = zpmat.max()
    zpmat_mask = (zpmat > 0) & (zpmat < maxval)

    bkg = zpmat[zpmat_mask].mean() if np.any(zpmat_mask) else -1.0

    if bkg > 0:  # pragma: no branch
        bkg = zpmat[zpmat_mask].mean()
        sig = maxval / np.sqrt(bkg)
        log.info("Found initial X and Y shifts of {:.4g}, {:.4g} "
                 "with significance of {:.4g} and {:d} matches."
                 .format(xp, yp, sig, flux))

    else:
        log.warning("Unable to estimate significance of the detection of the "
                    "initial shift.")
        log.info("Found initial X and Y shifts of {:.4g}, {:.4g} "
                 "with {:d} matches."
                 .format(xp, yp, flux))

    return xp, yp


def _find_peak(data, peak_fit_box=5, mask=None):
    """
    Find location of the peak in an array. This is done by fitting a second
    degree 2D polynomial to the data within a `peak_fit_box` and computing the
    location of its maximum. An initial
    estimate of the position of the maximum will be performed by searching
    for the location of the pixel/array element with the maximum value.

    Parameters
    ----------
    data: numpy.ndarray
        2D data.

    peak_fit_box: int, optional
        Size (in pixels) of the box around the initial estimate of the maximum
        to be used for quadratic fitting from which peak location is computed.
        It is assumed that fitting box is a square with sides of length
        given by ``peak_fit_box``.

    mask: numpy.ndarray, optional
        A boolean type `~numpy.ndarray` indicating "good" pixels in image data
        (`True`) and "bad" pixels (`False`). If not provided all pixels
        in `image_data` will be used for fitting.

    Returns
    -------
    coord: tuple of float
        A pair of coordinates of the peak.

    fit_status: str
        Status of the peak search. Currently the following values can be
        returned:

        - ``'SUCCESS'``: Fit was successful and peak is not on the edge of
          the input array;
        - ``'ERROR:NODATA'``: Not enough valid data to perform the fit; The
          returned coordinate is the center of input array;
        - ``'WARNING:EDGE'``: Peak lies on the edge of the input array.
          Returned coordinates are the result of a discreet search;
        - ``'WARNING:BADFIT'``: Performed fid did not find a maximum. Returned
          coordinates are the result of a discreet search;
        - ``'WARNING:CENTER-OF-MASS'``: Returned coordinates are the result
          of a center-of-mass estimate instead of a polynomial fit. This is
          either due to too few points to perform a fit or due to a
          failure of the polynomial fit.

    fit_box: a tuple of two tuples
        A tuple of two tuples of the form ``((x1, x2), (y1, y2))`` that
        indicates pixel ranges used for fitting (these indices can be used
        directly for slicing input data)

    """
    # check arguments:
    if peak_fit_box < 1:
        raise ValueError("peak_fit_box must be at least 1 pixel in size.")
    data = np.asarray(data, dtype=np.double)
    ny, nx = data.shape

    # find index of the pixel having maximum value:
    if mask is None:
        jmax, imax = np.unravel_index(np.argmax(data), data.shape)
        coord = (float(imax), float(jmax))

    else:
        j, i = np.indices(data.shape)
        i = i[mask]
        j = j[mask]

        if i.size == 0:
            # no valid data:
            coord = ((nx - 1.0) / 2.0, (ny - 1.0) / 2.0)
            return coord, 'ERROR:NODATA', np.s_[0:ny, 0:nx]

        ind = np.argmax(data[mask])
        imax = i[ind]
        jmax = j[ind]
        coord = (float(imax), float(jmax))

    if data[jmax, imax] < 1:
        # no valid data: we need some counts in the histogram bins
        coord = ((nx - 1.0) / 2.0, (ny - 1.0) / 2.0)
        return coord, 'ERROR:NODATA', np.s_[0:ny, 0:nx]

    # choose a box around maxval pixel:
    x1 = max(0, imax - peak_fit_box // 2)
    x2 = min(nx, x1 + peak_fit_box)
    y1 = max(0, jmax - peak_fit_box // 2)
    y2 = min(ny, y1 + peak_fit_box)

    # if peak is at the edge of the box, return integer indices of the max:
    if imax == x1 or imax == x2 - 1 or jmax == y1 or jmax == y2 - 1:
        return (float(imax), float(jmax)), 'WARNING:EDGE', np.s_[y1:y2, x1:x2]

    # expand the box if needed:
    if (x2 - x1) < peak_fit_box:  # pragma: no branch
        if x1 == 0:  # pragma: no branch
            x2 = min(nx, x1 + peak_fit_box)
        if x2 == nx:  # pragma: no branch
            x1 = max(0, x2 - peak_fit_box)

    if (y2 - y1) < peak_fit_box:  # pragma: no branch
        if y1 == 0:  # pragma: no branch
            y2 = min(ny, y1 + peak_fit_box)
        if y2 == ny:  # pragma: no branch
            y1 = max(0, y2 - peak_fit_box)

    assert x2 - x1 > 0 or y2 - y1 > 0

    # fit a 2D 2nd degree polynomial to data:
    xi = np.arange(x1, x2)
    yi = np.arange(y1, y2)
    x, y = np.meshgrid(xi, yi)
    x = x.ravel()
    y = y.ravel()
    v = np.vstack((np.ones_like(x), x, y, x * y, x * x, y * y)).T
    d = data[y1:y2, x1:x2].ravel()
    if mask is not None:
        m = mask[y1:y2, x1:x2].ravel()
        v = v[m]
        d = d[m]

    if d.size < 6:
        # we need at least 6 points to fit a 2D quadratic polynomial
        # attempt center-of-mass instead:
        m = np.logical_not(np.isfinite(d))
        vx = v[:, 1].flatten()
        vy = v[:, 2].flatten()
        d[m] = 0
        vx[m] = 0
        vy[m] = 0
        dt = d.sum()
        if dt == 0.0:
            coord = ((nx - 1.0) / 2.0, (ny - 1.0) / 2.0)
            return coord, 'ERROR:NODATA', np.s_[y1:y2, x1:x2]
        xc = np.dot(vx, d) / dt
        yc = np.dot(vy, d) / dt
        return (xc, yc), 'WARNING:CENTER-OF-MASS', np.s_[y1:y2, x1:x2]

    try:
        c = np.linalg.lstsq(v, d, rcond=None)[0]
        if not np.all(np.isfinite(c)):
            raise np.linalg.LinAlgError("Results of the fit are not finite.")
    except np.linalg.LinAlgError as e:
        print("WARNING: Least squares failed!\n{}".format(e))

        # attempt center-of-mass instead:
        m = np.logical_not(np.isfinite(d))
        vx = v[:, 1].flatten()
        vy = v[:, 2].flatten()
        d[m] = 0
        vx[m] = 0
        vy[m] = 0
        dt = d.sum()
        if dt == 0.0:
            coord = ((nx - 1.0) / 2.0, (ny - 1.0) / 2.0)
            return coord, 'ERROR:NODATA', np.s_[y1:y2, x1:x2]
        xc = np.dot(vx, d) / dt
        yc = np.dot(vy, d) / dt
        return (xc, yc), 'WARNING:CENTER-OF-MASS', np.s_[y1:y2, x1:x2]

    # find maximum of the polynomial:
    _, c10, c01, c11, c20, c02 = c
    det = 4 * c02 * c20 - c11**2
    if det <= 0 or ((c20 > 0.0 and c02 >= 0.0) or (c20 >= 0.0 and c02 > 0.0)):
        # polynomial does not have max. return maximum value in the data:
        return coord, 'WARNING:BADFIT', np.s_[y1:y2, x1:x2]

    xm = (c01 * c11 - 2.0 * c02 * c10) / det
    ym = (c10 * c11 - 2.0 * c01 * c20) / det

    if 0.0 <= xm <= (nx - 1.0) and 0.0 <= ym <= (ny - 1.0):
        coord = (xm, ym)
        fit_status = 'SUCCESS'

    else:
        xm = 0.0 if xm < 0.0 else min(xm, nx - 1.0)
        ym = 0.0 if ym < 0.0 else min(ym, ny - 1.0)
        fit_status = 'WARNING:EDGE'

    return coord, fit_status, np.s_[y1:y2, x1:x2]
