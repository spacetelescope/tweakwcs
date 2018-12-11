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

import stsci.imagestats as imagestats
from stsci.stimage import xyxymatch

try:
    from . import chelp
except ImportError:
    chelp = None

from . import __version__, __version_date__

__author__ = 'Mihai Cara'

__all__ = ['MatchCatalogs', 'TPMatch', 'center_of_mass', 'build_xy_zeropoint',
           'find_xy_peak']

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

        refcat : astropy.table.Table
            A reference source catalog. Reference catalog must contain
            ``'TPx'`` and ``'TPy'`` columns that provide undistorted
            (distortion-correction applied) source coordinates coordinate
            system common (shared) with the image catalog ``imcat``.

        imcat : astropy.table.Table
            Source catalog associated with an image. Image catalog must contain
            ``'TPx'`` and ``'TPy'`` columns that provide undistorted
            (distortion-correction applied) source coordinates coordinate
            system common (shared) with the image catalog ``refcat``.

        Returns
        -------
        (refcat_idx, imcat_idx) : tuple of numpy.ndarray
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
    def __init__(self, searchrad=1.0, separation=0.5, use2dhist=True,
                 xoffset=0.0, yoffset=0.0, tolerance=1.0):
        """
        Parameters
        ----------

        searchrad : float, optional
            The search radius for a match (in units of the tangent plane).

        separation : float, optional
            The  minimum  separation in the tangent plane (in units of
            the tangent plane) for sources in the image and reference
            catalogs in order to be considered to be disctinct sources.
            Objects closer together than ``separation`` distance
            are removed from the image and reference coordinate catalogs prior
            to matching. This parameter gets passed directly to
            :py:func:`~stsci.stimage.xyxymatch` for use in matching the object
            lists from each image with the reference catalog's object list.

        use2dhist : bool, optional
            Use 2D histogram to find initial offset?

        xoffset : float, optional
            Initial estimate for the offset in X (in units of the tangent
            plane) between the sources in the image and the reference catalogs.
            This offset will be used for all input images provided.
            This parameter is ignored when ``use2dhist`` is `True`.

        yoffset : float, optional
            Initial estimate for the offset in Y (in units of the tangent
            plane) between the sources in the image and the reference catalogs.
            This offset will be used for all input images provided.
            This parameter is ignored when ``use2dhist`` is `True`.

        tolerance : float, optional
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
        """ Performs catalog matching.

        Parameters
        ----------

        refcat : astropy.table.Table
            A reference source catalog. When a tangent-plane ``WCS`` is
            provided through ``tp_wcs``, the catalog must contain ``'RA'`` and
            ``'DEC'`` columns which indicate reference source world
            coordinates (in degrees). Alternatively, when ``tp_wcs`` is `None`,
            reference catalog must contain ``'TPx'`` and ``'TPy'`` columns that
            provide undistorted (distortion-correction applied) source
            coordinates in some *tangent plane*. In this case, the ``'RA'``
            and ``'DEC'`` columns in the ``refcat`` catalog will be ignored.

        imcat : astropy.table.Table
            Source catalog associated with an image. Must contain ``'x'`` and
            ``'y'`` columns which indicate source coordinates (in pixels) in
            the associated image. Alternatively, when ``tp_wcs`` is `None`,
            catalog must contain ``'TPx'`` and ``'TPy'`` columns that
            provide undistorted (distortion-correction applied) source
            coordinates in **the same**\ *tangent plane* used to define
            ``refcat``'s tangent plane coordinates. In this case, the ``'x'``
            and ``'y'`` columns in the ``imcat`` catalog will be ignored.

        tp_wcs : TPWCS, None, optional
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
        (refcat_idx, imcat_idx) : tuple of numpy.ndarray
            A tuple of two 1D `numpy.ndarray` containing indices of matched
            sources in the ``refcat`` and ``imcat`` catalogs accordingly.

        """
        # Check catalogs:
        if not isinstance(refcat, astropy.table.Table):
            raise TypeError("'refcat' must be an instance of "
                            "astropy.table.Table")

        if len(refcat) < 1:
            raise ValueError("Reference catalog must contain at least one "
                             "source.")

        if not isinstance(imcat, astropy.table.Table):
            raise TypeError("'imcat' must be an instance of "
                            "astropy.table.Table")

        if len(imcat) < 1:
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

        tolerance = self._tolerance

        ps = 1.0 if tp_wcs is None else tp_wcs.tanp_center_pixel_scale

        if self._use2dhist:
            # Determine xyoff (X,Y offset) and tolerance
            # to be used with xyxymatch:
            zpxoff, zpyoff, flux, zpqual = build_xy_zeropoint(
                imxy / ps,
                refxy / ps,
                searchrad=self._searchrad
            )

            if zpqual is None:
                xyoff = (0.0, 0.0)
            else:
                xyoff = (zpxoff * ps, zpyoff * ps)

        else:
            xyoff = (self._xoffset * ps, self._yoffset * ps)

        matches = xyxymatch(
            imxy,
            refxy,
            origin=xyoff,
            tolerance=ps * tolerance,
            separation=ps * self._separation
        )

        return matches['ref_idx'], matches['input_idx']


def center_of_mass(img, labels=None, index=None):
    """
    Calculate the center of mass of the values of an array at labels.

    Parameters
    ----------
    img : ndarray
        Data from which to calculate center-of-mass.

    Returns
    -------
    centerofmass : tuple, or list of tuples
        Coordinates of centers-of-masses.

    Examples
    --------
    >>> from tweakwcs import matchutils
    >>> a = np.array(([0,0,0,0],
                      [0,1,1,0],
                      [0,1,1,0],
                      [0,1,1,0]))
    >>> matchutils.center_of_mass(a)
    (2.0, 1.5)

    """
    normalizer = img.sum(dtype=np.float64)
    if normalizer == 0.0:
        invnorm = np.nan
    else:
        invnorm = 1.0 / normalizer

    grids = np.ogrid[[slice(0, i) for i in img.shape]]

    results = [(img * grids[d].astype(np.float)).sum(dtype=np.float64) *
               invnorm for d in range(img.ndim)]

    if np.isscalar(results[0]):
        return tuple(results)

    return [tuple(v) for v in np.array(results).T]


def build_xy_zeropoint(imgxy, refxy, searchrad=3.0):
    """ Create a matrix which contains the delta between each XY position and
        each UV position.
    """
    log.info("Computing initial guess for X and Y shifts...")

    if chelp is None:
        raise ImportError('cannot import chelp')

    # run C function to create ZP matrix
    zpmat = chelp.arrxyzero(imgxy.astype(np.float32),
                            refxy.astype(np.float32), searchrad)

    xp, yp, flux, zpqual = find_xy_peak(zpmat, center=(searchrad, searchrad))
    if zpqual is None:
        # try with a lower sigma to detect a peak in a sparse set of sources
        xp, yp, flux, zpqual = find_xy_peak(
            zpmat,
            center=(searchrad, searchrad),
            sigma=1.0
        )

    if zpqual is None:
        log.warning("No valid shift found within a search radius of {:g} "
                    "pixels.".format(searchrad))
    else:
        log.info("Found initial X and Y shifts of {:.4g}, {:.4g} "
                 "with significance of {:.4g} and {} matches"
                 .format(xp, yp, zpqual, flux))

    return xp, yp, flux, zpqual


def find_xy_peak(img, center=None, sigma=3.0):
    """ Find the center of the peak of offsets
    """
    # find level of noise in histogram
    istats = imagestats.ImageStats(img, nclip=1,
                                   fields='stddev,mode,mean,max,min')

    if istats.stddev == 0.0:
        istats = imagestats.ImageStats(img, fields='stddev,mode,mean,max,min')

    imgsum = img.sum()

    # clip out all values below mean+3*sigma from histogram
    imgc = img[:, :].copy()
    imgc[imgc < istats.mode + istats.stddev * sigma] = 0.0

    # identify position of peak
    yp0, xp0 = np.where(imgc == imgc.max())

    # Perform bounds checking on slice from img
    ymin = max(0, int(yp0[0]) - 3)
    ymax = min(img.shape[0], int(yp0[0]) + 4)
    xmin = max(0, int(xp0[0]) - 3)
    xmax = min(img.shape[1], int(xp0[0]) + 4)

    # take sum of at most a 7x7 pixel box around peak
    xp_slice = (slice(ymin, ymax), slice(xmin, xmax))
    yp, xp = center_of_mass(img[xp_slice])

    if np.isnan(xp) or np.isnan(yp):
        xp = 0.0
        yp = 0.0
        flux = 0.0
        zpqual = None

    else:
        xp += xp_slice[1].start
        yp += xp_slice[0].start

        # compute S/N criteria for this peak: flux/sqrt(mean of rest of array)
        flux = imgc[xp_slice].sum()

        delta_size = float(img.size - imgc[xp_slice].size)
        if delta_size == 0:
            delta_size = 1

        delta_flux = float(imgsum - flux)

        if flux > imgc[xp_slice].max():
            delta_flux = flux - imgc[xp_slice].max()
        else:
            delta_flux = flux

        if delta_flux > 0.0 and delta_size > 0.0:
            zpqual = flux / np.sqrt(delta_flux / delta_size)
            if np.isnan(zpqual) or np.isinf(zpqual):
                zpqual = None
        else:
            zpqual = None

        if center is not None:
            xp -= center[0]
            yp -= center[1]

        flux = imgc[xp_slice].max()

    return xp, yp, flux, zpqual
