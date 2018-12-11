# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides support for manipulating tangent-plane corrections
of ``WCS``.

:Authors: Mihai Cara (contact: help@stsci.edu)

:License: :doc:`../LICENSE`

"""
# STDLIB
import logging
import sys
from copy import deepcopy
from abc import ABC, abstractmethod

# THIRD-PARTY
import numpy as np
import gwcs
from astropy import wcs as fitswcs

try:
    from jwst.transforms.tpcorr import TPCorr  # pylint: disable=W0611
except:
    TPCorr = None

# LOCAL
from . import __version__, __version_date__

__author__ = 'Mihai Cara'

__all__ = ['TPWCS', 'JWSTgWCS', 'FITSWCS']


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class TPWCS(ABC):
    """ A class that provides common interface for manipulating WCS information
    and for managing tangent-plane corrections.

    """
    def __init__(self, wcs):
        """
        Parameters
        ----------

        wcs : ``WCS object``
            A `WCS` object supported by a child class.

        """
        self._owcs = wcs
        self._wcs = deepcopy(wcs)
        self._meta = {}

    @property
    def wcs(self):
        """ Get current WCS object. """
        return self._wcs

    @property
    def original_wcs(self):
        """ Get original WCS object. """
        return self._owcs

    def copy(self):
        """ Returns a deep copy of this object. """
        return deepcopy(self)

    @abstractmethod
    def set_correction(self, matrix=[[1, 0], [0, 1]], shift=[0, 0], meta=None,
                       **kwargs):
        """
        Sets a tangent-plane correction of the WCS object according to
        the provided liniar parameters. In addition, this function updates
        the ``meta`` attribute of the `TPWCS`-derived object with the values
        of keyword arguments except for the argument ``meta`` which is
        *merged* with the *attribute* ``meta``.

        Parameters
        ----------
        matrix : list, numpy.ndarray, optional
            A ``2x2`` array or list of lists coefficients representing scale,
            rotation, and/or skew transformations.

        shift : list, numpy.ndarray, optional
            A list of two coordinate shifts to be applied to coordinates
            *before* ``matrix`` transformations are applied.

        meta : dict, None, optional
            Dictionary that will be merged to the object's ``meta`` fields.

        **kwargs : optional parameters
            Optional parameters for the WCS corrector.

        """
        pass

    @abstractmethod
    def det_to_world(self, x, y):
        """
        Convert pixel coordinates to sky coordinates using full
        (i.e., including distortions) transformations.

        """
        ra, dec = x, y
        return ra, dec

    @abstractmethod
    def world_to_det(self, ra, dec):
        """
        Convert sky coordinates to image's pixel coordinates using full
        (i.e., including distortions) transformations.

        """
        x, y = ra, dec
        return x, y

    @abstractmethod
    def det_to_tanp(self, x, y):
        """
        Convert detector (pixel) coordinates to tangent plane coordinates.

        """
        return x, y

    @abstractmethod
    def tanp_to_det(self, x, y):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        return x, y

    @abstractmethod
    def world_to_tanp(self, ra, dec):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        x, y = ra, dec
        return x, y

    @abstractmethod
    def tanp_to_world(self, x, y):
        """
        Convert tangent plane coordinates to world coordinates.

        """
        ra, dec = x, y
        return ra, dec

    def tanp_pixel_scale(self, x, y):
        """ Estimate pixel scale in the tangent plane near a location
        in the detector's coordinate system given by parameters ``x`` and
        ``y``.

        Parameters
        ----------
        x : int, float
            X-coordinate of the pixel in the detector's coordinate system
            near which pixel scale in the tangent plane needs to be estimated.

        y : int, float
            Y-coordinate of the pixel in the detector's coordinate system
            near which pixel scale in the tangent plane needs to be estimated.

        Returns
        -------

        pscale : float
            Pixel scale [in units used in the tangent plane coordinate system]
            in the tangent plane near a location specified by detector
            coordinates ``x`` and ``y``.

        """
        xs = np.asarray([x - 0.5, x - 0.5, x + 0.5, x + 0.5])
        ys = np.asarray([y - 0.5, y + 0.5, y + 0.5, y - 0.5])
        xt, yt = self.det_to_tanp(xs, ys)
        area = 0.5 * np.abs(
            xt[0] * yt[1] + xt[1] * yt[2] + xt[2] * yt[3] + xt[3] * yt[0] -
            xt[1] * yt[0] - xt[2] * yt[1] - xt[3] * yt[2] - xt[0] * yt[3]
        )
        pscale = float(np.sqrt(area))
        return pscale

    @property
    def tanp_center_pixel_scale(self):
        """ Estimate pixel scale in the tangent plane near a location
        in the detector's coordinate system corresponding to the origin of the
        tangent plane.

        Returns
        -------

        pscale : float
            Pixel scale [in units used in the tangent plane coordinate system]
            in the tangent plane near a location in the detector's plane
            corresponding to the origin of the tangent plane.

        """
        x, y = self.tanp_to_det(0.0, 0.0)
        pscale = self.tanp_pixel_scale(x, y)
        return pscale

    @property
    def meta(self):
        return self._meta


class JWSTgWCS(TPWCS):
    """ A class for holding ``JWST`` ``gWCS`` information and for managing
    tangent-plane corrections.

    """
    def __init__(self, wcs, wcsinfo):
        """
        Parameters
        ----------

        wcs : GWCS
            A `GWCS` object.

        wcsinfo : dict
            A dictionary containing reference angles of ``JWST`` instrument
            provided in the ``wcsinfo`` property of ``JWST`` meta data.

            This dictionary **must contain** the following keys and values:


            'v2_ref' : float
                V2 position of the reference point in arc seconds.

            'v3_ref' : float
                V3 position of the reference point in arc seconds.

            'roll_ref' : float
                Roll angle in degrees.

        """
        if TPCorr is None:
            raise ImportError("The 'jwst' package must be installed in order "
                              "to correct JWST WCS.")

        valid, message =  self._check_wcs_structure(wcs)
        if not valid:
            raise ValueError("Unsupported WCS structure: {}".format(message))

        v2_ref = wcsinfo['v2_ref']
        v3_ref = wcsinfo['v3_ref']
        roll_ref = wcsinfo['roll_ref']

        self._wcsinfo = {'v2_ref': v2_ref, 'v3_ref': v3_ref,
                         'roll_ref': roll_ref}

        # perform additional check that if tangent plane correction is already
        # present in the WCS pipeline, it is of TPCorr class and that
        # its parameters are consistent with reference angles:
        frms = wcs.available_frames
        if 'v2v3corr' in frms:
            self._v23name = 'v2v3corr'
            self._tpcorr = deepcopy(
                wcs.pipeline[frms.index('v2v3corr') - 1][1]
            )
            self._default_tpcorr = None
            if not isinstance(self._tpcorr, TPCorr):
                raise ValueError("Unsupported tangent-plance correction type.")

            # check that transformation parameters are consistent with
            # reference angles:
            v2ref = self._tpcorr.v2ref.value
            v3ref = self._tpcorr.v3ref.value
            roll = self._tpcorr.roll.value
            eps_v2 = 10.0 * np.finfo(v2_ref).eps
            eps_v3 = 10.0 * np.finfo(v3_ref).eps
            eps_roll = 10.0 * np.finfo(roll_ref).eps
            if not (np.isclose(v2_ref, v2ref * 3600.0, rtol=eps_v2) and
                    np.isclose(v3_ref, v3ref * 3600.0, rtol=eps_v3) and
                    np.isclose(roll_ref, roll, rtol=eps_roll)):
                raise ValueError(
                    "WCS/TPCorr parameters 'v2ref', 'v3ref', and/or 'roll' "
                    "differ from the corresponding reference values."
                )

        else:
            self._v23name = 'v2v3'
            self._tpcorr = None
            self._default_tpcorr = TPCorr(
                v2ref=v2_ref / 3600.0, v3ref=v3_ref / 3600.0, roll=roll_ref,
                name='tangent-plane linear correction'
            )

        super().__init__(wcs)
        self._update_transformations()

    def _update_transformations(self):
        # define transformations from detector/world coordinates to
        # the tangent plane:
        detname = self._wcs.pipeline[0][0].name
        worldname = self._wcs.pipeline[-1][0].name

        self._world_to_v23 = self._wcs.get_transform(worldname, self._v23name)
        self._v23_to_world = self._wcs.get_transform(self._v23name, worldname)
        self._det_to_v23 = self._wcs.get_transform(detname, self._v23name)
        self._v23_to_det = self._wcs.get_transform(self._v23name, detname)

        self._det_to_world = self._wcs.__call__
        self._world_to_det = self._wcs.invert

    @property
    def ref_angles(self):
        """ Return a ``wcsinfo``-like dictionary of main WCS parameters. """
        return {k: v for k, v in self._wcsinfo.items()}

    def set_correction(self, matrix=[[1, 0], [0, 1]], shift=[0, 0], meta=None,
                       **kwargs):
        """
        Sets a tangent-plane correction of the GWCS object according to
        the provided liniar parameters. In addition, this function updates
        the ``meta`` attribute of the `JWSTgWCS` object with the values
        of keyword arguments except for the argument ``meta`` which is
        *merged* with the *attribute* ``meta``.

        Parameters
        ----------
        matrix : list, numpy.ndarray
            A ``2x2`` array or list of lists coefficients representing scale,
            rotation, and/or skew transformations.

        shift : list, numpy.ndarray
            A list of two coordinate shifts to be applied to coordinates
            *before* ``matrix`` transformations are applied.

        meta : dict, None, optional
            Dictionary that will be merged to the object's ``meta`` fields.

        **kwargs : optional parameters
            Optional parameters for the WCS corrector. `JWSTgWCS` ignores these
            arguments (except for storing them in the ``meta`` attribute).

        """
        frms = self._wcs.available_frames

        # if original WCS did not have tangent-plane corrections, create
        # new correction and add it to the WCs pipeline:
        if self._tpcorr is None:
            self._tpcorr = TPCorr(
                v2ref=self._wcsinfo['v2_ref'] / 3600.0,
                v3ref=self._wcsinfo['v3_ref'] / 3600.0,
                roll=self._wcsinfo['roll_ref'],
                matrix=matrix,
                shift=shift,
                name='tangent-plane linear correction'
            )
            idx_v2v3 = frms.index(self._v23name)
            pipeline = deepcopy(self._wcs.pipeline)
            pf, pt = pipeline[idx_v2v3]
            pipeline[idx_v2v3] = (pf, deepcopy(self._tpcorr))
            frm_v2v3corr = deepcopy(pf)
            frm_v2v3corr.name = 'v2v3corr'
            pipeline.insert(idx_v2v3 + 1, (frm_v2v3corr, pt))
            self._wcs = gwcs.WCS(pipeline, name=self._owcs.name)
            self._v23name = 'v2v3corr'

        else:
            # combine old and new corrections into a single one and replace
            # old transformation with the combined correction transformation:
            tpcorr2 = self._tpcorr.__class__(
                v2ref=self._tpcorr.v2ref, v3ref=self._tpcorr.v3ref,
                roll=self._tpcorr.roll, matrix=matrix, shift=shift,
                name='tangent-plane linear correction'
            )

            self._tpcorr = tpcorr2.combine(tpcorr2, self._tpcorr)

            idx_v2v3 = frms.index(self._v23name)
            pipeline = deepcopy(self._wcs.pipeline)
            pipeline[idx_v2v3 - 1] = (pipeline[idx_v2v3 - 1][0],
                                      deepcopy(self._tpcorr))
            self._wcs = gwcs.WCS(pipeline, name=self._owcs.name)

        # reset definitions of the transformations from detector/world
        # coordinates to the tangent plane:
        self._update_transformations()

        # save linear transformation info to the meta attribute:
        self._meta['matrix'] = matrix
        self._meta['shift'] = shift
        if meta is not None:
            self._meta.update(meta)

    def _check_wcs_structure(self, wcs):
        if wcs is None or wcs.pipeline is None:
            return False, "Either WCS or its pipeline is None."

        frms = wcs.available_frames
        nframes = len(frms)

        if nframes < 3:
            return False, "There are fewer than 3 frames in the WCS pipeline."

        if frms.count(frms[0]) > 1 or frms.count(frms[-1]) > 1:
            return (False, "First and last frames in the WCS pipeline must "
                    "be unique.")

        if frms.count('v2v3') != 1:
            return (False, "Only one 'v2v3' frame is allowed in the WCS "
                    "pipeline.")

        idx_v2v3 = frms.index('v2v3')
        if idx_v2v3 == 0 or idx_v2v3 == (nframes - 1):
            return (False, "'v2v3' frame cannot be first or last in the WCS "
                    "pipeline.")

        nv2v3corr = frms.count('v2v3corr')
        if nv2v3corr == 0:
            return True, ''
        elif nv2v3corr > 1:
            return (False, "Only one 'v2v3corr' correction frame is allowed "
                    "in the WCS pipeline.")

        idx_v2v3corr = frms.index('v2v3corr')
        if idx_v2v3corr != (idx_v2v3 + 1) or idx_v2v3corr == (nframes - 1):
            return (False, "'v2v3corr' frame is not in the correct position "
                    "in the WCS pipeline.")

        return True, ''


    def det_to_world(self, x, y):
        """
        Convert pixel coordinates to sky coordinates using full
        (i.e., including distortions) transformations.

        """
        ra, dec = self._det_to_world(x, y)
        return ra, dec

    def world_to_det(self, ra, dec):
        """
        Convert sky coordinates to image's pixel coordinates using full
        (i.e., including distortions) transformations.

        """
        x, y = self._world_to_det(ra, dec)
        return x, y

    def det_to_tanp(self, x, y):
        """
        Convert detector (pixel) coordinates to tangent plane coordinates.

        """
        tpc = self._default_tpcorr if self._tpcorr is None else self._tpcorr
        v2, v3 = self._det_to_v23(x, y)
        x, y = tpc.v2v3_to_tanp(v2, v3)
        return x, y

    def tanp_to_det(self, x, y):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        tpc = self._default_tpcorr if self._tpcorr is None else self._tpcorr
        v2, v3 = tpc.tanp_to_v2v3(x, y)
        x, y = self._v23_to_det(v2, v3)
        return x, y

    def world_to_tanp(self, ra, dec):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        tpc = self._default_tpcorr if self._tpcorr is None else self._tpcorr
        v2, v3 = self._world_to_v23(ra, dec)
        x, y = tpc.v2v3_to_tanp(v2, v3)
        return x, y

    def tanp_to_world(self, x, y):
        """
        Convert tangent plane coordinates to world coordinates.

        """
        tpc = self._default_tpcorr if self._tpcorr is None else self._tpcorr
        v2, v3 = tpc.tanp_to_v2v3(x, y)
        ra, dec = self._v23_to_world(v2, v3)
        return ra, dec


class FITSWCS(TPWCS):
    """ A class for holding ``FITS`` ``WCS`` information and for managing
    tangent-plane corrections.

    .. note::
        Currently only WCS objects that have ``CPDIS``, ``DET2IM``, and ``SIP``
        distortions *before* the application of the ``CD`` or ``PC`` matrix are
        supported.

    """
    def __init__(self, wcs):
        """
        Parameters
        ----------

        wcs : astropy.wcs.WCS
            An `astropy.wcs.WCS` object.

        """
        valid, message =  self._check_wcs_structure(wcs)
        if not valid:
            raise ValueError("Unsupported WCS structure." + message)

        super().__init__(wcs)

        self._owcs = wcs.deepcopy()
        self._wcs = wcs.deepcopy()
        wcslin = wcs.deepcopy()

        # strip all *known* distortions:
        wcslin.cpdis1 = None
        wcslin.cpdis2 = None
        wcslin.det2im1 = None
        wcslin.det2im2 = None
        wcslin.sip = None
        wcslin.wcs.set()

        self._wcslin = wcslin

    def _check_wcs_structure(self, wcs):
        """
        Attempt detecting unknown distortion corrections. We basically
        want to make sure that we can turn off all distortions that are
        happening between detector's plane and the intermediate coordinate
        plane. This is necessary until we can find a better way of getting
        from intermediate coordinates to world coordinates.

        """
        if wcs is None:
            return False, "WCS cannot be None."

        if not wcs.is_celestial:
            return False, "WCS must be exclusively a celestial WCS."

        wcs = wcs.deepcopy()
        naxis1, naxis2 = wcs.pixel_shape

        # check mapping of corners and CRPIX:
        pts = np.array([[1.0, 1.0], [1.0, naxis2], [naxis1, 1.0],
                        [naxis1, naxis2], wcs.wcs.crpix])

        sky_all = wcs.all_pix2world(pts, 1)
        sky_wcs = wcs.wcs_pix2world(pts, 1)
        foc_all = wcs.pix2foc(pts, 1)
        crpix = np.array(wcs.wcs.crpix)

        # strip all *known* distortions:
        wcs.cpdis1 = None
        wcs.cpdis2 = None
        wcs.det2im1 = None
        wcs.det2im2 = None
        wcs.sip = None

        # check that pix2foc includes no other distortions besides the ones
        # that we have turned off above:
        if not np.allclose(pts, wcs.pix2foc(pts, 1)):
            False, "'pix2foc' contains unknow distortions"

        wcs.wcs.set()

        # check that pix2foc contains all known distortions:
        test2 = np.allclose(
            wcs.all_world2pix(sky_all, 1), foc_all, atol=1e-3, rtol=0
        )
        if not test2:
            return False, "'WCS.pix2foc()' does not include all distortions."

        return True, ''

    def set_correction(self, matrix=[[1, 0], [0, 1]], shift=[0, 0], meta=None,
                       **kwargs):
        """
        Computes a corrected (aligned) wcs based on the provided linear
        transformation. In addition, this function updates the ``meta``
        attribute of the `FITSWCS` object with the the values of keyword
        arguments except for the argument ``meta`` which is *merged* with
        the *attribute* ``meta``.

        Parameters
        ----------
        matrix : list, numpy.ndarray
            A ``2x2`` array or list of lists coefficients representing scale,
            rotation, and/or skew transformations.

        shift : list, numpy.ndarray
            A list of two coordinate shifts to be applied to coordinates
            *before* ``matrix`` transformations are applied.

        meta : dict, None, optional
            Dictionary that will be merged to the object's ``meta`` fields.

        **kwargs : optional parameters
            Optional parameters for the WCS corrector. `FITSWCS` ignores these
            arguments (except for storing them in the ``meta`` attribute).

        """
        # compute the matrix for the scale and rotation correction
        shift = (np.asarray(shift) - np.dot(self._wcslin.wcs.crpix, matrix) +
                 self._wcslin.wcs.crpix)

        if matrix.shape == (2, 2):
            matrix = _inv2x2(matrix).T
        else:
            matrix = np.linalg.inv(matrix).T

        cwcs = self._wcs.deepcopy()
        cd_eye = np.eye(self._wcs.wcs.cd.shape[0], dtype=np.longdouble)
        zero_shift = np.zeros(2, dtype=np.longdouble)

        # estimate precision necessary for iterative processes:
        maxiter = 100
        crpix2corners = np.dstack([i.flatten() for i in np.meshgrid(
            [1, self._wcs.pixel_shape[0]],
            [1, self._wcs.pixel_shape[1]])])[0] - self._wcs.wcs.crpix
        maxUerr = 1.0e-5 / np.amax(np.linalg.norm(crpix2corners, axis=1))

        # estimate step for numerical differentiation. We need a step
        # large enough to avoid rounding errors and small enough to get a
        # better precision for numerical differentiation.
        # TODO: The logic below should be revised at a later time so that it
        # better takes into account the two competing requirements.
        hx = max(1.0,
                 min(20.0, (self._wcs.wcs.crpix[0] - 1.0) / 100.0,
                     (self._wcs.pixel_shape[0] - self._wcs.wcs.crpix[0]) / 100.0))
        hy = max(1.0,
                 min(20.0, (self._wcs.wcs.crpix[1] - 1.0) / 100.0,
                     (self._wcs.pixel_shape[1] - self._wcs.wcs.crpix[1]) / 100.0))

        # compute new CRVAL for the image WCS:
        crpixinref = self._wcslin.wcs_world2pix(
            self._wcs.wcs_pix2world([self._wcs.wcs.crpix],1),1)
        crpixinref = np.dot(matrix, (crpixinref - shift).T).T
        self._wcs.wcs.crval = self._wcslin.wcs_pix2world(crpixinref, 1)[0]
        self._wcs.wcs.set()

        # initial approximation for CD matrix of the image WCS:
        (U, u) = _linearize(cwcs, self._wcs, self._wcslin, self._wcs.wcs.crpix,
                            matrix, shift, hx=hx, hy=hy)
        err0 = np.amax(np.abs(U - cd_eye)).astype(np.float64)
        self._wcs.wcs.cd = np.dot(self._wcs.wcs.cd.astype(np.longdouble),
                                  U).astype(np.float64)
        self._wcs.wcs.set()

        # save linear transformation info to the meta attribute:
        self._meta['matrix'] = matrix
        self._meta['shift'] = shift
        if meta is not None:
            self._meta.update(meta)

    def det_to_world(self, x, y):
        """
        Convert pixel coordinates to sky coordinates using full
        (i.e., including distortions) transformations.

        """
        ra, dec = self._wcs.all_pix2world(x, y, 0)
        return ra, dec

    def world_to_det(self, ra, dec):
        """
        Convert sky coordinates to image's pixel coordinates using full
        (i.e., including distortions) transformations.

        """
        x, y = self._wcs.all_world2pix(ra, dec, 0)
        return x, y

    def det_to_tanp(self, x, y):
        """
        Convert detector (pixel) coordinates to tangent plane coordinates.

        """
        crpix1, crpix2 = self._wcs.wcs.crpix - 1
        x, y = self._wcs.pix2foc(x, y, 0)
        x -= crpix1
        y -= crpix2
        return x, y

    def tanp_to_det(self, x, y):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        crpix1, crpix2 = self._wcs.wcs.crpix
        x = x + crpix1
        y = y + crpix2
        ra, dec = self._wcslin.all_pix2world(x, y, 1)
        x, y = self._wcslin.all_world2pix(ra, dec, 0)
        return x, y

    def world_to_tanp(self, ra, dec):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        crpix1, crpix2 = self._wcs.wcs.crpix
        x, y = self._wcslin.all_world2pix(ra, dec, 1)
        x -= crpix1
        y -= crpix2
        return x, y

    def tanp_to_world(self, x, y):
        """
        Convert tangent plane coordinates to world coordinates.

        """
        crpix1, crpix2 = self._wcs.wcs.crpix
        x = x + crpix1
        y = y + crpix2
        ra, dec = self._wcslin.all_pix2world(x, y, 1)
        return ra, dec


def _linearize(wcsim, wcsima, wcsref, imcrpix, f, shift, hx=1.0, hy=1.0):
    """ linearization using 5-point formula for first order derivative. """
    x0 = imcrpix[0]
    y0 = imcrpix[1]
    p = np.asarray([[x0, y0],
                    [x0 - hx, y0],
                    [x0 - hx * 0.5, y0],
                    [x0 + hx * 0.5, y0],
                    [x0 + hx, y0],
                    [x0, y0 - hy],
                    [x0, y0 - hy * 0.5],
                    [x0, y0 + hy * 0.5],
                    [x0, y0 + hy]],
                   dtype=np.float64)
    # convert image coordinates to reference image coordinates:
    p = wcsref.wcs_world2pix(wcsim.wcs_pix2world(p, 1), 1).astype(np.longdouble)
    # apply linear fit transformation:
    p = np.dot(f, (p - shift).T).T
    # convert back to image coordinate system:
    p = wcsima.wcs_world2pix(
        wcsref.wcs_pix2world(p.astype(np.float64), 1), 1).astype(np.longdouble)

    # derivative with regard to x:
    u1 = ((p[1] - p[4]) + 8 * (p[3] - p[2])) / (6 * hx)
    # derivative with regard to y:
    u2 = ((p[5] - p[8]) + 8 * (p[7] - p[6])) / (6 * hy)

    return (np.asarray([u1, u2]).T, p[0])


def _inv2x2(x):
    assert(x.shape == (2, 2))
    inv = x.astype(np.longdouble)
    det = inv[0, 0] * inv[1, 1] - inv[0,1] * inv[1, 0]
    if np.abs(det) < np.finfo(np.float64).tiny:
        raise ArithmeticError('Singular matrix.')
    a = inv[0, 0]
    d = inv[1, 1]
    inv[1, 0] *= -1.0
    inv[0, 1] *= -1.0
    inv[0, 0] = d
    inv[1, 1] = a
    inv /= det
    inv = inv.astype(np.float64)
    if not np.all(np.isfinite(inv)):
        raise ArithmeticError('Singular matrix.')
    return inv
