# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides support for manipulating tangent-plane corrections
of ``WCS``.

:Authors: Mihai Cara

:License: :doc:`../LICENSE`

"""
import logging
from copy import deepcopy
from abc import ABC, abstractmethod
from distutils.version import LooseVersion

import numpy as np
import astropy

try:
    import gwcs
    if LooseVersion(gwcs.__version__) > '0.12.0':
        from gwcs.geometry import CartesianToSpherical, SphericalToCartesian
        _GWCS_VER_GT_0P12 = True
    else:
        _GWCS_VER_GT_0P12 = False
except ImportError:
    _GWCS_VER_GT_0P12 = False

if LooseVersion(astropy.__version__) >= '4.0':
    _ASTROPY_VER_GE_4 = True
    from astropy.modeling import CompoundModel
    from astropy.modeling.models import (
        AffineTransformation2D, Scale, Identity, Mapping, Const1D,
        RotationSequence3D
    )
else:
    _ASTROPY_VER_GE_4 = False

from .linalg import inv
from . import __version__  # noqa: F401

__author__ = 'Mihai Cara'

__all__ = ['TPWCS', 'JWSTgWCS', 'FITSWCS']

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

_RAD2ARCSEC = 3600.0 * np.rad2deg(1.0)
_ARCSEC2RAD = 1.0 / _RAD2ARCSEC


def _tp2tp(tpwcs1, tpwcs2, s=None):
    x = np.array([-0.5, 0.5, -0.5, 0.0], dtype=np.double)
    y = np.array([-0.5, -0.5, 0.5, 0.0], dtype=np.double)

    if s is None:
        if 'fit_info' in tpwcs1.meta and 'center' in tpwcs1.meta['fit_info']:
            center = np.array(tpwcs1.meta['fit_info']['center'])
        else:
            if tpwcs2.wcs.pixel_bounds is None:
                # TODO: A possible improvement would be to get an estimate
                #       of "center" (where scale is estimated) from source
                #       positions (if any).
                center = np.zeros(2)
            else:
                center = np.mean(tpwcs2.wcs.pixel_bounds, axis=1)

        xt, yt = tpwcs1.world_to_tanp(*tpwcs2.det_to_world(center[0] + x, center[1] + y))
        m = np.array([(xt[1:-1] - xt[0]), (yt[1:-1] - yt[0])])
        s = np.sqrt(np.fabs(np.linalg.det(m)))

    xrp, yrp = tpwcs2.world_to_tanp(*tpwcs1.tanp_to_world(x * s, y * s))

    xrp = np.array(xrp, np.longdouble)
    yrp = np.array(yrp, np.longdouble)

    matrix = np.array([(xrp[1:-1] - xrp[0]), (yrp[1:-1] - yrp[0])]) / s
    shift = np.array([xrp[-1], yrp[-1]])

    return matrix, shift


class TPWCS(ABC):
    """ A class that provides common interface for manipulating WCS information
    and for managing tangent-plane corrections.

    """

    def __init__(self, wcs, meta=None):
        """
        Parameters
        ----------
        wcs: ``WCS object``
            A `WCS` object supported by a child class.

        meta: dict, None, optional
            Dictionary that will be merged to the object's ``meta`` fields.

        """
        self._owcs = wcs
        self._wcs = deepcopy(wcs)
        self._meta = {} if meta is None else dict(meta)

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
    def set_correction(self, matrix=[[1, 0], [0, 1]], shift=[0, 0],
                       ref_tpwcs=None, meta=None, **kwargs):
        """
        Sets a tangent-plane correction of the WCS object according to
        the provided liniar parameters. In addition, this function updates
        the ``meta`` attribute of the `TPWCS`-derived object with the values
        of keyword arguments except for the argument ``meta`` which is
        *merged* with the *attribute* ``meta``.

        Parameters
        ----------
        matrix: list, numpy.ndarray, optional
            A ``2x2`` array or list of lists coefficients representing scale,
            rotation, and/or skew transformations.

        shift: list, numpy.ndarray, optional
            A list of two coordinate shifts to be applied to coordinates
            *after* ``matrix`` transformations are applied.

        ref_tpwcs: TPWCS, None, optional
            A reference WCS of the type ``TPWCS`` that provides the tangent
            plane in which corrections (``matrix`` and ``shift``) were defined.
            When not provided (i.e., set to `None`), it is assumed that the
            transformations are being applied directly to *this* image WCS'
            tangent plane.

        meta: dict, None, optional
            Dictionary that will be merged to the object's ``meta`` fields.

        **kwargs: optional parameters
            Optional parameters for the WCS corrector.

        """
        # save linear transformation info to the meta attribute:
        self._meta['matrix'] = np.array(matrix, dtype=np.double)
        self._meta['shift'] = shift
        if meta is not None:
            self._meta.update(meta)

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
        x: int, float
            X-coordinate of the pixel in the detector's coordinate system
            near which pixel scale in the tangent plane needs to be estimated.

        y: int, float
            Y-coordinate of the pixel in the detector's coordinate system
            near which pixel scale in the tangent plane needs to be estimated.

        Returns
        -------

        pscale: float
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
        pscale: float
            Pixel scale [in units used in the tangent plane coordinate system]
            in the tangent plane near a location in the detector's plane
            corresponding to the origin of the tangent plane.

        """
        pscale = self._get_tanp_center_pixel_scale()
        return pscale

    def _get_tanp_center_pixel_scale(self):
        x, y = self.tanp_to_det(0.0, 0.0)
        pscale = self.tanp_pixel_scale(x, y)
        return pscale

    @property
    def meta(self):
        return self._meta

    @property
    def bounding_box(self):
        """
        Get the bounding box (if any) of the underlying image for which
        the original WCS is defined.

        """
        return None


class FITSWCS(TPWCS):
    """ A class for holding ``FITS`` ``WCS`` information and for managing
    tangent-plane corrections.

    .. note::
        Currently only WCS objects that have ``CPDIS``, ``DET2IM``, and ``SIP``
        distortions *before* the application of the ``CD`` or ``PC`` matrix are
        supported.

    """
    def __init__(self, wcs, meta=None):
        """
        Parameters
        ----------

        wcs: astropy.wcs.WCS
            An `astropy.wcs.WCS` object.

        """
        valid, message = self._check_wcs_structure(wcs)
        if not valid:
            raise ValueError("Unsupported WCS structure: " + message)

        super().__init__(wcs=wcs, meta=meta)
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
        foc_all = wcs.pix2foc(pts, 1)

        # strip all *known* distortions:
        wcs.cpdis1 = None
        wcs.cpdis2 = None
        wcs.det2im1 = None
        wcs.det2im2 = None
        wcs.sip = None

        # check that pix2foc includes no other distortions besides the ones
        # that we have turned off above:
        if not np.allclose(pts, wcs.pix2foc(pts, 1)):
            False, "'pix2foc' contains unknown distortions"

        wcs.wcs.set()

        # check that pix2foc contains all known distortions:
        if not np.allclose(wcs.all_world2pix(sky_all, 1), foc_all, atol=1e-3,
                           rtol=0):
            return False, "'WCS.pix2foc()' does not include all distortions."

        return True, ''

    def _get_tanp_center_pixel_scale(self):
        pscale = self.tanp_pixel_scale(*self._wcs.wcs.crpix)
        return pscale

    def set_correction(self, matrix=[[1, 0], [0, 1]], shift=[0, 0],
                       ref_tpwcs=None, meta=None, **kwargs):
        """
        Computes a corrected (aligned) wcs based on the provided linear
        transformation. In addition, this function updates the ``meta``
        attribute of the `FITSWCS` object with the the values of keyword
        arguments except for the argument ``meta`` which is *merged* with
        the *attribute* ``meta``.

        Parameters
        ----------
        matrix: list, numpy.ndarray
            A ``2x2`` array or list of lists coefficients representing scale,
            rotation, and/or skew transformations.

        shift: list, numpy.ndarray
            A list of two coordinate shifts to be applied to coordinates
            *after* ``matrix`` transformations are applied.

        ref_tpwcs: TPWCS, None, optional
            A reference WCS of the type ``TPWCS`` that provides the tangent
            plane in which corrections (``matrix`` and ``shift``) were defined.
            When not provided (i.e., set to `None`), it is assumed that the
            transformations are being applied directly to *this* image WCS'
            tangent plane.

        meta: dict, None, optional
            Dictionary that will be merged to the object's ``meta`` fields.

        **kwargs: optional parameters
            Optional parameters for the WCS corrector. `FITSWCS` ignores these
            arguments (except for storing them in the ``meta`` attribute).

        """
        wcs = self._wcs
        orig_wcs = wcs.deepcopy()
        if ref_tpwcs is None:
            ref_tpwcs = FITSWCS(wcs.deepcopy())

        naxis1, naxis2 = wcs.pixel_shape

        shift = -np.dot(inv(matrix), shift)

        # estimate step for numerical differentiation. We need a step
        # large enough to avoid rounding errors and small enough to get a
        # better precision for numerical differentiation.
        # TODO: The logic below should be revised at a later time so that it
        # better takes into account the two competing requirements.
        hx = max(1.0, min(10, (wcs.wcs.crpix[0] - 1.0) / 100.0,
                          (naxis1 - wcs.wcs.crpix[0]) / 100.0))
        hy = max(1.0, min(10, (wcs.wcs.crpix[1] - 1.0) / 100.0,
                          (naxis2 - wcs.wcs.crpix[1]) / 100.0))

        # compute new CRVAL for the image WCS:
        crpix2d = np.atleast_2d(wcs.wcs.crpix)
        crval = wcs.wcs_pix2world(crpix2d, 1).ravel()
        crpixinref = ref_tpwcs.world_to_tanp(*crval)

        crpixinref = np.dot(
            np.subtract(crpixinref, shift),
            np.transpose(matrix)
        ).astype(np.double)
        wcs.wcs.crval = ref_tpwcs.tanp_to_world(crpixinref[0], crpixinref[1])
        wcs.wcs.set()

        # initial approximation for CD matrix of the image WCS:
        (U, u) = self._linearize(orig_wcs, ref_tpwcs, wcs.wcs.crpix,
                                 matrix, shift, hx=hx, hy=hy)
        if hasattr(wcs.wcs, 'pc'):
            wcs.wcs.pc = np.dot(wcs.wcs.pc, U).astype(np.double)
        else:
            wcs.wcs.cd = np.dot(wcs.wcs.cd, U).astype(np.double)
        wcs.wcs.set()

        # save linear transformation info to the meta attribute:
        super().set_correction(matrix=matrix, shift=shift, meta=meta, **kwargs)

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
        x, y = self._wcs.all_world2pix(ra, dec, 0, tolerance=1e-6, maxiter=50)
        return x, y

    def det_to_tanp(self, x, y):
        """
        Convert detector (pixel) coordinates to tangent plane coordinates.

        """
        x, y = self._wcs.pix2foc(x, y, 0)
        return x, y

    def tanp_to_det(self, x, y):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        ndim = np.ndim(x)
        ra, dec = self._wcs.wcs_pix2world(x, y, 0)
        if not ndim:
            # deal with a bug/feature in stwcs' all_world2pix v.1.5.3
            ra = np.atleast_1d(ra)
            dec = np.atleast_1d(dec)
        x, y = self._wcs.all_world2pix(ra, dec, 0, tolerance=1e-6, maxiter=50)
        if not ndim:
            x = float(x)
            y = float(y)
        return x, y

    def world_to_tanp(self, ra, dec):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        x, y = self._wcs.wcs_world2pix(ra, dec, 0)
        return x, y

    def tanp_to_world(self, x, y):
        """ Convert tangent plane coordinates to world coordinates. """
        ra, dec = self._wcs.wcs_pix2world(x, y, 0)
        return ra, dec

    @property
    def bounding_box(self):
        """
        Get the bounding box (if any) of the underlying image for which
        the original WCS is defined.

        """
        if self._owcs.pixel_bounds is None:
            if self._owcs.pixel_shape is not None:
                nx, ny = self._owcs.pixel_shape
            elif self._owcs.array_shape is not None:
                ny, nx = self._owcs.array_shape
            else:
                return None

            return ((-0.5, nx - 0.5), (-0.5, ny - 0.5))

        else:
            return self._owcs.pixel_bounds

    def _linearize(self, wcsima, ref_tpwcs, imcrpix, f, shift, hx=1.0, hy=1.0):
        """ linearization using 5-point formula for first order derivative. """
        x0 = imcrpix[0] - 1
        y0 = imcrpix[1] - 1

        px = np.asarray([x0, x0 - hx, x0 - hx * 0.5, x0 + hx * 0.5,
                         x0 + hx, x0, x0, x0, x0], dtype=np.double)
        py = np.asarray([y0, y0, y0, y0, y0, y0 - hy, y0 - hy * 0.5,
                         y0 + hy * 0.5, y0 + hy], dtype=np.double)

        # convert image coordinates to reference image coordinates:
        ra, dec = wcsima.wcs_pix2world(px, py, 0)
        px, py = np.array(ref_tpwcs.world_to_tanp(ra, dec))

        # apply linear fit transformation:
        px, py = np.dot(f, [px - shift[0], py - shift[1]]).astype(np.double)
        # convert back to image coordinate system:
        p = np.array(self.wcs.wcs_world2pix(*ref_tpwcs.tanp_to_world(px, py), 0),
                     dtype=np.longdouble).T

        # derivative with regard to x:
        u1 = ((p[1] - p[4]) + 8 * (p[3] - p[2])) / (6 * hx)
        # derivative with regard to y:
        u2 = ((p[5] - p[8]) + 8 * (p[7] - p[6])) / (6 * hy)

        return (np.asarray([u1, u2]).T, p[0])


def _get_submodel(model, name):
    """ Return the first occurence of a sub-model. Search is performed by
        model name.
    """
    if not isinstance(model, CompoundModel):
        return model if model.name == name else None

    for m in model.traverse_postorder():
        if m.name == name:
            return m

    return None


class JWSTgWCS(TPWCS):
    """ A class for holding ``JWST`` ``gWCS`` information and for managing
    tangent-plane corrections.

    """
    def __init__(self, wcs, wcsinfo, meta=None):
        """
        Parameters
        ----------
        wcs: GWCS
            A `GWCS` object.

        wcsinfo: dict
            A dictionary containing reference angles of ``JWST`` instrument
            provided in the ``wcsinfo`` property of ``JWST`` meta data.

            This dictionary **must contain** the following keys and values:


            'v2_ref': float
                V2 position of the reference point in arc seconds.

            'v3_ref': float
                V3 position of the reference point in arc seconds.

            'roll_ref': float
                Roll angle in degrees.

        meta: dict, None, optional
            Dictionary that will be merged to the object's ``meta`` fields.

        """
        if not _ASTROPY_VER_GE_4:
            raise NotImplementedError(
                "JWST support requires astropy version >= 4.0"
            )

        if not _GWCS_VER_GT_0P12:
            raise NotImplementedError(
                "JWST support requires gwcs version > 0.12.0 "
                "To pip install minimal required version, do the following:\n"
                "pip install git+https://github.com/spacetelescope/gwcs@"
                "1c1cb3bb35caddef80fb760ea68bc71e189d32de"
            )

        valid, message = self._check_wcs_structure(wcs)
        if not valid:
            raise ValueError("Unsupported WCS structure: {}".format(message))

        super().__init__(wcs=wcs, meta=meta)

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
            if not JWSTgWCS._check_tpcorr_structure(self._tpcorr):
                raise ValueError("Unsupported tangent-plance correction type.")

            # check that transformation parameters are consistent with
            # reference angles:
            v2ref, v3ref, roll = self._tpcorr['det_to_optic_axis'].angles.value

            eps_v2 = 10.0 * np.finfo(v2_ref).eps
            eps_v3 = 10.0 * np.finfo(v3_ref).eps
            eps_roll = 10.0 * np.finfo(roll_ref).eps
            if not (np.isclose(v2_ref, v2ref * 3600.0, rtol=eps_v2) and
                    np.isclose(v3_ref, -v3ref * 3600.0, rtol=eps_v3) and
                    np.isclose(roll_ref, roll, rtol=eps_roll)):
                raise ValueError(
                    "WCS/TPCorr parameters 'v2ref', 'v3ref', and/or 'roll' "
                    "differ from the corresponding reference values."
                )
            self._partial_tpcorr = JWSTgWCS._v2v3_to_tpcorr_from_full(
                self._tpcorr
            )

        else:
            self._v23name = 'v2v3'
            self._tpcorr = None
            self._default_tpcorr = JWSTgWCS._tpcorr_init(
                v2_ref=v2_ref / 3600.0,
                v3_ref=v3_ref / 3600.0,
                roll_ref=roll_ref
            )
            self._partial_tpcorr = JWSTgWCS._v2v3_to_tpcorr_from_full(
                self._default_tpcorr
            )

        self._update_transformations()

    @staticmethod
    def _check_tpcorr_structure(tpcorr):
        # implement a more sophisticated check later
        if tpcorr.name != 'JWST tangent-plane linear correction. v1':
            return False
        return True

    def _update_transformations(self):
        # define transformations from detector/world coordinates to
        # the tangent plane:
        detname = self._wcs.pipeline[0][0].name
        worldname = self._wcs.pipeline[-1][0].name

        # Generally needed transformations:
        self._world_to_v23 = self._wcs.get_transform(worldname, self._v23name)
        self._v23_to_world = self._wcs.get_transform(self._v23name, worldname)
        self._det_to_v23 = self._wcs.get_transform(detname, self._v23name)
        self._det_to_world = self._wcs.__call__

        # Optional / convenience transformations:
        try:
            self._v23_to_det = self._wcs.get_transform(self._v23name, detname)
        except NotImplementedError:
            self._v23_to_det = None

        try:
            self._world_to_det = self._wcs.invert
        except NotImplementedError:
            self._world_to_det = None

    @staticmethod
    def _tpcorr_combine_affines(tpcorr, matrix, shift):
        AffineTransformation2D(matrix, shift)  # check input parameters are OK
        m = np.dot(matrix, tpcorr['tp_affine'].matrix.value)
        t = np.dot(matrix, tpcorr['tp_affine'].translation.value) + shift
        tpcorr['tp_affine'].matrix = m
        tpcorr['tp_affine'].translation = t

        # update the affine transformation of the inverse model as well:
        invm = np.linalg.inv(m)
        tpcorr.inverse['tp_affine_inv'].matrix = invm
        tpcorr.inverse['tp_affine_inv'].translation = -np.dot(invm, t)

    @staticmethod
    def _tpcorr_init(v2_ref, v3_ref, roll_ref):
        s2c = SphericalToCartesian(name='s2c', wrap_lon_at=180)
        c2s = CartesianToSpherical(name='c2s', wrap_lon_at=180)

        unit_conv = Scale(1.0 / 3600.0, name='arcsec_to_deg_1D')
        unit_conv = unit_conv & unit_conv
        unit_conv.name = 'arcsec_to_deg_2D'

        unit_conv_inv = Scale(3600.0, name='deg_to_arcsec_1D')
        unit_conv_inv = unit_conv_inv & unit_conv_inv
        unit_conv_inv.name = 'deg_to_arcsec_2D'

        affine = AffineTransformation2D(name='tp_affine')
        affine_inv = AffineTransformation2D(name='tp_affine_inv')

        rot = RotationSequence3D(
            [v2_ref, -v3_ref, roll_ref],
            'zyx',
            name='det_to_optic_axis'
        )
        rot_inv = rot.inverse
        rot_inv.name = 'optic_axis_to_det'

        # projection submodels:
        c2tan = ((Mapping((0, 1, 2), name='xyz') /
                  Mapping((0, 0, 0), n_inputs=3, name='xxx')) |
                 Mapping((1, 2), name='xtyt'))
        c2tan.name = 'Cartesian 3D to TAN'
        tan2c = (Mapping((0, 0, 1), n_inputs=2, name='xtyt2xyz') |
                 (Const1D(1, name='one') & Identity(2, name='I(2D)')))
        tan2c.name = 'TAN to cartesian 3D'

        total_corr = (
            unit_conv | s2c | rot | c2tan | affine |
            tan2c | rot_inv | c2s | unit_conv_inv
        )
        total_corr.name = 'JWST tangent-plane linear correction. v1'

        inv_total_corr = (
            unit_conv | s2c | rot | c2tan | affine_inv |
            tan2c | rot_inv | c2s | unit_conv_inv
        )
        inv_total_corr.name = 'Inverse JWST tangent-plane linear correction. v1'

        # TODO
        # re-enable circular inverse definitions once
        # https://github.com/spacetelescope/asdf/issues/744 or
        # https://github.com/spacetelescope/asdf/issues/745 are resolved.
        #
        # inv_total_corr.inverse = total_corr
        total_corr.inverse = inv_total_corr

        return total_corr

    @staticmethod
    def _v2v3_to_tpcorr_from_full(tpcorr):
        s2c = tpcorr['s2c']
        c2s = tpcorr['c2s']

        # unit_conv = _get_submodel(tpcorr, 'arcsec_to_deg_2D')
        # unit_conv_inv = _get_submodel(tpcorr, 'deg_to_arcsec_2D')
        #
        # The code below is a work-around to the code above not working.
        # TODO: understand why _get_submodel is unable to retrieve
        #       some submodels in a compound model.
        #
        unit_conv = _get_submodel(tpcorr, 'arcsec_to_deg_1D')
        unit_conv = unit_conv & unit_conv
        unit_conv.name = 'arcsec_to_deg_2D'
        unit_conv_inv = _get_submodel(tpcorr, 'deg_to_arcsec_1D')
        unit_conv_inv = unit_conv_inv & unit_conv_inv
        unit_conv_inv.name = 'deg_to_arcsec_2D'

        affine = tpcorr['tp_affine']
        affine_inv = affine.inverse
        affine_inv.name = 'tp_affine_inv'

        rot = tpcorr['det_to_optic_axis']
        rot_inv = rot.inverse
        rot_inv.name = 'optic_axis_to_det'

        # c2tan = _get_submodel(tpcorr, 'Cartesian 3D to TAN')
        # tan2c = _get_submodel(tpcorr, 'TAN to cartesian 3D')
        #
        # The code below is a work-around to the code above not working.
        # TODO: understand why _get_submodel is unable to retrieve
        #       some submodels in a compound model.
        #
        c2tan = ((Mapping((0, 1, 2), name='xyz') /
                  Mapping((0, 0, 0), n_inputs=3, name='xxx')) |
                 Mapping((1, 2), name='xtyt'))
        c2tan.name = 'Cartesian 3D to TAN'
        tan2c = (Mapping((0, 0, 1), n_inputs=2, name='xtyt2xyz') |
                 (Const1D(1, name='one') & Identity(2, name='I(2D)')))
        tan2c.name = 'TAN to cartesian 3D'

        v2v3_to_tpcorr = unit_conv | s2c | rot | c2tan | affine
        v2v3_to_tpcorr.name = 'jwst_v2v3_to_tpcorr'

        tpcorr_to_v2v3 = affine_inv | tan2c | rot_inv | c2s | unit_conv_inv
        tpcorr_to_v2v3.name = 'jwst_tpcorr_to_v2v3'

        v2v3_to_tpcorr.inverse = tpcorr_to_v2v3
        tpcorr_to_v2v3.inverse = v2v3_to_tpcorr

        return v2v3_to_tpcorr

    @property
    def ref_angles(self):
        """ Return a ``wcsinfo``-like dictionary of main WCS parameters. """
        return {k: v for k, v in self._wcsinfo.items()}

    def set_correction(self, matrix=[[1, 0], [0, 1]], shift=[0, 0],
                       ref_tpwcs=None, meta=None, **kwargs):
        """
        Sets a tangent-plane correction of the GWCS object according to
        the provided liniar parameters. In addition, this function updates
        the ``meta`` attribute of the `JWSTgWCS` object with the values
        of keyword arguments except for the argument ``meta`` which is
        *merged* with the *attribute* ``meta``.

        Parameters
        ----------
        matrix: list, numpy.ndarray
            A ``2x2`` array or list of lists coefficients representing scale,
            rotation, and/or skew transformations.

        shift: list, numpy.ndarray
            A list of two coordinate shifts to be applied to coordinates
            *after* ``matrix`` transformations are applied.

        ref_tpwcs: TPWCS, None, optional
            A reference WCS of the type ``TPWCS`` that provides the tangent
            plane in which corrections (``matrix`` and ``shift``) were defined.
            When not provided (i.e., set to `None`), it is assumed that the
            transformations are being applied directly to *this* image WCS'
            tangent plane.

        meta: dict, None, optional
            Dictionary that will be merged to the object's ``meta`` fields.

        **kwargs: optional parameters
            Optional parameters for the WCS corrector. `JWSTgWCS` ignores these
            arguments (except for storing them in the ``meta`` attribute).

        """
        frms = self._wcs.available_frames

        if ref_tpwcs is None:
            matrix = np.array(matrix, dtype=np.double)
            shift = np.array(shift, dtype=np.double)
        else:
            # compute linear transformation from the tangent plane used for
            # alignment to the tangent plane of this wcs:
            r, t = _tp2tp(ref_tpwcs, self)
            matrix = np.linalg.multi_dot([r, matrix, inv(r)]).astype(np.double)
            shift = (np.dot(r, shift) - np.dot(matrix, t) + t).astype(np.double)

        # if original WCS did not have tangent-plane corrections, create
        # new correction and add it to the WCs pipeline:
        if self._tpcorr is None:
            self._tpcorr = JWSTgWCS._tpcorr_init(
                v2_ref=self._wcsinfo['v2_ref'] / 3600.0,
                v3_ref=self._wcsinfo['v3_ref'] / 3600.0,
                roll_ref=self._wcsinfo['roll_ref']
            )

            JWSTgWCS._tpcorr_combine_affines(
                self._tpcorr,
                matrix,
                _ARCSEC2RAD * np.asarray(shift)
            )

            self._partial_tpcorr = JWSTgWCS._v2v3_to_tpcorr_from_full(self._tpcorr)

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
            JWSTgWCS._tpcorr_combine_affines(
                self._tpcorr,
                matrix,
                _ARCSEC2RAD * np.asarray(shift)
            )

            self._partial_tpcorr = JWSTgWCS._v2v3_to_tpcorr_from_full(self._tpcorr)

            idx_v2v3 = frms.index(self._v23name)
            pipeline = deepcopy(self._wcs.pipeline)
            pipeline[idx_v2v3 - 1] = (pipeline[idx_v2v3 - 1][0],
                                      deepcopy(self._tpcorr))
            self._wcs = gwcs.WCS(pipeline, name=self._owcs.name)

        # reset definitions of the transformations from detector/world
        # coordinates to the tangent plane:
        self._update_transformations()

        # save linear transformation info to the meta attribute:
        super().set_correction(matrix=matrix, shift=shift, meta=meta, **kwargs)

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
        if self._world_to_det is None:
            raise NotImplementedError("World to detector transformation has "
                                      "not been implemented.")
        x, y = self._world_to_det(ra, dec)
        return x, y

    def det_to_tanp(self, x, y):
        """
        Convert detector (pixel) coordinates to tangent plane coordinates.

        """
        v2, v3 = self._det_to_v23(x, y)
        x, y = self._partial_tpcorr(v2, v3)
        return _RAD2ARCSEC * x, _RAD2ARCSEC * y

    def tanp_to_det(self, x, y):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        if self._partial_tpcorr.inverse is None or self._v23_to_det is None:
            raise NotImplementedError(
                "Tangent plane to detector transformation has not been "
                "implemented."
            )

        v2, v3 = self._partial_tpcorr.inverse(
            _ARCSEC2RAD * np.asanyarray(x),
            _ARCSEC2RAD * np.asanyarray(y)
        )
        x, y = self._v23_to_det(v2, v3)
        return x, y

    def world_to_tanp(self, ra, dec):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        v2, v3 = self._world_to_v23(ra, dec)
        x, y = self._partial_tpcorr(v2, v3)
        return _RAD2ARCSEC * x, _RAD2ARCSEC * y

    def tanp_to_world(self, x, y):
        """ Convert tangent plane coordinates to world coordinates. """
        v2, v3 = self._partial_tpcorr.inverse(
            _ARCSEC2RAD * np.asanyarray(x),
            _ARCSEC2RAD * np.asanyarray(y)
        )
        ra, dec = self._v23_to_world(v2, v3)
        return ra, dec

    @property
    def bounding_box(self):
        """
        Get the bounding box (if any) of the underlying image for which
        the original WCS is defined.

        """
        if self._owcs.pixel_bounds is None:
            if self._owcs.pixel_shape is not None:
                nx, ny = self._owcs.pixel_shape
            elif self._owcs.array_shape is not None:
                ny, nx = self._owcs.array_shape
            else:
                return None

            return ((-0.5, nx - 0.5), (-0.5, ny - 0.5))

        else:
            return self._owcs.pixel_bounds
