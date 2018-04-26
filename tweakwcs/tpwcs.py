"""
This module provides support for manipulating tangent-plane corrections
of ``WCS``.

:Authors: Mihai Cara (contact: help@stsci.edu)

:License: :doc:`../LICENSE`

"""
from __future__ import (absolute_import, division, unicode_literals,
                        print_function)

# STDLIB
import logging
import sys
from copy import deepcopy
from abc import ABC, abstractmethod

# THIRD-PARTY
import numpy as np
import gwcs
from jwst.transforms.tpcorr import TPCorr

# LOCAL
#from .tpcorr import TPCorr, rot_mat3D

from . import __version__, __version_date__

__author__ = 'Mihai Cara'

__all__ = ['TPWCS', 'JWSTgWCS']


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


_INT_TYPE = (int, long,) if sys.version_info < (3,) else (int,)


def _is_int(n):
    return (
        (isinstance(n, _INT_TYPE) and not isinstance(n, bool)) or
        (isinstance(n, np.generic) and np.issubdtype(n, np.integer))
    )


class TPWCS(ABC):
    """ A class that provides common interface for manipulating WCS information
    and for managing tangent-plane corrections.

    """
    def __init__(self, wcs):
        """
        Parameters
        ----------

        wcs : GWCS
            A `GWCS` object.

        """
        self._owcs = wcs
        self._wcs = deepcopy(wcs)

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
    def set_correction(self, matrix=[[1, 0], [0, 1]], shift=[0, 0]):
        """
        Sets a tangent-plane correction of the GWCS object according to
        the provided liniar parameters.

        Parameters
        ----------
        matrix : list, numpy.ndarray
            A ``2x2`` array or list of lists coefficients representing scale,
            rotation, and/or skew transformations.

        shift : list, numpy.ndarray
            A list of two coordinate shifts to be applied to coordinates
            *before* ``matrix`` transformations are applied.

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
        if not self._check_wcs_structure(wcs):
            raise ValueError("Unsupported WCS structure.")

        v2_ref = wcsinfo['v2_ref']
        v3_ref = wcsinfo['v3_ref']
        roll_ref = wcsinfo['roll_ref']

        self._wcsinfo = {'v2_ref': v2_ref, 'v3_ref': v3_ref,
                         'roll_ref': roll_ref}

        # perform additional check that if tangent plane correction is already
        # present in the WCS pipeline, it is of TPCorr class and that
        # its parameters are consistent with reference angles:
        frms = [f[0] for f in wcs.pipeline]
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
        detname = self._wcs.pipeline[0][0]
        worldname = self._wcs.pipeline[-1][0]

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

    def set_correction(self, matrix=[[1, 0], [0, 1]], shift=[0, 0]):
        """
        Sets a tangent-plane correction of the GWCS object according to
        the provided liniar parameters.

        Parameters
        ----------
        matrix : list, numpy.ndarray
            A ``2x2`` array or list of lists coefficients representing scale,
            rotation, and/or skew transformations.

        shift : list, numpy.ndarray
            A list of two coordinate shifts to be applied to coordinates
            *before* ``matrix`` transformations are applied.

        """
        frms = [f[0] for f in self._wcs.pipeline]

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
            pipeline.insert(idx_v2v3 + 1, ('v2v3corr', pt))
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

    def _check_wcs_structure(self, wcs):
        if wcs is None or wcs.pipeline is None:
            return False

        frms = [f[0] for f in wcs.pipeline]
        nframes = len(frms)
        if nframes < 3:
            return False

        if frms.count(frms[0]) > 1 or frms.count(frms[-1]) > 1:
            return False

        if frms.count('v2v3') != 1:
            return False

        idx_v2v3 = frms.index('v2v3')
        if idx_v2v3 == 0 or idx_v2v3 == (nframes - 1):
            return False

        nv2v3corr = frms.count('v2v3corr')
        if nv2v3corr == 0:
            return True
        elif nv2v3corr > 1:
            return False

        idx_v2v3corr = frms.index('v2v3corr')
        if idx_v2v3corr != (idx_v2v3 + 1) or idx_v2v3corr == (nframes - 1):
            return False

        return True

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
