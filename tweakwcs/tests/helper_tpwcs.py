"""
A module containing unit tests for the `wcsutil` module.

Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
import math
import numpy as np
from astropy.modeling import Model, Parameter, InputParameterError
from astropy import coordinates as coord
from astropy import units as u
import gwcs

from tweakwcs.tpwcs import TPWCS


class DummyTPWCS(TPWCS):
    def set_correction(self, matrix=[[1, 0], [0, 1]], shift=[0, 0], meta=None,
                       **kwargs):
        super().set_correction(matrix=matrix, shift=shift, meta=meta, **kwargs)

    def det_to_world(self, x, y):
        return super().det_to_world(x, y)

    def world_to_det(self, ra, dec):
        return super().world_to_det(ra, dec)

    def det_to_tanp(self, x, y):
        return super().det_to_tanp(x, y)

    def tanp_to_det(self, x, y):
        return super().tanp_to_det(x, y)

    def world_to_tanp(self, ra, dec):
        return super().world_to_tanp(ra, dec)

    def tanp_to_world(self, x, y):
        return super().tanp_to_world(x, y)


def rot_mat3d(angle, axis):
    cs = math.cos(angle)
    sn = math.sin(angle)
    axisv = np.array(axis * [0.0] + [1.0] + (2 - axis) * [0.0],
                     dtype=np.float)
    mat2d = np.array([[cs, sn], [-sn, cs]], dtype=np.float)
    return np.insert(np.insert(mat2d, axis, [0.0, 0.0], 1), axis, axisv, 0)


class DetToV2V3(Model):
    """
    Apply ``V2ref``, ``V3ref``, and ``roll`` to input angles and project
    the point from the tangent plane onto a celestial sphere.

    Parameters
    ----------
    v2ref : float
        V2 position of the reference point in degrees. Default is 0 degrees.

    v3ref : float
        V3 position of the reference point in degrees. Default is 0 degrees.

    roll : float
        Roll angle in degrees. Default is 0 degrees.

    """
    v2ref = Parameter(default=0.0)
    v3ref = Parameter(default=0.0)
    roll = Parameter(default=0.0)
    cd = Parameter(default=[[1.0, 0.0], [0.0, 1.0]])
    crpix = Parameter(default=[0.0, 0.0])

    inputs = ('x', 'y')
    outputs = ('v2', 'v3')

    _separable = False
    standard_broadcasting = False

    r0 = 1  # 3600.0 * np.rad2deg(1.0)
    """
    Radius of the generating sphere. This sets the circumference to 360 deg
    so that arc length is measured in deg.
    """

    def __init__(self, v2ref=v2ref.default, v3ref=v3ref.default,
                 roll=roll.default, cd=cd.default,
                 crpix=crpix.default, **kwargs):
        super().__init__(
            v2ref=v2ref, v3ref=v3ref, roll=roll, cd=cd,
            crpix=crpix, **kwargs
        )

    @property
    def input_units(self):
        return {'x': None, 'y': None}

    @property
    def return_units(self):
        return {'v2': None, 'v3': None}

    @cd.validator
    def cd(self, value):
        """ Validates that the input CD matrix is a 2x2 2D array. """
        if np.shape(value) != (2, 2):
            raise InputParameterError(
                "Expected CD matrix to be a 2x2 array")

    @crpix.validator
    def crpix(self, value):
        """
        Validates that the crpix vector is a 2D vector.  This allows
        either a "row" vector.

        """
        if not (np.ndim(value) == 1 and np.shape(value) == (2,)):
            raise InputParameterError(
                "Expected 'crpix' to be a 2 element row vector."
            )

    @staticmethod
    def cartesian2spherical(x, y, z):
        """
        Convert cartesian coordinates to spherical coordinates (in acrsec).

        """
        h = np.hypot(x, y)
        alpha = np.rad2deg(np.arctan2(y, x))
        delta = np.rad2deg(np.arctan2(z, h))
        return alpha, delta

    def evaluate(self, x, y, v2ref, v3ref, roll, cd, crpix):
        """
        Evaluate the model on some input variables.

        """
        (x, y), format_info = self.prepare_inputs(x, y)

        # build Euler rotation matrices:
        rotm = [rot_mat3d(np.deg2rad(alpha), axis)
                for axis, alpha in enumerate([roll[0], v3ref[0], v2ref[0]])]
        euler_rot = np.linalg.multi_dot(rotm)
        inv_euler_rot = np.linalg.inv(euler_rot)

        # apply corrections:
        # NOTE: order of transforms may need to be swapped depending on
        #       how shifts are defined.
        x -= crpix[0][0]
        y -= crpix[0][1]
        x, y = np.dot(cd[0].T, (x, y))

        xt = self.__class__.r0 * x  # / zr
        yt = self.__class__.r0 * y  # / zr
        zt = np.full_like(x, self.__class__.r0)

        # "unrotate" cartezian coordinates back to their original
        # v2ref, v3ref, and roll "positions":
        zcr, xcr, ycr = np.dot(inv_euler_rot, (zt.ravel(), xt.ravel(), yt.ravel()))

        # convert cartesian to spherical coordinates:
        v2, v3 = self.cartesian2spherical(zcr, xcr, ycr)

        return self.prepare_outputs(format_info, v2.reshape(x.shape), v3.reshape(y.shape))

    @property
    def inverse(self):
        """
        Returns a new `TPCorr` instance which performs the inverse
        transformation of the transformation defined for this `TPCorr` model.

        """
        return V2V3ToDet(v2ref=self.v2ref.value, v3ref=self.v3ref.value,
                         roll=self.roll.value, cd=self.cd.value,
                         crpix=self.crpix.value)


class V2V3ToDet(Model):
    """
    Apply ``V2ref``, ``V3ref``, and ``roll`` to input angles and project
    the point from the tangent plane onto a celestial sphere.

    Parameters
    ----------
    v2ref : float
        V2 position of the reference point in degrees. Default is 0 degrees.

    v3ref : float
        V3 position of the reference point in degrees. Default is 0 degrees.

    roll : float
        Roll angle in degrees. Default is 0 degrees.

    """
    v2ref = Parameter(default=0.0)
    v3ref = Parameter(default=0.0)
    roll = Parameter(default=0.0)
    cd = Parameter(default=[[1.0, 0.0], [0.0, 1.0]])
    crpix = Parameter(default=[0.0, 0.0])

    inputs = ('v2', 'v3')
    outputs = ('x', 'y')

    # input_units_strict = False
    # input_units_allow_dimensionless = True
    _separable = False
    standard_broadcasting = False

    r0 = 1  # 3600.0 * np.rad2deg(1.0)
    """
    Radius of the generating sphere. This sets the circumference to 360 deg
    so that arc length is measured in deg.
    """
    def __init__(self, v2ref=v2ref.default, v3ref=v3ref.default,
                 roll=roll.default, cd=cd.default,
                 crpix=crpix.default, **kwargs):
        super().__init__(
            v2ref=v2ref, v3ref=v3ref, roll=roll, cd=cd,
            crpix=crpix, **kwargs
        )

    @property
    def input_units(self):
        return {'v2': None, 'v3': None}

    @property
    def return_units(self):
        return {'x': None, 'y': None}

    @cd.validator
    def cd(self, value):
        """
        Validates that the input CD matrix is a 2x2 2D array.

        """
        if np.shape(value) != (2, 2):
            raise InputParameterError(
                "Expected CD matrix to be a 2x2 array")

    @crpix.validator
    def crpix(self, value):
        """
        Validates that the crpix vector is a 2D vector.  This allows
        either a "row" vector.

        """
        if not (np.ndim(value) == 1 and np.shape(value) == (2,)):
            raise InputParameterError(
                "Expected 'crpix' to be a 2 element row vector."
            )

    @staticmethod
    def spherical2cartesian(alpha, delta):
        """
        Convert spherical coordinates (in arcsec) to cartesian.

        """
        alpha = np.deg2rad(alpha)
        delta = np.deg2rad(delta)
        x = np.cos(alpha) * np.cos(delta)
        y = np.cos(delta) * np.sin(alpha)
        z = np.sin(delta)
        return x, y, z

    def evaluate(self, v2, v3, v2ref, v3ref, roll, cd, shift):
        """
        Evaluate the model on some input variables.

        """

        # convert spherical coordinates to cartesian assuming unit sphere:
        xyz = self.spherical2cartesian(v2.ravel(), v3.ravel())

        # build Euler rotation matrices:
        rotm = [rot_mat3d(np.deg2rad(alpha), axis)
                for axis, alpha in enumerate([roll[0], v3ref[0], v2ref[0]])]
        euler_rot = np.linalg.multi_dot(rotm)

        # rotate cartezian coordinates:
        zr, xr, yr = np.dot(euler_rot, xyz)

        # project points onto the tanject plane
        # (tangent to a sphere of radius r0):
        xt = self.__class__.r0 * xr / zr
        yt = self.__class__.r0 * yr / zr

        # apply corrections:
        # NOTE: order of transforms may need to be swapped depending on
        #       how shifts are defined.
        x, y = np.dot(np.linalg.inv(cd[0]).T, (xt, yt))
        x += shift[0][0]
        y += shift[0][1]

        return x, y

    @property
    def inverse(self):
        """
        Returns a new `TPCorr` instance which performs the inverse
        transformation of the transformation defined for this `TPCorr` model.

        """
        return DetToV2V3(v2ref=self.v2ref.value, v3ref=self.v3ref.value,
                         roll=self.roll.value, cd=self.cd.value,
                         crpix=self.crpix.value)


class V2V3ToSky(Model):
    """
    Rotates V2-V3 sphere on the sky.
    """
    inputs = ("v2", "v3")
    outputs = ("ra", "dec")

    angles = Parameter()

    _separable = False
    standard_broadcasting = False

    def __init__(self, angles, axes_order, name=None):
        self.axes_order = axes_order
        super().__init__(angles=angles, name=name)

    @staticmethod
    def build_euler_matrix(axis_angle):
        # build Euler rotation matrices:
        rotm = [rot_mat3d(np.deg2rad(alpha), axis)
                for axis, alpha in axis_angle]
        euler_rot = np.linalg.multi_dot(rotm)
        return euler_rot

    @staticmethod
    def cartesian2spherical(x, y, z):
        """ Convert cartesian coordinates to spherical (in acrsec). """
        h = np.hypot(x, y)
        alpha = np.rad2deg(np.arctan2(y, x))
        delta = np.rad2deg(np.arctan2(z, h))
        return alpha, delta

    @staticmethod
    def spherical2cartesian(alpha, delta):
        """ Convert spherical coordinates (in arcsec) to cartesian. """
        alpha = np.deg2rad(alpha)
        delta = np.deg2rad(delta)
        x = np.cos(alpha) * np.cos(delta)
        y = np.cos(delta) * np.sin(alpha)
        z = np.sin(delta)
        return x, y, z

    def evaluate(self, v2, v3, angles):
        """ Evaluate the model on some input variables. """

        # convert spherical coordinates to cartesian assuming unit sphere:
        xyz = self.spherical2cartesian(v2.ravel(), v3.ravel())

        # build Euler rotation matrices:
        euler_rot = self.__class__.build_euler_matrix(
            [(v[0], -v[1]) if v[0] != 1 else v for v in
             zip(self.axes_order, angles[0])][::-1]
        )

        # rotate cartezian coordinates:
        z, x, y = np.dot(euler_rot, xyz)

        # convert cartesian to spherical coordinates:
        ra, dec = self.cartesian2spherical(z, x, y)

        return ra, dec

    @property
    def inverse(self):
        return self.__class__(
            [-a for a in self.angles.value[::-1]], self.axes_order[::-1]
        )


def make_mock_jwst_pipeline(v2ref=0, v3ref=0, roll=0, crpix=[512, 512],
                            cd=[[1e-5, 0], [0, 1e-5]], crval=[0, 0]):
    detector = gwcs.coordinate_frames.Frame2D(
        name='detector', axes_order=(0, 1), unit=(u.pix, u.pix)
    )
    v2v3 = gwcs.coordinate_frames.Frame2D(
        name='v2v3', axes_order=(0, 1), unit=(u.arcsec, u.arcsec)
    )
    world = gwcs.coordinate_frames.CelestialFrame(reference_frame=coord.ICRS(),
                                                  name='world')
    det2v2v3 = DetToV2V3(v2ref=v2ref / 3600.0, v3ref=v3ref / 3600.0,
                         roll=roll, cd=cd, crpix=crpix)
    v23sky = V2V3ToSky([-v2ref / 3600.0, v3ref / 3600.0, -roll,
                        -crval[1], crval[0]], [2, 1, 0, 1, 2])
    pipeline = [(detector, det2v2v3), (v2v3, v23sky), (world, None)]
    return pipeline


def make_mock_jwst_wcs(v2ref=0, v3ref=0, roll=0, crpix=[512, 512],
                       cd=[[1e-5, 0], [0, 1e-5]], crval=[0, 0]):
    pipeline = make_mock_jwst_pipeline(v2ref, v3ref, roll, crpix, cd, crval)
    wcs = gwcs.wcs.WCS(pipeline)
    wcs.bounding_box = ((-0.5, 1024 - 0.5), (-0.5, 2048 - 0.5))
    wcs.array_shape = (2048, 1024)
    return wcs


# `IncompatibleCorrections` and `TPCorr` have been copied from
# `jwst.transforms.tpcorr` in order to provide ability to test `tpwcs` module
# even when `jwst` package is not installed.
class IncompatibleCorrections(Exception):
    """
    An exception class used to report cases when two or more tangent plane
    corrections cannot be combined into a single one.
    """
    pass


class TPCorr(Model):
    """
    Apply ``V2ref``, ``V3ref``, and ``roll`` to input angles and project
    the point from the tangent plane onto a celestial sphere.

    Parameters
    ----------
    v2ref : float
        V2 position of the reference point in degrees. Default is 0 degrees.

    v3ref : float
        V3 position of the reference point in degrees. Default is 0 degrees.

    roll : float
        Roll angle in degrees. Default is 0 degrees.

    """
    v2ref = Parameter(default=0.0)
    v3ref = Parameter(default=0.0)
    roll = Parameter(default=0.0)
    matrix = Parameter(default=[[1.0, 0.0], [0.0, 1.0]])
    shift = Parameter(default=[0.0, 0.0])

    inputs = ('v2', 'v3')
    outputs = ('v2c', 'v3c')

    # input_units_strict = False
    # input_units_allow_dimensionless = True
    _separable = False
    standard_broadcasting = False

    r0 = 3600.0 * np.rad2deg(1.0)
    """
    Radius of the generating sphere. This sets the circumference to 360 deg
    so that arc length is measured in deg.
    """

    def __init__(self, v2ref=v2ref.default, v3ref=v3ref.default,
                 roll=roll.default, matrix=matrix.default,
                 shift=shift.default, **kwargs):
        super(TPCorr, self).__init__(
            v2ref=v2ref, v3ref=v3ref, roll=roll, matrix=matrix,
            shift=shift, **kwargs
        )

    @property
    def input_units(self):
        return {'v2': None, 'v3': None}

    @property
    def return_units(self):
        return {'v2c': None, 'v3c': None}

    @matrix.validator
    def matrix(self, value):
        """ Validates that the input matrix is a 2x2 2D array. """
        if np.shape(value) != (2, 2):
            raise InputParameterError(  # pragma: no cover
                "Expected transformation matrix to be a 2x2 array")

    @shift.validator
    def shift(self, value):
        """
        Validates that the shift vector is a 2D vector.  This allows
        either a "row" vector.
        """
        if not (np.ndim(value) == 1 and np.shape(value) == (2,)):
            raise InputParameterError(  # pragma: no cover
                "Expected 'shift' to be a 2 element row vector."
            )

    @staticmethod
    def cartesian2spherical(x, y, z):
        """ Convert cartesian coordinates to spherical coordinates (in acrsec).
        """
        h = np.hypot(x, y)
        alpha = 3600.0 * np.rad2deg(np.arctan2(y, x))
        delta = 3600.0 * np.rad2deg(np.arctan2(z, h))
        return alpha, delta

    @staticmethod
    def spherical2cartesian(alpha, delta):
        """ Convert spherical coordinates (in arcsec) to cartesian. """
        alpha = np.deg2rad(alpha / 3600.0)
        delta = np.deg2rad(delta / 3600.0)
        x = np.cos(alpha) * np.cos(delta)
        y = np.cos(delta) * np.sin(alpha)
        z = np.sin(delta)
        return x, y, z

    def v2v3_to_tanp(self, v2, v3):
        """ Converts V2V3 spherical coordinates to tangent plane coordinates.
        """
        (v2, v3), format_info = self.prepare_inputs(v2, v3)

        # convert spherical coordinates to cartesian assuming unit sphere:
        xyz = self.spherical2cartesian(v2.ravel(), v3.ravel())

        # build Euler rotation matrices:
        rotm = [
            rot_mat3d(np.deg2rad(alpha), axis)
            for axis, alpha in enumerate(
                [self.roll.value, self.v3ref.value, self.v2ref.value]
            )
        ]
        euler_rot = np.linalg.multi_dot(rotm)

        # rotate cartezian coordinates:
        zr, xr, yr = np.dot(euler_rot, xyz)

        # project points onto the tanject plane
        # (tangent to a sphere of radius r0):
        xt = self.__class__.r0 * xr / zr
        yt = self.__class__.r0 * yr / zr

        # apply corrections:
        # NOTE: order of transforms may need to be swapped depending on
        #       how shifts are defined.
        xt -= self.shift.value[0]
        yt -= self.shift.value[1]
        xt, yt = np.dot(self.matrix, (xt, yt))

        return self.prepare_outputs(format_info, xt.reshape(v2.shape),
                                    yt.reshape(v3.shape))

    def tanp_to_v2v3(self, xt, yt):
        """ Converts tangent plane coordinates to V2V3 spherical coordinates.
        """
        (xt, yt), format_info = self.prepare_inputs(xt, yt)
        zt = np.full_like(xt, self.__class__.r0)

        # undo corrections:
        xt, yt = np.dot(np.linalg.inv(self.matrix), (xt, yt))
        xt += self.shift.value[0]
        yt += self.shift.value[1]

        # build Euler rotation matrices:
        rotm = [
            rot_mat3d(np.deg2rad(alpha), axis)
            for axis, alpha in enumerate(
                [self.roll.value, self.v3ref.value, self.v2ref.value]
            )
        ]
        inv_euler_rot = np.linalg.inv(np.linalg.multi_dot(rotm))

        # "unrotate" cartezian coordinates back to their original
        # v2ref, v3ref, and roll "positions":
        zcr, xcr, ycr = np.dot(inv_euler_rot, (zt.ravel(), xt.ravel(),
                                               yt.ravel()))

        # convert cartesian to spherical coordinates:
        v2c, v3c = self.cartesian2spherical(zcr, xcr, ycr)

        return self.prepare_outputs(format_info, v2c.reshape(xt.shape),
                                    v3c.reshape(yt.shape))

    def evaluate(self, v2, v3, v2ref, v3ref, roll, matrix, shift):

        """ Evaluate the model on some input variables. """

        # convert spherical coordinates to cartesian assuming unit sphere:
        xyz = self.spherical2cartesian(v2.ravel(), v3.ravel())

        # build Euler rotation matrices:
        rotm = [rot_mat3d(np.deg2rad(alpha), axis)
                for axis, alpha in enumerate([roll, v3ref, v2ref])]
        euler_rot = np.linalg.multi_dot(rotm)
        inv_euler_rot = np.linalg.inv(euler_rot)

        # rotate cartezian coordinates:
        zr, xr, yr = np.dot(euler_rot, xyz)

        # project points onto the tanject plane
        # (tangent to a sphere of radius r0):
        xt = self.__class__.r0 * xr / zr
        yt = self.__class__.r0 * yr / zr
        zt = np.full_like(xt, self.__class__.r0)

        # apply corrections:
        # NOTE: order of transforms may need to be swapped depending on
        #       how shifts are defined.
        xt -= shift[0][0]
        yt -= shift[0][1]
        xt, yt = np.dot(matrix[0], (xt, yt))

        # "unrotate" cartezian coordinates back to their original
        # v2ref, v3ref, and roll "positions":
        zcr, xcr, ycr = np.dot(inv_euler_rot, (zt, xt, yt))

        # convert cartesian to spherical coordinates:
        v2c, v3c = self.cartesian2spherical(zcr, xcr, ycr)

        return v2c.reshape(v2.shape), v3c.reshape(v3.shape)

    @property
    def inverse(self):
        """
        Returns a new `TPCorr` instance which performs the inverse
        transformation of the transformation defined for this `TPCorr` model.

        """
        ishift = -np.dot(self.matrix.value, self.shift.value)
        imatrix = np.linalg.inv(self.matrix.value)
        return TPCorr(v2ref=self.v2ref.value, v3ref=self.v3ref.value,
                      roll=self.roll.value, matrix=imatrix, shift=ishift)

    @classmethod
    def combine(cls, t2, t1):
        """
        Combine transformation ``t2`` with another transformation (``t1``)
        *previously applied* to the coordinates. That is,
        transformation ``t2`` is assumed to *follow* (=applied after) the
        transformation provided by the argument ``t1``.

        """
        if not isinstance(t1, TPCorr):
            raise IncompatibleCorrections(  # pragma: no cover
                "Tangent plane correction 't1' is not a TPCorr instance."
            )

        if not isinstance(t2, TPCorr):
            raise IncompatibleCorrections(  # pragma: no cover
                "Tangent plane correction 't2' is not a TPCorr instance."
            )

        eps_v2 = 10.0 * np.finfo(t2.v2ref.value).eps
        eps_v3 = 10.0 * np.finfo(t2.v3ref.value).eps
        eps_roll = 10.0 * np.finfo(t2.roll.value).eps
        if not (np.isclose(t2.v2ref.value, t1.v2ref.value, rtol=eps_v2) and
                np.isclose(t2.v3ref.value, t1.v3ref.value, rtol=eps_v3) and
                np.isclose(t2.roll.value, t1.roll.value, rtol=eps_roll)):
            raise IncompatibleCorrections(  # pragma: no cover
                "Only combining of correction transformations within the same "
                "tangent plane is supported."
            )

        t1m = t1.matrix.value
        it1m = np.linalg.inv(t1m)
        shift = t1.shift.value + np.dot(it1m, t2.shift.value)
        matrix = np.dot(t2.matrix.value, t1m)

        name = t1.name if t2.name is None else t2.name

        return cls(v2ref=t2.v2ref.value, v3ref=t2.v3ref.value,
                   roll=t2.roll.value, matrix=matrix, shift=shift, name=name)
