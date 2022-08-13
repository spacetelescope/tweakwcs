"""
A module containing unit tests for the `wcsutil` module.

Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
import math
import numpy as np
from astropy.modeling import Model, Parameter
from astropy.modeling.models import AffineTransformation2D, Identity
from astropy import coordinates as coord
from astropy import units as u
import gwcs
from gwcs.geometry import CartesianToSpherical, SphericalToCartesian
from tweakwcs.correctors import WCSCorrector, JWSTWCSCorrector


_S2C = SphericalToCartesian(name='s2c', wrap_lon_at=180)
_C2S = CartesianToSpherical(name='c2s', wrap_lon_at=180)


class DummyWCSCorrector(WCSCorrector):
    def set_correction(self, matrix=[[1, 0], [0, 1]], shift=[0, 0],
                       ref_tpwcs=None, meta=None, **kwargs):
        super().set_correction(matrix=matrix, shift=shift,
                               ref_tpwcs=ref_tpwcs, meta=meta, **kwargs)

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
                     dtype=np.double)
    mat2d = np.array([[cs, sn], [-sn, cs]], dtype=np.double)
    return np.insert(np.insert(mat2d, axis, [0.0, 0.0], 1), axis, axisv, 0)


def create_DetToV2V3(v2ref=0.0, v3ref=0.0, roll=0.0,
                     cd=[[1.0, 0.0], [0.0, 1.0]], crpix=[0, 0]):
    tpcorr = JWSTWCSCorrector._tpcorr_init(v2_ref=v2ref, v3_ref=v3ref, roll_ref=roll)

    afinv = AffineTransformation2D(cd, -np.dot(cd, crpix)).inverse

    JWSTWCSCorrector._tpcorr_combine_affines(
        tpcorr,
        afinv.matrix.value,
        afinv.translation.value
    )

    p = JWSTWCSCorrector._v2v3_to_tpcorr_from_full(tpcorr)
    partial_tpcorr = p.inverse
    partial_tpcorr.inverse = p

    return partial_tpcorr


def create_V2V3ToDet(v2ref=0.0, v3ref=0.0, roll=0.0,
                     cd=[[1.0, 0.0], [0.0, 1.0]], crpix=[0, 0]):
    inv_partial_tpcorr = create_DetToV2V3(
        v2ref=v2ref, v3ref=v3ref, roll=roll, crpix=crpix, cd=cd
    ).inverse
    return inv_partial_tpcorr


class V2V3ToSky(Model):
    """
    Rotates V2-V3 sphere on the sky.
    """
    angles = Parameter()

    _separable = False
    standard_broadcasting = False

    n_inputs = 2
    n_outputs = 2

    def __init__(self, angles, axes_order, name=None):
        self.axes_order = axes_order
        super().__init__(angles=angles, name=name)
        self.inputs = ("v2", "v3")
        self.outputs = ("ra", "dec")

    @staticmethod
    def build_euler_matrix(axis_angle):
        # build Euler rotation matrices:
        rotm = [rot_mat3d(np.deg2rad(alpha), axis)
                for axis, alpha in axis_angle]
        euler_rot = np.linalg.multi_dot(rotm)
        return euler_rot

    @staticmethod
    def cartesian2spherical(x, y, z):
        """ Convert cartesian coordinates to spherical (in deg). """
        return _C2S(x, y, z)

    @staticmethod
    def spherical2cartesian(alpha, delta):
        """ Convert spherical coordinates (in deg) to cartesian. """
        return _S2C(alpha, delta)

    def evaluate(self, v2, v3, angles):
        """ Evaluate the model on some input variables. """

        # convert spherical coordinates to cartesian assuming unit sphere:
        xyz = self.spherical2cartesian(v2.ravel() / 3600., v3.ravel() / 3600.0)

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
        return V2V3ToSkyInv(self.angles.value, self.axes_order)


class V2V3ToSkyInv(Model):
    """
    Rotates V2-V3 sphere on the sky.
    """
    angles = Parameter()

    _separable = False
    standard_broadcasting = False

    n_inputs = 2
    n_outputs = 2

    def __init__(self, angles, axes_order, name=None):
        self.axes_order = axes_order
        super().__init__(angles=angles, name=name)
        self.inputs = ("v2", "v3")
        self.outputs = ("ra", "dec")

    @staticmethod
    def build_euler_matrix(axis_angle):
        # build Euler rotation matrices:
        rotm = [rot_mat3d(np.deg2rad(alpha), axis)
                for axis, alpha in axis_angle]
        euler_rot = np.linalg.multi_dot(rotm)
        return euler_rot

    @staticmethod
    def cartesian2spherical(x, y, z):
        """ Convert cartesian coordinates to spherical (in deg). """
        return _C2S(x, y, z)

    @staticmethod
    def spherical2cartesian(alpha, delta):
        """ Convert spherical coordinates (in deg) to cartesian. """
        return _S2C(alpha, delta)

    def evaluate(self, v2, v3, angles):
        """ Evaluate the model on some input variables. """

        # convert spherical coordinates to cartesian assuming unit sphere:
        xyz = self.spherical2cartesian(v2.ravel(), v3.ravel())

        # build Euler rotation matrices:
        euler_rot = self.__class__.build_euler_matrix(
            [(v[0], -v[1]) if v[0] != 1 else v for v in
             zip(self.axes_order[::-1], (-angles[0])[::-1])][::-1]
        )

        # rotate cartezian coordinates:
        z, x, y = np.dot(euler_rot, xyz)

        # convert cartesian to spherical coordinates:
        ra, dec = self.cartesian2spherical(z, x, y)

        return 3600 * ra, 3600 * dec

    @property
    def inverse(self):
        return V2V3ToSky(self.angles.value, self.axes_order)


def make_mock_jwst_pipeline(v2ref=0, v3ref=0, roll=0, crpix=[512, 512],
                            cd=[[1e-5, 0], [0, 1e-5]], crval=[0, 0],
                            enable_vacorr=True):
    detector = gwcs.coordinate_frames.Frame2D(
        name='detector', axes_order=(0, 1), unit=(u.pix, u.pix)
    )
    v2v3 = gwcs.coordinate_frames.Frame2D(
        name='v2v3', axes_order=(0, 1), unit=(u.arcsec, u.arcsec)
    )
    v2v3vacorr = gwcs.coordinate_frames.Frame2D(
        name='v2v3vacorr', axes_order=(0, 1), unit=(u.arcsec, u.arcsec)
    )
    world = gwcs.coordinate_frames.CelestialFrame(reference_frame=coord.ICRS(),
                                                  name='world')
    det2v2v3 = create_DetToV2V3(v2ref=v2ref / 3600.0, v3ref=v3ref / 3600.0,
                                roll=roll, cd=cd, crpix=crpix)

    v23sky = V2V3ToSky([-v2ref / 3600.0, v3ref / 3600.0, -roll,
                        -crval[1], crval[0]], [2, 1, 0, 1, 2])
    if enable_vacorr:
        pipeline = [(detector, det2v2v3), (v2v3, Identity(2)),
                    (v2v3vacorr, v23sky), (world, None)]
    else:
        pipeline = [(detector, det2v2v3), (v2v3, v23sky), (world, None)]

    return pipeline


def make_mock_jwst_wcs(v2ref=0, v3ref=0, roll=0, crpix=[512, 512],
                       cd=[[1e-5, 0], [0, 1e-5]], crval=[0, 0],
                       enable_vacorr=True):
    pipeline = make_mock_jwst_pipeline(v2ref, v3ref, roll, crpix, cd, crval, enable_vacorr)
    wcs = gwcs.wcs.WCS(pipeline)
    wcs.bounding_box = ((-0.5, 1024 - 0.5), (-0.5, 2048 - 0.5))
    wcs.array_shape = (2048, 1024)
    return wcs
