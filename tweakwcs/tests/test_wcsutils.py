"""
A module containing unit tests for the `wcsutil` module.

Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
import pytest
import numpy as np

from tweakwcs import wcsutils


@pytest.mark.parametrize('x,y,z', [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (-1, 0, 0),
    (0, -1, 0),
    (0, 0, -1),
])
def test_cartesian_spherical_cartesian_roundtrip_special(x, y, z):
    feps = 100 * np.finfo(np.float64).eps
    xyz = wcsutils.spherical_to_cartesian(
        *wcsutils.cartesian_to_spherical(x, y, z)
    )
    assert np.allclose((x, y, z), xyz, rtol=0, atol=feps)


def test_cartesian_spherical_cartesian_roundtrip_rand():
    feps = 100 * np.finfo(np.float64).eps
    xyz = np.random.random((100, 3))
    xyz /= np.linalg.norm(xyz, axis=1)[:, np.newaxis]
    x, y, z = xyz.T
    rx, ry, rz = wcsutils.spherical_to_cartesian(
        *wcsutils.cartesian_to_spherical(x, y, z)
    )
    assert np.allclose(rx, x, rtol=0, atol=feps)
    assert np.allclose(ry, y, rtol=0, atol=feps)
    assert np.allclose(rz, z, rtol=0, atol=feps)


def test_spherical_cartesian_spherical_roundtrip_ugrid():
    feps = 100 * np.finfo(np.float64).eps
    angles = np.linspace(0, 360, 13) - 180
    alpha0 = np.repeat(angles, angles.size)
    delta0 = np.tile(angles / 2, angles.size)
    alpha, delta = wcsutils.cartesian_to_spherical(
        *wcsutils.spherical_to_cartesian(alpha0, delta0)
    )
    assert np.allclose(alpha, alpha0, rtol=0, atol=feps)
    assert np.allclose(delta, delta0, rtol=0, atol=feps)


@pytest.mark.parametrize('angle', np.linspace(-2 * np.pi, 2 * np.pi, 100))
def test_planar_rot_3d(angle):
    feps = 100 * np.finfo(np.float64).eps
    ref = ((angle - np.pi) % (2 * np.pi)) - np.pi
    for axis in (0, 1, 2, 1.0, 2.0, -0.0):
        mat_2d = np.delete(
            np.delete(
                wcsutils.planar_rot_3d(angle=angle, axis=axis), int(axis), 0
            ), int(axis), 1
        )
        assert np.allclose(np.linalg.det(mat_2d), 1, rtol=0, atol=feps)
        assert np.allclose(np.arctan2(*([-1, 1] * mat_2d[:, 0][::-1])),
                           ref, rtol=feps, atol=feps)
        assert np.allclose(np.arctan2(*mat_2d[:, 1]), ref, rtol=feps,
                           atol=feps)


@pytest.mark.parametrize('axis', [-1, 3, 4.5])
def test_test_planar_rot_3d_axis_out_of_range(axis):
    with pytest.raises(ValueError):
        wcsutils.planar_rot_3d(angle=np.random.random(), axis=axis)
