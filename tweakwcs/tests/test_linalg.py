"""
A module containing unit tests for the `linalg` module.

Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
import pytest
import numpy as np

from tweakwcs import linalg, linearfit


@pytest.mark.parametrize('tp,expected', [
    (np.float16, False),
    (np.longdouble, True),
])
def test_longdouble_cmp(tp, expected):
    assert linalg._is_longdouble_lte_flt_type(tp) == expected


def test_inv_order2():
    feps = 1000 * np.finfo(np.double).eps

    # Generate a 2D rotation/scale/skew matrix + some random noise.
    # The reason for this complicated approach is to avoid creating
    # singular matrices.
    angle1 = 360 * np.random.random()
    angle2 = angle1 + 50 * (np.random.random() - 0.5)
    scale1 = 0.7 + 0.6 * np.random.random()
    scale2 = 0.7 + 0.6 * np.random.random()
    a = linearfit.build_fit_matrix((angle1, angle2), (scale1, scale2))

    x = linalg.inv(a)
    r = np.identity(2) - np.dot(a, x)

    # Use Morris Newman's formula to asses the quality of the inversion
    # (see https://nvlpubs.nist.gov/nistpubs/jres/78B/jresv78Bn2p65_A1b.pdf
    #  January 3, 1974).
    err = 2.0 * np.abs(np.dot(x, r)).max() / (1.0 - np.abs(r).max())
    assert err < feps


def test_inv_nonsquare():
    with pytest.raises(np.linalg.LinAlgError):
        linalg.inv(np.empty((1, 2)))


def test_inv_singular():
    arr = np.array([[1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(np.linalg.LinAlgError):
        linalg.inv(arr)


def test_inv_nan():
    arr = np.array([[1.0, 1.0, -1.], [2.0, -2.1, 4.0], [-1.0, np.nan, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        linalg.inv(arr)


def test_inv_high_dim():
    arr = np.random.random((4, 4, 4))
    with pytest.raises(np.linalg.LinAlgError):
        linalg.inv(arr)
