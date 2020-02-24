"""
A module containing unit tests for the `wcsutil` module.

Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
from itertools import product
import math
import pytest
import numpy as np
from tweakwcs import linearfit, linalg


_LARGE_SAMPLE_SIZE = 1000

_SMALL_SAMPLE_SIZE = 10

_BAD_DATA_FRACTION = 0.2

_TRANSFORM_SELECTOR = {
    'rscale': linearfit.fit_rscale,
    'general': linearfit.fit_general,
    'shift': linearfit.fit_shifts,
}

_ATOL = 10 * _LARGE_SAMPLE_SIZE * np.sqrt(
    np.finfo(linalg._MAX_LINALG_TYPE).eps
)


@pytest.fixture(scope="module")
def ideal_small_data(request):
    # rscale data with proper rotations and no noise
    uv = np.random.random((_SMALL_SAMPLE_SIZE, 2))
    xy = np.random.random((_SMALL_SAMPLE_SIZE, 2))
    wuv = np.random.random(_SMALL_SAMPLE_SIZE)
    wxy = np.random.random(_SMALL_SAMPLE_SIZE)
    return uv, xy, wuv, wxy


@pytest.fixture(scope="function", params=[
    'shifts', 'rscale', 'rscale-flip-x', 'rscale-flip-y', 'affine'
])
def ideal_large_data(request):
    # rscale data with proper rotations and no noise
    uv = np.random.random((_LARGE_SAMPLE_SIZE, 2))

    # assume an image size of 4096x2048:
    uv[:, 0] *= 2047.0
    uv[:, 1] *= 4095.0

    # rotation angle(s):
    angle = 360.0 * np.random.random()  # 0 ... 360

    if request.param == 'shifts':
        angle = (0, 0)
        scale = (1, 1)
        proper = True
        transform = 'shift'

    elif request.param == 'rscale':
        angle = (angle, angle)
        scale = 2 * (0.8 + 0.4 * np.random.random(), )  # 0.8 ... 1.2
        proper = True
        transform = 'rscale'

    elif request.param == 'rscale-flip-x':
        angle = ((angle + 180.0) % 360.0, angle)
        scale = 2 * (0.8 + 0.4 * np.random.random(), )  # 0.8 ... 1.2
        proper = False
        transform = 'rscale'

    elif request.param == 'rscale-flip-y':
        angle = (angle, (angle + 180.0) % 360.0)
        scale = 2 * (0.8 + 0.4 * np.random.random(), )  # 0.8 ... 1.2
        proper = False
        transform = 'rscale'

    elif request.param == 'affine':
        # rotation angles:
        offset = 150.0 * (np.random.random() - 0.5)  # -75 ... 75
        offset += 180.0 * np.random.choice([0.0, 1.0])  # add random axis flip
        angle = (angle, (angle + offset) % 360.0)
        # scales:
        scale = 0.8 + 0.4 * np.random.random(2)  # 0.8 ... 1.2
        # proper:
        rad = np.deg2rad(angle)
        proper = (np.prod(np.cos(rad)) + np.prod(np.sin(rad))) > 0
        transform = 'general'

    shift = 200.0 * (np.random.random(2) - 0.5)  # -100 ... +100
    rmat = linearfit.build_fit_matrix(angle, scale)
    skew = angle[1] - angle[0]

    # apply rscale
    xy = np.dot(uv, rmat.T) + shift

    return uv, xy, angle, scale, shift, rmat, proper, skew, transform


@pytest.fixture(scope="function",
                params=[v for v in product(*(2 * [[False, True]]))])
def weight_data(request):
    nbd = int(_BAD_DATA_FRACTION * _LARGE_SAMPLE_SIZE)
    minv = 1000.0
    maxv = 1.0e6

    if not any(request.param):
        wxy = None
        wuv = None
        idx_xy = (np.array([], dtype=np.int), )
        idx_uv = (np.array([], dtype=np.int), )
        bd_xy = np.zeros((0, 2))
        bd_uv = np.zeros((0, 2))

    elif all(request.param):
        wxy = np.random.random(_LARGE_SAMPLE_SIZE)
        wuv = np.random.random(_LARGE_SAMPLE_SIZE)

        # split indices into (almost) equal parts:
        nbdxy = int((0.3 + 0.4 * np.random.random()) * nbd)
        nbduv = nbd - nbdxy
        idx_xy = (np.random.choice(np.arange(_LARGE_SAMPLE_SIZE),
                                   nbdxy, replace=False), )
        idx_uv = (np.random.choice(np.arange(_LARGE_SAMPLE_SIZE),
                                   nbduv, replace=False), )

        wxy[idx_xy] = 0.0
        wuv[idx_uv] = 0.0

        sign = np.random.choice([-1, 1], (nbdxy, 2))
        bd_xy = sign * (minv + (maxv - minv) * np.random.random((nbdxy, 2)))

        sign = np.random.choice([-1, 1], (nbduv, 2))
        bd_uv = sign * (minv + (maxv - minv) * np.random.random((nbduv, 2)))

    elif request.param[0] and not request.param[1]:
        idx = np.random.choice(np.arange(_LARGE_SAMPLE_SIZE),
                               nbd, replace=False)
        idx_xy = (idx, )
        idx_uv = (np.array([], dtype=np.int), )
        wxy = np.random.random(_LARGE_SAMPLE_SIZE)
        wxy[idx_xy] = 0.0
        wuv = None
        sign = np.random.choice([-1, 1], (nbd, 2))
        bd_xy = sign * (minv + (maxv - minv) * np.random.random((nbd, 2)))
        bd_uv = np.zeros((0, 2))

    else:
        idx = np.random.choice(np.arange(_LARGE_SAMPLE_SIZE), nbd,
                               replace=False)
        idx_uv = (idx, )
        idx_xy = (np.array([], dtype=np.int), )
        wuv = np.random.random(_LARGE_SAMPLE_SIZE)
        wuv[idx_uv] = 0.0
        wxy = None
        sign = np.random.choice([-1, 1], (nbd, 2))
        bd_uv = sign * (minv + (maxv - minv) * np.random.random((nbd, 2)))
        bd_xy = np.zeros((0, 2))

    return wxy, wuv, idx_xy, idx_uv, bd_xy, bd_uv


@pytest.fixture(scope="module")
def tiny_zero_data():
    """ Return a tuple of (xy, uv, wxy, wuv)"""
    return np.zeros((3, 2)), np.zeros((3, 2)), np.zeros(3), np.zeros(3)


@pytest.fixture(scope="function", params=[
    linearfit.fit_shifts, linearfit.fit_rscale, linearfit.fit_general
])
def fit_functions(request):
    return request.param


def test_build_fit_matrix_identity():
    i = np.identity(2)

    r = linearfit.build_fit_matrix(0)  # also test that default scale value = 1
    assert np.allclose(i, r, rtol=0, atol=_ATOL)

    r = linearfit.build_fit_matrix((0, 0), (1, 1))
    assert np.allclose(i, r, rtol=0, atol=_ATOL)


@pytest.mark.parametrize('rot', [1, 35, 75, 95, 155, 189, 261, 299, 358])
def test_build_fit_matrix_rot(rot):
    i = np.identity(2)
    m = linearfit.build_fit_matrix(rot)

    minv = linearfit.build_fit_matrix(360 - rot)
    assert np.allclose(i, np.dot(m, minv), rtol=0, atol=_ATOL)


@pytest.mark.parametrize('rot, scale', [
    ((1, 4), (2.4, 5.6)),
    ((31, 78), (0.9, 1.3)),
])
def test_build_fit_matrix_generalized(rot, scale):
    i = np.identity(2)
    m = linearfit.build_fit_matrix(rot, scale)

    # check scale:
    assert np.allclose(np.sqrt(np.sum(m**2, axis=0)), scale,
                       rtol=0, atol=_ATOL)
    ms = np.diag(scale)

    # check rotations:
    mr = linearfit.build_fit_matrix(rot, 1)
    mrinv = linearfit.build_fit_matrix(rot[::-1], 1).T
    assert np.allclose(np.linalg.det(mr) * i, np.dot(mr, mrinv),
                       rtol=0, atol=_ATOL)

    assert np.allclose(m, np.dot(mr, ms), rtol=0, atol=_ATOL)


@pytest.mark.parametrize('uv, xy, wuv, wxy', [
    (np.zeros(10), np.zeros(10), None, None),
    (np.zeros((10, 2, 2)), np.zeros(10), None, None),
    (np.zeros((10, 2)), np.zeros((11, 2)), None, None),
    3 * (np.zeros((10, 2)), ) + (None, ),
    2 * (np.zeros((10, 2)), ) + (None, np.zeros((10, 2))),
    2 * (np.zeros((10, 2)), ) + (None, np.zeros((5, 2))),
    2 * (np.zeros((10, 2)), ) + (np.zeros((5, 2)), None),
])
def test_iter_linear_fit_invalid_shapes(uv, xy, wuv, wxy):
    # incorrect coordinate array dimensionality:
    with pytest.raises(ValueError):
        linearfit.iter_linear_fit(xy, uv, wxy=wxy, wuv=wuv)


@pytest.mark.parametrize('nclip, sigma', [
    (3, None), (-3, None), (3, -1), (-1, 3), (3, (1.0, 'invalid')),
])
def test_iter_linear_fit_invalid_sigma_nclip(ideal_small_data, nclip, sigma):
    uv, xy, _, _ = ideal_small_data

    with pytest.raises(ValueError):
        linearfit.iter_linear_fit(xy, uv, nclip=nclip, sigma=sigma)


def test_iter_linear_fit_invalid_fitgeom(ideal_small_data):
    uv, xy, _, _ = ideal_small_data

    with pytest.raises(ValueError):
        linearfit.iter_linear_fit(xy, uv, fitgeom='invalid')


@pytest.mark.parametrize('nclip, sigma, clip_accum, weights, noise', [
    (None, 2, True, False, False),
    (None, 2, True, True, False),
    (2, 0.05, False, True, True),
])
def test_iter_linear_fit_special_cases(ideal_large_data, nclip, sigma,
                                       clip_accum, weights, noise):
    uv, xy, _, _, shift, rmat, _, _, fitgeom = ideal_large_data
    if weights:
        wxy, wuv = 0.1 + 0.9 * np.random.random((2, xy.shape[0]))
    else:
        wxy = None
        wuv = None

    if noise:
        xy = xy + np.random.normal(0, 0.01, xy.shape)
        atol = 0.01
    else:
        atol = _ATOL

    fit = linearfit.iter_linear_fit(xy, uv, wxy, wuv, fitgeom=fitgeom,
                                    nclip=nclip, center=(0, 0), sigma=1,
                                    clip_accum=clip_accum)

    assert np.allclose(fit['shift'], shift, rtol=0, atol=atol)
    assert np.allclose(fit['matrix'], rmat, rtol=0, atol=atol)


@pytest.mark.parametrize('weights', [False, True])
def test_iter_linear_fit_1point(weights):
    xy = np.array([[1.0, 2.0]])
    shifts = 20 * (np.random.random(2) - 0.5)
    if weights:
        wxy, wuv = 0.1 + 0.9 * np.random.random((2, xy.shape[0]))
    else:
        wxy, wuv = None, None

    fit = linearfit.iter_linear_fit(xy, xy + shifts, wxy=wxy, wuv=wuv,
                                    fitgeom='shift', nclip=0)

    assert np.allclose(fit['shift'], -shifts, rtol=0, atol=_ATOL)
    assert np.allclose(fit['matrix'], np.identity(2), rtol=0, atol=_ATOL)


def test_iter_linear_fit_fitgeom_clip_all_data(ideal_large_data):
    # Test that clipping is interrupted if number of sources after clipping
    # is below minobj for a given fit:
    xy, uv, _, _, _, _, _, _, fitgeom = ideal_large_data
    ndata = xy.shape[0]
    uv = uv + np.random.normal(0, 0.01, (ndata, 2))
    wxy, wuv = 0.1 + 0.9 * np.random.random((2, ndata))

    fit = linearfit.iter_linear_fit(
        xy, uv, wxy, wuv, fitgeom=fitgeom, sigma=1e-50, nclip=100
    )

    assert np.count_nonzero(fit['fitmask']) == len(xy)
    assert fit['eff_nclip'] == 0


def test_compute_stat_invalid_weights(ideal_small_data):
    pts, _, _, _ = ideal_small_data
    weights = np.zeros(pts.shape[0])
    fit = {}
    linearfit._compute_stat(fit, pts, weights)
    assert math.isnan(fit['rmse'])
    assert math.isnan(fit['mae'])
    assert math.isnan(fit['std'])


@pytest.mark.parametrize('fit_function', [
    linearfit.fit_rscale, linearfit.fit_general,
])
def test_fit_detect_colinear_points(fit_function, tiny_zero_data):
    xy, uv, _, _ = tiny_zero_data
    xy = xy + [1, 2]
    with pytest.raises(linearfit.SingularMatrixError):
        fit_function(xy, uv)


def test_fit_detect_zero_weights(fit_functions, tiny_zero_data):
    xy, uv, wxy, _ = tiny_zero_data
    # all weights are zero:
    with pytest.raises(ValueError):
        fit_functions(xy, uv, wxy=wxy)


def test_fit_detect_negative_weights(fit_functions, tiny_zero_data):
    xy, uv, wuv, _ = tiny_zero_data
    wuv.copy()
    wuv[0] = -1

    # some weights are negative (=invalid):
    with pytest.raises(ValueError):
        fit_functions(xy, uv, wuv=wuv)


@pytest.mark.parametrize('fit_function, npts', [
    (linearfit.fit_shifts, 0),
    (linearfit.fit_rscale, 1),
    (linearfit.fit_general, 2),
])
def test_fit_general_too_few_points(fit_function, npts):
    with pytest.raises(linearfit.NotEnoughPointsError):
        fit_function(np.zeros((npts, 2)), np.zeros((npts, 2)))


@pytest.mark.parametrize(
    'clip_accum, noise',
    [v for v in product(*(2 * [[False, True]]))]
)
def test_iter_linear_fit_clip_style(ideal_large_data, weight_data,
                                    clip_accum, noise):
    """ Test clipping behavior. Test that weights exclude "bad" data. """
    uv, xy, angle, scale, shift, rmat, proper, skew, fitgeom = ideal_large_data
    wxy, wuv, idx_xy, idx_uv, bd_xy, bd_uv = weight_data

    noise_sigma = 0.01
    npts = xy.shape[0]

    # add noise to data
    if noise:
        xy = xy + np.random.normal(0, noise_sigma, (npts, 2))
        atol = 10 * noise_sigma
        nclip = 3
    else:
        atol = _ATOL
        nclip = 0

    if wxy is not None:
        xy[idx_xy] += bd_xy

    if wuv is not None:
        uv = uv.copy()
        uv[idx_uv] += bd_uv

    fit = linearfit.iter_linear_fit(
        xy, uv, wxy=wxy, wuv=wuv, fitgeom=fitgeom, sigma=2,
        clip_accum=clip_accum, nclip=nclip
    )

    shift_with_center = np.dot(rmat, fit['center']) - fit['center'] + shift

    assert np.allclose(fit['shift'], shift_with_center, rtol=0, atol=atol)
    assert np.allclose(fit['matrix'], rmat, rtol=0, atol=atol)
    assert np.allclose(fit['rmse'], 0, rtol=0, atol=atol)
    assert np.allclose(fit['mae'], 0, rtol=0, atol=atol)
    assert np.allclose(fit['std'], 0, rtol=0, atol=atol)
    assert fit['proper'] == proper
    if nclip:
        assert fit['eff_nclip'] > 0
        assert fit['fitmask'].sum(dtype=np.int) < npts
    else:
        assert fit['eff_nclip'] == 0
        assert (fit['fitmask'].sum(dtype=np.int) == npts -
                np.union1d(idx_xy[0], idx_uv[0]).size)
