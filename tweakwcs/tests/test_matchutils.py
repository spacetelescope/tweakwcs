"""
A module containing unit tests for the `wcsutil` module.

Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
from itertools import product
import random
import pytest
import numpy as np

from astropy.table import Table
from astropy.wcs import WCS

import tweakwcs
from tweakwcs.matchutils import (_xy_2dhist, _estimate_2dhist_shift,
                                 _find_peak, TPMatch, MatchCatalogs)
from .helper_tpwcs import DummyTPWCS


_ATOL = 10 * np.finfo(np.array([1.]).dtype).eps


def test_xy_2dhist():
    npts = 2000
    uv = 1023 * np.random.random((npts, 2))
    xy = uv + [2, 5]
    xy += np.random.normal(0, 0.5, (npts, 2))
    h = _xy_2dhist(xy, uv, 10)
    assert np.argmax(h) == 327
    assert npts <= h.sum() < 2 * npts


@pytest.mark.parametrize('shape, mask', (
    x for x in product([(3, 4), (7, 8), (3, 8)], [None, 0, 1])
))
def test_find_peak_nodata_all_zeros(shape, mask):
    data = np.zeros(shape)
    if mask is not None:
        if mask:
            mask = np.random.choice([True, False], shape)
            mask[0, 0] = True  # make sure at least one element is non-zero
        else:
            mask = np.zeros(shape, dtype=np.bool_)

    coord, fit_status, fit_box = _find_peak(data, peak_fit_box=5, mask=mask)
    assert data[fit_box].shape == shape
    assert np.allclose(coord, [0.5 * (x - 1) for x in shape[::-1]],
                       rtol=0, atol=_ATOL)
    assert fit_status == 'ERROR:NODATA'


@pytest.mark.parametrize('along_row', [True, False])
def test_find_peak_edge_1pix_valid_strip(along_row):
    data = np.zeros((10, 20))
    mask = np.zeros((10, 20), dtype=np.bool_)
    if along_row:
        row = random.choice(range(10))
        data[row, :] = 1.0
        mask[row, :] = True
        coord0 = (0, row)
    else:
        col = random.choice(range(20))
        data[:, col] = 1.0
        mask[:, col] = True
        coord0 = (col, 0)

    coord, fit_status, fit_box = _find_peak(data, peak_fit_box=5, mask=mask)
    assert fit_status == 'WARNING:EDGE'
    assert np.allclose(coord, coord0, rtol=0, atol=_ATOL)


def test_find_peak_nodata_peak_is_invalid():
    data = np.zeros((14, 17))
    mask = np.zeros((14, 17), dtype=np.bool_)

    col = random.choice(range(17))
    mask[9, col] = True
    data[9, 4] = 1.0
    mask[9, 4] = False
    coord0 = (8, 6.5)

    coord, fit_status, fit_box = _find_peak(data, peak_fit_box=5, mask=mask)
    assert fit_status == 'ERROR:NODATA'
    assert np.allclose(coord, coord0, rtol=0, atol=_ATOL)


def test_find_peak_few_data_center_of_mass():
    data = np.zeros((20, 20))
    mask = np.zeros((20, 20), dtype=np.bool_)

    col = random.choice(range(5, 16))
    data[9, col] = 1.0
    mask[9, col] = True
    coord0 = (col, 9)

    coord, fit_status, fit_box = _find_peak(data, mask=mask)
    assert fit_status == 'WARNING:CENTER-OF-MASS'
    assert np.allclose(coord, coord0, rtol=0, atol=_ATOL)


def test_find_peak_few_data_for_center_of_mass():
    data = np.zeros((21, 21))
    mask = np.zeros((21, 21), dtype=np.bool_)
    i = random.choice(range(5, 16))
    j = random.choice(range(5, 16))
    data[i, j] = 1.0
    data[i - 1, j - 1] = -1.0
    mask[i, j] = True
    mask[i - 1, j - 1] = True
    coord, fit_status, fit_box = _find_peak(data, peak_fit_box=3, mask=mask)
    assert fit_status == 'ERROR:NODATA'
    assert np.allclose(coord, (10, 10), rtol=0, atol=_ATOL)


def test_find_peak_negative_peak():
    data = np.zeros((11, 11))
    i = random.choice(range(2, 9))
    j = random.choice(range(2, 9))
    data[i, j] = -1.0
    coord, fit_status, fit_box = _find_peak(data, peak_fit_box=2)
    assert fit_status == 'ERROR:NODATA'
    assert np.allclose(coord, (5, 5), rtol=0, atol=_ATOL)


def test_find_peak_tiny_box_1pix():
    data = np.zeros((4, 4))
    mask = np.zeros((4, 4), dtype=np.bool_)

    mask[2, 2] = True
    data[2, 2] = 1.0
    coord0 = (2, 2)

    coord, fit_status, fit_box = _find_peak(data, peak_fit_box=5, mask=mask)
    assert np.allclose(coord, coord0, rtol=0, atol=_ATOL)
    assert fit_status == 'WARNING:CENTER-OF-MASS'


def test_find_peak_negative_box_size():
    with pytest.raises(ValueError):
        _find_peak(np.zeros((2, 2)), peak_fit_box=-1)


def test_find_peak_success():
    data = np.zeros((21, 21))
    y, x = np.indices(data.shape)
    data = 100 * np.exp(-0.5 * ((x - 8)**2 + (y - 11)**2))
    coord, fit_status, fit_box = _find_peak(data, peak_fit_box=3)
    assert fit_status == 'SUCCESS'
    assert np.allclose(coord, (8, 11), rtol=0, atol=1e-6)


def test_find_peak_fail_lstsq():
    data = np.zeros((11, 21))
    data[6, 7] = 10
    data[8, 7] = np.nan
    coord, fit_status, fit_box = _find_peak(data, peak_fit_box=7)
    assert fit_status == 'WARNING:CENTER-OF-MASS'
    assert np.allclose(coord, (7, 6), rtol=0, atol=1e-6)


def test_find_peak_nodata_after_fail():
    data = np.zeros((21, 21))
    i = random.choice(range(6, 14))
    j = random.choice(range(6, 14))
    data[i, j] = 1.0
    data[i - 1, j - 1] = -1.0
    data[i + 1, j + 1] = np.nan
    coord, fit_status, fit_box = _find_peak(data, peak_fit_box=5)
    assert fit_status == 'ERROR:NODATA'
    assert np.allclose(coord, (10, 10), rtol=0, atol=_ATOL)


def test_find_peak_badfit():
    data = np.zeros((21, 21))
    y, x = np.indices(data.shape)
    data = x + y
    data[(x < 5) | (x > 11) | (y < 8) | (y > 14)] = 0
    coord, fit_status, fit_box = _find_peak(data, peak_fit_box=7)
    assert fit_status == 'WARNING:BADFIT'
    assert np.allclose(coord, (11, 14), rtol=0, atol=1e-6)


def test_find_peak_fit_over_edge():
    data = np.zeros((21, 21))
    y, x = np.indices(data.shape)
    data = 100 * np.exp(-0.5 * (x**2 + (y - 11)**2))
    data[:, 0] = 0.0
    coord, fit_status, fit_box = _find_peak(data, peak_fit_box=7)
    assert fit_status == 'WARNING:EDGE'
    assert np.allclose(coord, (1, 11), rtol=0, atol=1e-6)


@pytest.mark.parametrize('shift', [100, 2])
def test_estimate_2dhist_shift_one_bin(shift):
    imgxy = np.zeros((1, 2))
    refxy = imgxy - shift
    expected = 2 * (0 if shift > 3 else shift, )
    assert _estimate_2dhist_shift(imgxy, refxy, searchrad=3) == expected


def test_estimate_2dhist_shift_edge():
    imgxy = np.array([[0, 0], [0, 1], [3, 4], [7, 8]])
    shifts = np.array([[3, 0], [3, 0], [1, 2], [0, 1]])
    refxy = imgxy - shifts
    assert _estimate_2dhist_shift(imgxy, refxy, searchrad=3) == (3.0, 0.0)


def test_estimate_2dhist_shift_fit_failed(monkeypatch):
    def fake_find_peak(data, peak_fit_box=5, mask=None):
        return (0, 0), 'ERROR', None

    monkeypatch.setattr(tweakwcs.matchutils, '_find_peak', fake_find_peak)

    imgxy = np.array([[0, 0], [0, 1], [3, 4], [7, 8]])
    shifts = np.array([[3, 0], [3, 0], [1, 2], [0, 1]])
    refxy = imgxy - shifts
    assert _estimate_2dhist_shift(imgxy, refxy, searchrad=3) == (0.0, 0.0)


def test_estimate_2dhist_shift_two_equal_maxima(caplog):
    imgxy = np.array([[0, 1], [0, 1]])
    refxy = np.array([[1, 0], [0, 2]])
    assert _estimate_2dhist_shift(imgxy, refxy, searchrad=3) == (-0.5, 0.0)
    assert (caplog.record_tuples[-2][-1] == "Unable to estimate significance "
            "of the detection of the initial shift.")


@pytest.mark.parametrize('searchrad, separation, tolerance', [
    (0, 1, 1), (1, 0, 1), (1, 1, 0)
])
def test_tpmatch_bad_pars(searchrad, separation, tolerance):
    with pytest.raises(ValueError):
        TPMatch(searchrad=searchrad, separation=separation,
                tolerance=tolerance)


@pytest.mark.parametrize('refcat, imcat, tp_wcs, exception', [
    ([], [], None, TypeError),
    (Table([[], []]), [], None, ValueError),
    (Table([[1], [1]]), [], None, TypeError),
    (Table([[1], [1]]), Table([[], []]), None, ValueError),
    (Table([[1], [1]]), Table([[1], [1]]), None, KeyError),
    (Table([[1], [1]], names=('TPx', '2')), Table([[1], [1]]), None, KeyError),
    (Table([[1], [1]], names=('TPx', 'TPy')), Table([[1], [1]]),
     None, KeyError),
    (Table([[1], [1]], names=('TPx', 'TPy')),
     Table([[1], [1]], names=('TPx', '2')),
     None, KeyError),
    (Table([[1], [1]], names=('RA', '-')),
     Table([[1], [1]], names=('TPx', '2')),
     DummyTPWCS(WCS()), KeyError),
    (Table([[1], [1]], names=('RA', 'DEC')),
     Table([[1], [1]], names=('TPx', '2')),
     DummyTPWCS(WCS()), KeyError),
])
def test_tpmatch_bad_call_pars(refcat, imcat, tp_wcs, exception):
    tpmatch = TPMatch()
    with pytest.raises(exception):
        tpmatch(refcat, imcat, tp_wcs)


@pytest.mark.parametrize('tp_wcs, use2dhist', [
    (None, False),
    (None, True),
    (DummyTPWCS(WCS()), False),
    (DummyTPWCS(WCS()), True),
])
def test_tpmatch(tp_wcs, use2dhist):
    tpmatch = TPMatch(use2dhist=use2dhist)
    if tp_wcs:
        imcat = Table([[1], [1]], names=('x', 'y'), meta={'name': None})
        refcat = Table([[1], [1]], names=('RA', 'DEC'), meta={'name': None})
    else:
        refcat = Table([[1], [1]], names=('TPx', 'TPy'), meta={'name': None})
        imcat = Table([[1], [1]], names=('TPx', 'TPy'), meta={'name': None})

    tpmatch(refcat, imcat, tp_wcs)


def test_match_catalogs_abc():
    class DummyMatchCatalogs(MatchCatalogs):
        def __call__(self, refcat, imcat):
            super().__call__(refcat, imcat)
    assert DummyMatchCatalogs()(None, None) is None
