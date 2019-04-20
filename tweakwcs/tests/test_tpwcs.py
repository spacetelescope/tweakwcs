"""
A module containing unit tests for the `tpwcs` module.

Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
# from itertools import product
# import random
# import math
import sys
import copy
import pytest

import numpy as np

from astropy.modeling import models
import gwcs
from astropy import wcs as fitswcs

from tweakwcs.linearfit import build_fit_matrix
from .helper_tpwcs import (make_mock_jwst_wcs, make_mock_jwst_pipeline,
                           DummyTPWCS, DetToV2V3, V2V3ToDet)
from .helper_tpwcs import TPCorr as MockTPCorr


_ATOL = 100 * np.finfo(np.array([1.]).dtype).eps


def test_tpwcs():
    tpwcs = DummyTPWCS(None, meta={})
    x, y, ra, dec = np.random.random(4)
    matrix = np.random.random((2, 2))
    shift = np.random.random(2)

    assert tpwcs.world_to_det(ra, dec) == (ra, dec)
    assert tpwcs.world_to_tanp(ra, dec) == (ra, dec)
    assert tpwcs.det_to_world(x, y) == (x, y)
    assert tpwcs.det_to_tanp(x, y) == (x, y)
    assert tpwcs.tanp_to_det(x, y) == (x, y)
    assert tpwcs.tanp_to_world(x, y) == (x, y)
    assert tpwcs.tanp_center_pixel_scale == 1
    assert tpwcs.wcs is None
    assert tpwcs.original_wcs is None
    assert isinstance(tpwcs.copy(), DummyTPWCS)
    assert tpwcs.bounding_box is None

    tpwcs.set_correction(matrix=matrix, shift=shift,
                         meta={'pytest': 'ABC.TPWCS'}, pytest_kwarg=True)
    assert np.all(tpwcs.meta['matrix'] == matrix)
    assert np.all(tpwcs.meta['shift'] == shift)
    assert tpwcs.meta['pytest'] == 'ABC.TPWCS'

    with pytest.raises(TypeError) as arg_err:
        tpwcs.set_correction(matrix, shift, {'pytest': 'ABC.TPWCS'},
                             'some_weird_arg')
    assert (arg_err.value.args[0] == "set_correction() takes from 1 "
            "to 4 positional arguments but 5 were given")


def test_mock_jwst_gwcs():
    w = make_mock_jwst_wcs(v2ref=123, v3ref=500, roll=115, crpix=[512, 512],
                           cd=[[1e-5, 0], [0, 1e-5]], crval=[82, 12])

    assert np.allclose(w.invert(*w(23, 1023)), (23, 1023))


@pytest.mark.parametrize('crpix, cd', [
    (np.zeros(3), np.diag(np.ones(3))),
    (np.zeros((2, 2)), np.diag(np.ones(2))),
])
def test_mock_wcs_fails(crpix, cd):
    from astropy.modeling import InputParameterError
    with pytest.raises(InputParameterError):
        make_mock_jwst_wcs(v2ref=123, v3ref=500, roll=15, crpix=crpix,
                           cd=cd, crval=[82, 12])
    with pytest.raises(InputParameterError):
        DetToV2V3(v2ref=123, v3ref=500, roll=15, crpix=crpix, cd=cd)
    with pytest.raises(InputParameterError):
        V2V3ToDet(v2ref=123, v3ref=500, roll=15, crpix=crpix, cd=cd)


def test_v2v3todet_roundtrips():
    s = 1.0e-5
    crpix = np.random.random(2)
    alpha = 0.25 * np.pi * np.random.random()
    x, y = 1024 * np.random.random(2)
    v2, v3 = 45 * np.random.random(2)
    cd = [[s * np.cos(alpha), -s * np.sin(alpha)],
          [s * np.sin(alpha), s * np.cos(alpha)]]

    d2v = DetToV2V3(v2ref=123.0, v3ref=500.0, roll=15.0, crpix=crpix, cd=cd)
    v2d = V2V3ToDet(v2ref=123.0, v3ref=500.0, roll=15.0, crpix=crpix, cd=cd)

    assert np.allclose(d2v.inverse(*d2v(x, y)), (x, y),
                       rtol=100 * _ATOL, atol=100 * _ATOL)

    assert (
        np.allclose(
            V2V3ToDet.spherical2cartesian(*v2d.inverse(*v2d(v2, v3))),
            V2V3ToDet.spherical2cartesian(v2, v3),
            rtol=100 * _ATOL, atol=_ATOL
        ) or np.allclose(
            -V2V3ToDet.spherical2cartesian(*v2d.inverse(*v2d(v2, v3))),
            V2V3ToDet.spherical2cartesian(v2, v3),
            rtol=100 * _ATOL, atol=_ATOL
        )
    )
    assert np.allclose(v2d(*d2v(x, y)), (x, y),
                       rtol=100 * _ATOL, atol=100 * _ATOL)

    assert (
        np.allclose(
            V2V3ToDet.spherical2cartesian(*d2v(*v2d(v2, v3))),
            V2V3ToDet.spherical2cartesian(v2, v3),
            rtol=100 * _ATOL, atol=100 * _ATOL
        ) or np.allclose(
            -V2V3ToDet.spherical2cartesian(*d2v(*v2d(v2, v3))),
            V2V3ToDet.spherical2cartesian(v2, v3),
            rtol=100 * _ATOL, atol=100 * _ATOL
        )
    )


def test_jwst_import_failed(monkeypatch):
    dummy_wcs = gwcs.wcs.WCS(models.Identity(2), 'det', 'world')
    restore_modules = {}
    for k in list(sys.modules.keys()):
        if k.startswith(('jwst')):
            restore_modules[k] = sys.modules[k]  # pragma: no cover
            sys.modules[k] = None  # pragma: no cover
        elif k.startswith('tweakwcs') or 'tpwcs' in k or 'TPCorr' in k:
            restore_modules[k] = sys.modules[k]
            del sys.modules[k]

    from tweakwcs import tpwcs

    with pytest.raises(ImportError):
        tpwcs.JWSTgWCS(dummy_wcs, {})
    sys.modules.update(restore_modules)


from tweakwcs import tpwcs
_TPCORRS = [MockTPCorr]
if tpwcs.TPCorr is not None:
    from jwst.transforms.tpcorr import TPCorr as JWSTTPCorr  # pragma: no cover
    _TPCORRS.append(JWSTTPCorr)  # pragma: no cover


@pytest.mark.parametrize('tpcorr', _TPCORRS)
def test_jwst_wcs_corr_applied(tpcorr, mock_jwst_wcs):
    tpwcs.TPCorr = tpcorr
    w = make_mock_jwst_wcs(
        v2ref=123.0, v3ref=500.0, roll=115.0, crpix=[512.0, 512.0],
        cd=[[1.0e-5, 0.0], [0.0, 1.0e-5]], crval=[82.0, 12.0]
    )

    wc = tpwcs.JWSTgWCS(
        w, {'v2_ref': 123.0, 'v3_ref': 500.0, 'roll_ref': 115.0}, meta={}
    )
    wc.set_correction(meta={'dummy_meta': None}, dummy_par=1)
    assert 'v2v3corr' in wc.wcs.available_frames
    assert 'dummy_meta' in wc.meta


@pytest.mark.parametrize('tpcorr', _TPCORRS)
def test_jwst_wcs_corr_are_being_combined(tpcorr, mock_jwst_wcs):
    tpwcs.TPCorr = tpcorr
    wc = tpwcs.JWSTgWCS(
        mock_jwst_wcs, {'v2_ref': 123.0, 'v3_ref': 500.0, 'roll_ref': 115.0}
    )
    matrix1 = np.array([[1.0, 0.2], [-0.3, 1.1]])
    shift1 = np.array([5.0, -7.0])
    wc.set_correction(matrix=matrix1, shift=shift1)
    assert 'v2v3corr' in wc.wcs.available_frames

    matrix2 = np.linalg.inv(matrix1)
    shift2 = -np.dot(matrix1, shift1)
    wc = tpwcs.JWSTgWCS(
        wc.wcs, {'v2_ref': 123.0, 'v3_ref': 500.0, 'roll_ref': 115.0}
    )
    wc.set_correction(matrix=matrix2, shift=shift2)

    v2v3idx = [k for k, n in enumerate(wc.wcs.available_frames)
               if n == 'v2v3corr']

    assert len(v2v3idx) == 1

    tp_corr = wc.wcs.pipeline[v2v3idx[0] - 1][1]

    assert isinstance(tp_corr, tpwcs.TPCorr)
    assert np.max(np.abs(tp_corr.matrix - np.identity(2))) < _ATOL
    assert np.max(np.abs(tp_corr.shift)) < _ATOL


def test_jwstgwcs_unsupported_wcs():
    from tweakwcs import tpwcs
    dummy_wcs = gwcs.wcs.WCS(models.Identity(2), 'det', 'world')
    with pytest.raises(ValueError):
        tpwcs.JWSTgWCS(dummy_wcs, {})


@pytest.mark.parametrize('tpcorr', _TPCORRS)
def test_jwstgwcs_inconsistent_ref(tpcorr, mock_jwst_wcs):
    tpwcs.TPCorr = tpcorr

    wc = tpwcs.JWSTgWCS(
        mock_jwst_wcs, {'v2_ref': 123.0, 'v3_ref': 500.0, 'roll_ref': 115.0},
    )
    wc.set_correction()

    with pytest.raises(ValueError):
        wc = tpwcs.JWSTgWCS(
            wc.wcs, {'v2_ref': 124.0, 'v3_ref': 500.0, 'roll_ref': 115.0},
        )


@pytest.mark.parametrize('tpcorr', _TPCORRS)
def test_jwstgwcs_wrong_tpcorr_type(tpcorr, mock_jwst_wcs):
    tpwcs.TPCorr = tpcorr
    wc = tpwcs.JWSTgWCS(
        mock_jwst_wcs, {'v2_ref': 123.0, 'v3_ref': 500.0, 'roll_ref': 115.0},
    )
    wc.set_correction()
    p = wc.wcs.pipeline

    np = [(v[0], V2V3ToDet()) if v[0].name == 'v2v3' else v for v in p]
    mangled_wc = gwcs.wcs.WCS(np)

    with pytest.raises(ValueError):
        wc = tpwcs.JWSTgWCS(
            mangled_wc, {'v2_ref': 123.0, 'v3_ref': 500.0, 'roll_ref': 115.0},
        )


@pytest.mark.parametrize('tpcorr', _TPCORRS)
def test_jwstgwcs_ref_angles_preserved(tpcorr, mock_jwst_wcs):
    tpwcs.TPCorr = tpcorr
    wc = tpwcs.JWSTgWCS(
        mock_jwst_wcs, {'v2_ref': 123.0, 'v3_ref': 500.0, 'roll_ref': 115.0},
    )
    assert wc.ref_angles['v2_ref'] == 123.0
    assert wc.ref_angles['v3_ref'] == 500.0
    assert wc.ref_angles['roll_ref'] == 115.0


@pytest.mark.parametrize('tpcorr', _TPCORRS)
def test_jwstgwcs_coord_transforms(tpcorr):
    tpwcs.TPCorr = tpcorr
    w = make_mock_jwst_wcs(
        v2ref=0.0, v3ref=0.0, roll=0.0, crpix=[500.0, 512.0],
        cd=[[1.0e-5, 0.0], [0.0, 1.0e-5]], crval=[12.0, 24.0]
    )
    wc = tpwcs.JWSTgWCS(w, {'v2_ref': 0.0, 'v3_ref': 0.0, 'roll_ref': 0.0})
    wc.set_correction()

    assert np.allclose(wc.det_to_world(500, 512), (12, 24), atol=_ATOL)
    assert np.allclose(wc.world_to_det(12, 24), (500, 512), atol=_ATOL)
    assert np.allclose(wc.det_to_tanp(500, 512), (0, 0), atol=_ATOL)
    assert np.allclose(wc.tanp_to_det(0, 0), (500, 512), atol=_ATOL)
    assert np.allclose(wc.world_to_tanp(12, 24), (0, 0), atol=_ATOL)
    assert np.allclose(wc.tanp_to_world(0, 0), (12, 24), atol=_ATOL)


@pytest.mark.parametrize('tpcorr', _TPCORRS)
def test_jwstgwcs_bbox(tpcorr):
    tpwcs.TPCorr = tpcorr
    w = make_mock_jwst_wcs(
        v2ref=0.0, v3ref=0.0, roll=0.0, crpix=[500.0, 512.0],
        cd=[[1.0e-5, 0.0], [0.0, 1.0e-5]], crval=[12.0, 24.0]
    )
    wc = tpwcs.JWSTgWCS(w, {'v2_ref': 0.0, 'v3_ref': 0.0, 'roll_ref': 0.0})
    wc.set_correction()

    assert np.allclose(
        wc.bounding_box,
        ((-0.5, 1024 - 0.5), (-0.5, 2048 - 0.5)),
        atol=_ATOL
    )

    wc._owcs.bounding_box = None
    assert np.allclose(
        wc.bounding_box,
        ((-0.5, 1024 - 0.5), (-0.5, 2048 - 0.5)),
        atol=_ATOL
    )

    wc._owcs.array_shape = None
    assert wc.bounding_box is None


@pytest.mark.parametrize('tpcorr', _TPCORRS)
def test_jwstgwcs_bad_pipelines(tpcorr):
    tpwcs.TPCorr = tpcorr
    p0 = make_mock_jwst_pipeline(
        v2ref=0.0, v3ref=0.0, roll=0.0, crpix=[500.0, 512.0],
        cd=[[1.0e-5, 0.0], [0.0, 1.0e-5]], crval=[12.0, 24.0]
    )

    # no pipeline or empty pipeline:
    with pytest.raises(ValueError):
        tpwcs.JWSTgWCS(None, {'v2_ref': 0.0, 'v3_ref': 0.0, 'roll_ref': 0.0})

    # fewer than 3 frames:
    w = gwcs.wcs.WCS(p0[:2])
    with pytest.raises(ValueError):
        tpwcs.JWSTgWCS(w, {'v2_ref': 0.0, 'v3_ref': 0.0, 'roll_ref': 0.0})

    # repeated (any one of the) last two frames:
    w = gwcs.wcs.WCS(p0 + [p0[-1]])
    with pytest.raises(ValueError):
        tpwcs.JWSTgWCS(w, {'v2_ref': 0.0, 'v3_ref': 0.0, 'roll_ref': 0.0})

    w = gwcs.wcs.WCS(p0 + [p0[-2]])
    with pytest.raises(ValueError):
        tpwcs.JWSTgWCS(w, {'v2_ref': 0.0, 'v3_ref': 0.0, 'roll_ref': 0.0})

    # multiple 'v2v3' frames:
    w = gwcs.wcs.WCS(p0)
    w = tpwcs.JWSTgWCS(w, {'v2_ref': 0.0, 'v3_ref': 0.0, 'roll_ref': 0.0})
    p = w.wcs.pipeline
    p.insert(1, p[1])
    w = gwcs.wcs.WCS(p)
    with pytest.raises(ValueError):
        tpwcs.JWSTgWCS(w, {'v2_ref': 0.0, 'v3_ref': 0.0, 'roll_ref': 0.0})

    # misplaced 'v2v3' frame:
    w = gwcs.wcs.WCS(p0)
    w = tpwcs.JWSTgWCS(w, {'v2_ref': 0.0, 'v3_ref': 0.0, 'roll_ref': 0.0})
    w.set_correction()
    p = w.wcs.pipeline
    del p[0]
    w = gwcs.wcs.WCS(p)
    with pytest.raises(ValueError):
        tpwcs.JWSTgWCS(w, {'v2_ref': 0.0, 'v3_ref': 0.0, 'roll_ref': 0.0})

    # multiple 'v2v3corr' frame:
    w = gwcs.wcs.WCS(p0)
    w = tpwcs.JWSTgWCS(w, {'v2_ref': 0.0, 'v3_ref': 0.0, 'roll_ref': 0.0})
    w.set_correction()
    p = w.wcs.pipeline
    p.insert(1, p[-2])
    w = gwcs.wcs.WCS(p)
    with pytest.raises(ValueError):
        tpwcs.JWSTgWCS(w, {'v2_ref': 0.0, 'v3_ref': 0.0, 'roll_ref': 0.0})

    # misplaced 'v2v3corr' frame:
    del p[-2]
    w = gwcs.wcs.WCS(p)
    with pytest.raises(ValueError):
        tpwcs.JWSTgWCS(w, {'v2_ref': 0.0, 'v3_ref': 0.0, 'roll_ref': 0.0})


def test_fitswcs_non_celestial():
    # non-celestial WCS
    w = fitswcs.WCS(naxis=3)
    with pytest.raises(ValueError):
        tpwcs.FITSWCS(w)

    # invalid WCS
    with pytest.raises(ValueError):
        tpwcs.FITSWCS(None)


def test_fitswcs_unaccounted_dist(mock_fits_wcs):
    w = copy.deepcopy(mock_fits_wcs)
    w.pix2foc = lambda x, o: x + 3
    with pytest.raises(ValueError):
        tpwcs.FITSWCS(w)

    w = copy.deepcopy(mock_fits_wcs)
    f = w.all_world2pix
    w.all_world2pix = lambda x, o: f(x, o) + 2
    with pytest.raises(ValueError):
        tpwcs.FITSWCS(w)


def test_fitswcs_1(mock_fits_wcs):
    wc = tpwcs.FITSWCS(mock_fits_wcs)
    wc.set_correction()


def test_fitswcs_coord_transforms(mock_fits_wcs):
    w = fitswcs.WCS(naxis=2)
    w.wcs.cd = build_fit_matrix(0, 1e-5)
    w.wcs.crval = [12, 24]
    w.wcs.crpix = [500, 512]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.pixel_shape = [1024, 2048]
    w.wcs.set()

    wc = tpwcs.FITSWCS(w)
    wc.set_correction()

    assert np.allclose(wc.det_to_world(499, 511), (12, 24), atol=_ATOL)
    assert np.allclose(wc.world_to_det(12, 24), (499, 511), atol=_ATOL)
    assert np.allclose(wc.det_to_tanp(499, 511), (0, 0), atol=_ATOL)
    assert np.allclose(wc.tanp_to_det(0, 0), (499, 511), atol=_ATOL)
    assert np.allclose(wc.world_to_tanp(12, 24), (0, 0), atol=1e-8)
    assert np.allclose(wc.tanp_to_world(0, 0), (12, 24), atol=_ATOL)


def test_fitswcs_bbox(mock_fits_wcs):
    w = copy.deepcopy(mock_fits_wcs)
    wc = tpwcs.FITSWCS(w)
    wc.set_correction()

    assert np.allclose(
        wc.bounding_box,
        ((-0.5, 1024 - 0.5), (-0.5, 2048 - 0.5)),
        atol=_ATOL
    )

    wc._owcs.pixel_bounds = None
    assert np.allclose(
        wc.bounding_box,
        ((-0.5, 1024 - 0.5), (-0.5, 2048 - 0.5)),
        atol=_ATOL
    )

    wc._owcs.bounding_box = None
    assert np.allclose(
        wc.bounding_box,
        ((-0.5, 1024 - 0.5), (-0.5, 2048 - 0.5)),
        atol=_ATOL
    )

    wc._owcs.array_shape = None
    assert wc.bounding_box is None
