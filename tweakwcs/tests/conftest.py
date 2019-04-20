import pytest

from astropy import wcs as fitswcs
from astropy.table import Table

from tweakwcs.linearfit import build_fit_matrix
from . helper_tpwcs import make_mock_jwst_wcs


@pytest.fixture(scope='module')
def mock_jwst_wcs():
    cd = build_fit_matrix((36, 47), 1e-5)
    w = make_mock_jwst_wcs(
        v2ref=123.0, v3ref=500.0, roll=115.0, crpix=[512.0, 512.0],
        cd=cd, crval=[82.0, 12.0]
    )
    return w


@pytest.fixture(scope='function')
def mock_fits_wcs():
    cd = build_fit_matrix((36, 47), 1e-5)
    w = fitswcs.WCS(naxis=2)
    w.wcs.cd = cd
    w.wcs.crval = [82.0, 12.0]
    w.wcs.crpix = [512.0, 512.0]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.pixel_shape = [1024, 2048]
    w.pixel_bounds = ((-0.5, 1024 - 0.5), (-0.5, 2048 - 0.5))
    w.wcs.set()
    return w


@pytest.fixture(scope='function')
def empty_imcat():
    imcat = Table([[], []], names=['x', 'y'])
    return imcat


@pytest.fixture(scope='function')
def empty_refcat():
    refcat = Table([[], []], names=['RA', 'DEC'])
    return refcat
