"""
A module containing unit tests for the `wcsimage` module.

Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
import copy
import pytest
import random

import numpy as np

from astropy.table import Table

from tweakwcs.linearfit import build_fit_matrix
from tweakwcs.imalign import fit_wcs, align_wcs, max_overlap_pair
from tweakwcs import FITSWCS


_ATOL = 1000 * np.finfo(np.array([1.]).dtype).eps


def test_fit_wcs_empty_cat(empty_refcat, empty_imcat, mock_fits_wcs):
    tpwcs = FITSWCS(
        mock_fits_wcs,
        meta={'catalog': Table([[], []], names=('x', 'y')), 'group_id': 1}
    )

    with pytest.raises(ValueError) as e:
        align_wcs([tpwcs, tpwcs, tpwcs])
    assert e.value.args[0] == ("Too few input images (or groups of images) "
                               "with non-empty catalogs.")


def test_fit_drop_empty(mock_fits_wcs):
    t0 = Table([[], []], names=('x', 'y'))
    t1 = Table([[1], [3]], names=('x', 'y'))
    wcscats = [
        FITSWCS(
            copy.deepcopy(mock_fits_wcs),
            meta={'catalog': t0.copy(), 'group_id': 1}
        ),
        FITSWCS(
            copy.deepcopy(mock_fits_wcs),
            meta={'catalog': t1.copy(), 'group_id': 2}
        ),
        FITSWCS(
            copy.deepcopy(mock_fits_wcs),
            meta={'catalog': t0.copy(), 'group_id': 2}
        ),
        FITSWCS(
            copy.deepcopy(mock_fits_wcs),
            meta={'catalog': t0.copy(), 'group_id': 3}
        ),
        FITSWCS(
            copy.deepcopy(mock_fits_wcs),
            meta={'catalog': t0.copy(), 'group_id': 3}
        ),
        FITSWCS(
            copy.deepcopy(mock_fits_wcs),
            meta={'catalog': t1.copy(), 'group_id': 4}
        ),
        FITSWCS(
            copy.deepcopy(mock_fits_wcs),
            meta={'catalog': t1.copy(), 'group_id': 4}
        )
    ]

    align_wcs(wcscats, fitgeom='shift')

    status = [w.meta.get('fit_info')['status'] for w in wcscats]

    assert status[0] == 'FAILED: empty source catalog'
    assert status[3] == 'FAILED: empty source catalog'
    assert status[4] == 'FAILED: empty source catalog'

    if status[1] == 'SUCCESS':
        assert status[2] == 'SUCCESS'
        assert status[5] == 'REFERENCE'
        assert status[6] == 'REFERENCE'

    elif status[1] == 'REFERENCE':
        assert status[2] == 'REFERENCE'
        assert status[5] == 'SUCCESS'
        assert status[6] == 'SUCCESS'

    else:
        assert False


def test_fit_wcs_missing_req_col_names(empty_refcat, mock_fits_wcs):
    tpwcs = FITSWCS(mock_fits_wcs)
    imcat = Table([[], []], names=('x', 'weird'))
    with pytest.raises(ValueError) as e:
        fit_wcs(empty_refcat, imcat, tpwcs)
    assert (e.value.args[0] == "An image catalog must contain 'x' "
            "and 'y' columns!")


def test_fit_wcs_1_image_source_empty_ref(empty_refcat, mock_fits_wcs):
    tpwcs = FITSWCS(mock_fits_wcs)
    imcat = Table([[1], [2]], names=('x', 'y'))
    with pytest.raises(ValueError) as e:
        fit_wcs(empty_refcat, imcat, tpwcs)
    assert (e.value.args[0] == "Reference catalog must contain at "
            "least one source.")


def test_fit_wcs_malformed_meta(mock_fits_wcs):
    tpwcs = FITSWCS(mock_fits_wcs)
    tpwcs._meta = None  # bad

    x = list(range(10))
    y = [10 * random.random() for _ in range(10)]
    imcat = Table([x, y], names=('x', 'y'))
    ra, dec = mock_fits_wcs.all_pix2world(x, y, 0)
    refcat = Table([ra, dec], names=('RA', 'DEC'))

    with pytest.raises(AttributeError) as e:
        tpwcs = fit_wcs(refcat, imcat, tpwcs, fitgeom='shift')
    assert e.value.args[0] == "Unable to set/modify tpwcs.meta attribute."


def test_fit_wcs_unsupported_fitgeom(mock_fits_wcs):
    tpwcs = FITSWCS(mock_fits_wcs)
    x = list(range(10))
    y = [10 * random.random() for _ in range(10)]
    imcat = Table([x, y], names=('x', 'y'))
    ra, dec = mock_fits_wcs.all_pix2world(x, y, 0)
    refcat = Table([ra, dec], names=('RA', 'DEC'))

    with pytest.raises(ValueError) as e:
        tpwcs = fit_wcs(refcat, imcat, tpwcs, fitgeom='unsupported')
    assert (e.value.args[0] == "Unsupported 'fitgeom'. Valid values are: "
            "'shift', 'rscale', or 'general'")


@pytest.mark.parametrize('x, y, fitgeom', [
    ([1], [2], 'shift'),
    ([1, 20], [1, 20], 'rscale'),
    ([1, 10, 20], [1, 20, 10], 'general'),
])
def test_fit_wcs_minsrc_img_ref(mock_fits_wcs, x, y, fitgeom):
    tpwcs = FITSWCS(mock_fits_wcs)
    imcat = Table([x, y], names=('x', 'y'))
    ra, dec = mock_fits_wcs.all_pix2world(x, y, 0)
    refcat = Table([ra, dec], names=('RA', 'DEC'))

    tpwcs = fit_wcs(refcat, imcat, tpwcs, fitgeom=fitgeom)

    fi = tpwcs.meta['fit_info']
    assert fi['status'] == 'SUCCESS'
    assert np.allclose(fi['shift'], (0, 0), rtol=0, atol=1e4 * _ATOL)
    assert np.max(np.abs(fi['matrix'] - np.identity(2))) < 1e4 * _ATOL


def test_fit_wcs_less_than_minsrc(mock_fits_wcs):
    x = [1, 20]
    y = [1, 20]
    tpwcs = FITSWCS(mock_fits_wcs)
    imcat = Table([x, y], names=('x', 'y'))
    ra, dec = mock_fits_wcs.all_pix2world(x, y, 0)
    refcat = Table([ra, dec], names=('RA', 'DEC'))
    tpwcs = fit_wcs(refcat, imcat, tpwcs, fitgeom='general')
    assert tpwcs.meta['fit_info']['status'] == 'FAILED: not enough matches'


def test_align_wcs_tpwcs_missing_cat(mock_fits_wcs):
    tpwcs = FITSWCS(mock_fits_wcs)
    with pytest.raises(ValueError) as e:
        align_wcs(tpwcs)
    assert (e.value.args[0] == "Each object in 'wcscat' must have a valid "
            "catalog.")


def test_align_wcs_tpwcs_type(mock_fits_wcs):
    imcat = Table([[1, 2, 3, 4], [0, 3, 1, 5]], names=('x', 'y'))

    class WrongTPWCS:
        def __init__(self, wcs):
            self.wcs = wcs
            self.meta = {'catalog': imcat}

    err = ("Input 'wcscat' must be either a single TPWCS-derived "
           " object or a list of TPWCS-derived objects.")

    tpwcs = WrongTPWCS(mock_fits_wcs)

    with pytest.raises(TypeError) as e:
        align_wcs(tpwcs)
    assert (e.value.args[0] == err)

    with pytest.raises(TypeError) as e:
        align_wcs([tpwcs, tpwcs])
    assert (e.value.args[0] == err)


@pytest.mark.parametrize('shift, rot, scale, fitgeom, weighted', [
    ((12, -34), 0, 1, 'shift', True),
    ((12, -34), 15, 1.0123, 'rscale', False),
    ((12, -34), (15, 17), (1.0123, 0.9876), 'general', True),
])
def test_align_wcs_simple_ref_image_general(shift, rot, scale, fitgeom,
                                            weighted, mock_fits_wcs):
    xy = 1024 * np.random.random((100, 2))
    if weighted:
        w = np.ones((100, 1))
        xy = np.hstack((xy, w))
        names = ('x', 'y', 'weight')
    else:
        names = ('x', 'y')
    m = build_fit_matrix(rot, scale)
    xyr = np.dot(xy[:, :2], m.T) + shift
    imcat = Table(xy, names=names)
    radec = mock_fits_wcs.wcs_pix2world(xyr, 0)
    if weighted:
        radec = np.hstack((radec, w))
        names = ('RA', 'DEC', 'weight')
    else:
        names = ('RA', 'DEC')
    refcat = Table(radec, names=names)
    tpwcs = FITSWCS(mock_fits_wcs, meta={'catalog': imcat})
    status = align_wcs(tpwcs, refcat, fitgeom=fitgeom, match=None)

    assert status
    assert tpwcs.meta['fit_info']['status'] == 'SUCCESS'
    assert tpwcs.meta['fit_info']['fitgeom'] == fitgeom
    assert np.allclose(tpwcs.meta['fit_info']['shift'], shift)
    assert np.allclose(tpwcs.meta['fit_info']['matrix'], m)
    assert np.allclose(tpwcs.meta['fit_info']['rot'], rot)
    assert tpwcs.meta['fit_info']['proper']
    assert np.allclose(tpwcs.meta['fit_info']['scale'], scale)
    assert tpwcs.meta['fit_info']['rmse'] < 1.0e-8


def test_align_wcs_simple_twpwcs_ref(mock_fits_wcs):
    shift = (12, -34)
    rot = (15, 17)
    scale = (1.0123, 0.9876)

    xy = 1024 * np.random.random((100, 2))
    m = build_fit_matrix(rot, scale)
    xyr = np.dot(xy, m.T) + shift
    imcat = Table(xy, names=('x', 'y'))
    refcat = Table(xyr, names=('x', 'y'))
    tpwcs = FITSWCS(mock_fits_wcs, meta={'catalog': imcat})
    reftpwcs = FITSWCS(mock_fits_wcs, meta={'catalog': refcat})
    status = align_wcs(tpwcs, reftpwcs, ref_tpwcs=tpwcs,
                       fitgeom='general', match=None)

    assert status
    assert tpwcs.meta['fit_info']['status'] == 'SUCCESS'
    assert tpwcs.meta['fit_info']['fitgeom'] == 'general'
    assert np.allclose(tpwcs.meta['fit_info']['shift'], shift)
    assert np.allclose(tpwcs.meta['fit_info']['matrix'], m)
    assert np.allclose(tpwcs.meta['fit_info']['rot'], rot)
    assert tpwcs.meta['fit_info']['proper']
    assert np.allclose(tpwcs.meta['fit_info']['scale'], scale)
    assert tpwcs.meta['fit_info']['rmse'] < 1.0e-8


def test_align_wcs_tpwcs_refcat_must_have_catalog(mock_fits_wcs):
    xy = np.array([[1, 0], [2, 3], [3, 1], [4, 5]])
    imcat = Table(xy, names=('x', 'y'))
    tpwcs = FITSWCS(mock_fits_wcs, meta={'catalog': imcat})
    reftpwcs = FITSWCS(mock_fits_wcs)

    with pytest.raises(ValueError) as e:
        align_wcs(tpwcs, refcat=reftpwcs)
    assert (e.value.args[0] == "Reference 'TPWCS' must contain a catalog.")


def test_align_wcs_unknown_fitgeom(mock_fits_wcs):
    xy = np.array([[1, 0], [2, 3], [3, 1], [4, 5]])
    imcat = Table(xy, names=('x', 'y'))
    tpwcs = FITSWCS(mock_fits_wcs, meta={'catalog': imcat})

    with pytest.raises(ValueError) as e:
        align_wcs(tpwcs, fitgeom='unknown')
    assert (e.value.args[0] == "Unsupported 'fitgeom'. Valid values are: "
            "'shift', 'rscale', or 'general'")


def test_align_wcs_no_radec_in_refcat(mock_fits_wcs):
    xy = np.array([[1, 0], [2, 3], [3, 1], [4, 5]])
    imcat = Table(xy, names=('x', 'y'))
    refcat = Table(xy, names=('x', 'y'))
    tpwcs = FITSWCS(mock_fits_wcs, meta={'catalog': imcat})

    with pytest.raises(KeyError) as e:
        align_wcs(tpwcs, refcat, fitgeom='shift', match=None)
    assert (e.value.args[0] == "Reference catalogs *must* contain *both* 'RA' "
            "and 'DEC' columns.")


def test_align_wcs_wrong_refcat_type(mock_fits_wcs):
    xy = np.array([[1, 0], [2, 3], [3, 1], [4, 5]])
    imcat = Table(xy, names=('x', 'y'))
    tpwcs = FITSWCS(mock_fits_wcs, meta={'catalog': imcat})

    with pytest.raises(TypeError) as e:
        align_wcs(tpwcs, refcat=xy, fitgeom='shift', match=None)
    assert (e.value.args[0] == "Unsupported 'refcat' type. Supported 'refcat' "
            "types are 'tweakwcs.tpwcs.TPWCS' and 'astropy.table.Table'")


@pytest.mark.parametrize('enforce', [True, False])
def test_align_wcs_refcat_from_imcat(mock_fits_wcs, enforce):
    shift = (12, -34)
    rot = (15, 17)
    scale = (1.0123, 0.9876)

    crpix = mock_fits_wcs.wcs.crpix - 1
    xy = 1024 * np.random.random((100, 2))
    m = build_fit_matrix(rot, scale)
    xyr = np.dot(xy - crpix, m) + crpix + shift
    imcat = Table(xy, names=('x', 'y'))
    refcat = Table(xyr, names=('x', 'y'))
    tpwcs1 = FITSWCS(mock_fits_wcs, meta={'catalog': imcat})
    tpwcs2 = copy.deepcopy(tpwcs1)
    reftpwcs = FITSWCS(mock_fits_wcs, meta={'catalog': refcat})

    input_catalogs = [reftpwcs, tpwcs1, tpwcs2]
    status = align_wcs(
        input_catalogs, refcat=None, fitgeom='general',
        match=None, enforce_user_order=enforce, expand_refcat=True
    )
    assert status

    for cat in input_catalogs:
        if cat.meta['fit_info']['status'] == 'REFERENCE':
            if enforce:
                assert cat is reftpwcs
            continue
        assert cat.meta['fit_info']['status'] == 'SUCCESS'


def test_multi_image_set(mock_fits_wcs):
    np.random.seed(1)
    v1 = 1e10 * np.finfo(np.double).eps
    v2 = 1 - v1
    corners = np.array([[v1, v1], [v1, v2], [v2, v2], [v2, v1]])
    n = 1

    def get_points():
        nonlocal n, v1, v2
        pts = []
        for _ in range(4):
            v1 *= n
            v2 = 1 - v1
            n += 1
            corners = np.array([[v1, v1], [v1, v2], [v2, v2], [v2, v1]])
            pts += [
                v1 + (v2 - v1) * np.random.random((250 - len(corners), 2)),
                corners
            ]
        return np.vstack(pts)

    # reference catalog sources:
    wcsref = copy.deepcopy(mock_fits_wcs)
    xyref = 512 * get_points()
    xyref[250:500, 0] += 512
    xyref[500:750, 1] += 512
    xyref[750:, :] += 512

    radec = wcsref.wcs_pix2world(xyref, 0)
    refcat = Table(radec, names=('RA', 'DEC'))

    wcsref = copy.deepcopy(mock_fits_wcs)
    refcat = Table(xyref, names=('x', 'y'))
    ref_img_tpwcs = FITSWCS(wcsref, meta={'catalog': refcat,
                                          'name': 'ref_img_tpwcs'})

    # single overlap catalog sources:
    wcsim1 = copy.deepcopy(mock_fits_wcs)
    wcsim1.wcs.crval += 1e-5

    xyim1 = 512 + 512 * np.vstack((np.random.random((1000, 2)), corners))
    xyim1[:250, :] = xyref[750:, :]  # overlap
    xyim1[250:500, 0] += 512
    xyim1[500:750, 1] += 512
    xyim1[750:, :] += 512

    imcat = Table(xyim1, names=('x', 'y'))
    im1_tpwcs = FITSWCS(wcsim1, meta={'catalog': imcat, 'name': 'im1_tpwcs'})

    # non-overlaping image:
    wcsim2 = copy.deepcopy(mock_fits_wcs)
    xyim2 = xyim1.copy()
    xyim2[:, 0] += 2000.0
    imcat = Table(xyim2, names=('x', 'y'))
    im2_tpwcs = FITSWCS(wcsim2, meta={'catalog': imcat, 'name': 'im2_tpwcs'})

    # grouped images overlap reference catalog sources:
    wcsim3 = copy.deepcopy(mock_fits_wcs)

    xyim3 = 512 * np.vstack((np.random.random((1000, 2)), corners))
    xyim3[:250, :] = xyref[250:500, :]  # overlap
    xyim3[250:750, 0] += 1024
    xyim3[750:, 0] += 512
    xyim3[500:, 1] -= 512

    imcat = Table(xyim3, names=('x', 'y'))
    im3_tpwcs = FITSWCS(wcsim3, meta={
        'catalog': imcat, 'group_id': 'group1', 'name': 'im3_tpwcs'
    })

    wcsim4 = copy.deepcopy(mock_fits_wcs)
    xyim4 = (512, -512) + 1024 * np.vstack((np.random.random((1000, 2)),
                                            corners))
    imcat = Table(xyim4, names=('x', 'y'))
    im4_tpwcs = FITSWCS(wcsim4, meta={  # noqa: F841
        'catalog': imcat, 'group_id': 'group1', 'name': 'im4_tpwcs'
    })

    wcsim5 = copy.deepcopy(mock_fits_wcs)
    xyim5 = (512, -512 - 1024) + 1024 * np.vstack((np.random.random((1000, 2)),
                                                   corners))
    imcat = Table(xyim5, names=('x', 'y'))
    im5_tpwcs = FITSWCS(wcsim5, meta={
        'catalog': imcat, 'group_id': 'group1', 'name': 'im5_tpwcs'
    })

    # Temporarily remove im4_tpwcs from imglist due to crashes in
    # spherical_geometry.
    imglist = [
        ref_img_tpwcs, im1_tpwcs, im2_tpwcs, im5_tpwcs, im3_tpwcs,  # im4_tpwcs
    ]

    status = align_wcs(imglist, None, fitgeom='general',
                       enforce_user_order=False, expand_refcat=True)

    assert status
    assert im1_tpwcs.meta['fit_info']['status'] == 'SUCCESS'
    assert im1_tpwcs.meta['fit_info']['fitgeom'] == 'general'
    assert im1_tpwcs.meta['fit_info']['rmse'] < 1e8 * _ATOL
    assert np.allclose(im1_tpwcs.wcs.wcs.crval, ref_img_tpwcs.wcs.wcs.crval,
                       rtol=0, atol=1.0e-10)


def test_max_overlap_pair():
    assert max_overlap_pair([], True) == (None, None)
    assert max_overlap_pair(['test'], True) == ('test', None)


def test_align_wcs_1im_no_ref(mock_fits_wcs):
    xy = np.array([[1, 0], [2, 3], [3, 1], [4, 5]])
    imcat = Table(xy, names=('x', 'y'))
    tpwcs = FITSWCS(mock_fits_wcs, meta={'catalog': imcat})

    with pytest.raises(ValueError) as e:
        align_wcs(tpwcs, refcat=None, fitgeom='shift', match=None)
    assert e.value.args[0] == ("Too few input images (or groups of images) "
                               "with non-empty catalogs.")
