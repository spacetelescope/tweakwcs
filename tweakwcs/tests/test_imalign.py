"""
A module containing unit tests for the `wcsimage` module.

Licensed under a 3-clause BSD style license - see LICENSE.rst

"""

import copy
import random

import numpy as np
import pytest

from astropy.table import MaskedColumn, Table
from tweakwcs import FITSWCSCorrector
from tweakwcs.imalign import (
    NotEnoughCatalogs,
    align_wcs,
    fit_wcs,
    max_overlap_pair,
)
from tweakwcs.linearfit import build_fit_matrix


_ATOL = 1000 * np.finfo(np.array([1.0]).dtype).eps


def _assert_unchanged(cat, cat_orig):
    """Check that a (masked) input catalog was not modified in place."""
    assert cat.colnames == cat_orig.colnames  # no RA/DEC/_valid added
    assert len(cat) == len(cat_orig)
    for name in cat_orig.colnames:
        col, col_orig = cat[name], cat_orig[name]

        assert np.array_equal(np.ma.getmaskarray(col), np.ma.getmaskarray(col_orig))

        data, data_orig = np.ma.getdata(col), np.ma.getdata(col_orig)
        assert data.dtype == data_orig.dtype

        nan_ok = data_orig.dtype.kind in "fc"
        assert np.array_equal(data, data_orig, equal_nan=nan_ok)


def _generate_sources(wcs, bounding_box=None, n=12, seed=1):
    if bounding_box is None or (bounding_box := getattr(wcs, "bounding_box", None)) is None:
        bounding_box = ((0, 1023), (0, 1023))
    rng = np.random.default_rng(seed)
    x = rng.uniform(bounding_box[0][0], bounding_box[0][1], n)
    y = rng.uniform(bounding_box[1][0], bounding_box[1][1], n)
    w = wcs.pixel_to_world(x, y)
    ra, dec = w.ra.deg, w.dec.deg
    return x, y, ra, dec


def _masked_refcat(ra, dec, bad_idx, masked_ra=True):
    """Reference catalog with one *garbage* source that is masked."""
    ra = np.append(ra, 0.0 if masked_ra else np.nan)
    dec = np.append(dec, 0.0)
    mask = np.zeros(len(ra), dtype=bool)
    mask[bad_idx] = True
    return Table(
        [MaskedColumn(ra, mask=mask), MaskedColumn(dec, mask=mask)],
        names=("RA", "DEC"),
        masked=True,
    )


def test_fit_wcs_empty_cat(empty_refcat, empty_imcat, mock_fits_wcs):
    tpwcs = FITSWCSCorrector(
        mock_fits_wcs, meta={"catalog": Table([[], []], names=("x", "y")), "group_id": 1}
    )

    with pytest.raises(NotEnoughCatalogs) as e:
        align_wcs([tpwcs, tpwcs, tpwcs])
    assert e.value.args[0] == (
        "Too few input images (or groups of images) with non-empty catalogs."
    )


def test_fit_drop_empty(mock_fits_wcs):
    t0 = Table([[], []], names=("x", "y"))
    t1 = Table([[1], [3]], names=("x", "y"))
    wcscats = [
        FITSWCSCorrector(copy.deepcopy(mock_fits_wcs), meta={"catalog": t0.copy(), "group_id": 1}),
        FITSWCSCorrector(copy.deepcopy(mock_fits_wcs), meta={"catalog": t1.copy(), "group_id": 2}),
        FITSWCSCorrector(copy.deepcopy(mock_fits_wcs), meta={"catalog": t0.copy(), "group_id": 2}),
        FITSWCSCorrector(copy.deepcopy(mock_fits_wcs), meta={"catalog": t0.copy(), "group_id": 3}),
        FITSWCSCorrector(copy.deepcopy(mock_fits_wcs), meta={"catalog": t0.copy(), "group_id": 3}),
        FITSWCSCorrector(copy.deepcopy(mock_fits_wcs), meta={"catalog": t1.copy(), "group_id": 4}),
        FITSWCSCorrector(copy.deepcopy(mock_fits_wcs), meta={"catalog": t1.copy(), "group_id": 4}),
    ]

    align_wcs(wcscats, fitgeom="shift")

    status = [w.meta.get("fit_info")["status"] for w in wcscats]

    assert status[0] == "FAILED: empty source catalog"
    assert status[3] == "FAILED: empty source catalog"
    assert status[4] == "FAILED: empty source catalog"

    if status[1] == "SUCCESS":
        assert status[2] == "SUCCESS"
        assert status[5] == "REFERENCE"
        assert status[6] == "REFERENCE"

    elif status[1] == "REFERENCE":
        assert status[2] == "REFERENCE"
        assert status[5] == "SUCCESS"
        assert status[6] == "SUCCESS"

    else:
        pytest.fail(f"Unexpected fit status for wcscats[1]: {status[1]}")


def test_fit_wcs_missing_req_col_names(empty_refcat, mock_fits_wcs):
    tpwcs = FITSWCSCorrector(mock_fits_wcs)
    imcat = Table([[], []], names=("x", "weird"))
    with pytest.raises(ValueError) as e:
        fit_wcs(empty_refcat, imcat, tpwcs)
    assert e.value.args[0] == "An image catalog must contain 'x' and 'y' columns!"


def test_fit_wcs_1_image_source_empty_ref(empty_refcat, mock_fits_wcs):
    tpwcs = FITSWCSCorrector(mock_fits_wcs)
    imcat = Table([[1], [2]], names=("x", "y"))
    with pytest.raises(ValueError) as e:
        fit_wcs(empty_refcat, imcat, tpwcs)
    assert e.value.args[0] == "Reference catalog must contain at least one valid source."


def test_fit_wcs_malformed_meta(mock_fits_wcs):
    corr = FITSWCSCorrector(mock_fits_wcs)
    corr._meta = None  # bad

    x = list(range(10))
    y = [10 * random.random() for _ in range(10)]
    imcat = Table([x, y], names=("x", "y"))
    ra, dec = mock_fits_wcs.all_pix2world(x, y, 0)
    refcat = Table([ra, dec], names=("RA", "DEC"))

    with pytest.raises(AttributeError) as e:
        corr = fit_wcs(refcat, imcat, corr, fitgeom="shift")
    assert e.value.args[0] == "Unable to set/modify corrector.meta attribute."


def test_fit_wcs_unsupported_fitgeom(mock_fits_wcs):
    tpwcs = FITSWCSCorrector(mock_fits_wcs)
    x = list(range(10))
    y = [10 * random.random() for _ in range(10)]
    imcat = Table([x, y], names=("x", "y"))
    ra, dec = mock_fits_wcs.all_pix2world(x, y, 0)
    refcat = Table([ra, dec], names=("RA", "DEC"))

    with pytest.raises(ValueError) as e:
        tpwcs = fit_wcs(refcat, imcat, tpwcs, fitgeom="unsupported")
    assert (
        e.value.args[0] == "Unsupported 'fitgeom'. Valid values are: "
        "'shift', 'rshift', 'rscale', or 'general'"
    )


@pytest.mark.parametrize(
    "x, y, fitgeom",
    [
        ([1], [2], "shift"),
        ([1, 20], [1, 20], "rscale"),
        ([1, 10, 20], [1, 20, 10], "general"),
    ],
)
def test_fit_wcs_minsrc_img_ref(mock_fits_wcs, x, y, fitgeom):
    tpwcs = FITSWCSCorrector(mock_fits_wcs)
    imcat = Table([x, y], names=("x", "y"))
    ra, dec = mock_fits_wcs.all_pix2world(x, y, 0)
    refcat = Table([ra, dec], names=("RA", "DEC"))
    refcat_orig = copy.deepcopy(refcat)

    tpwcs = fit_wcs(refcat, imcat, tpwcs, fitgeom=fitgeom)

    # Check that the reference catalog was not modified in place.
    _assert_unchanged(refcat, refcat_orig)

    fi = tpwcs.meta["fit_info"]
    assert fi["status"] == "SUCCESS"
    assert np.allclose(fi["shift"], (0, 0), rtol=0, atol=1e4 * _ATOL)
    assert np.max(np.abs(fi["matrix"] - np.identity(2))) < 1e4 * _ATOL


def test_fit_wcs_less_than_minsrc(mock_fits_wcs):
    x = [1, 20]
    y = [1, 20]
    tpwcs = FITSWCSCorrector(mock_fits_wcs)
    imcat = Table([x, y], names=("x", "y"))
    ra, dec = mock_fits_wcs.all_pix2world(x, y, 0)
    refcat = Table([ra, dec], names=("RA", "DEC"))
    tpwcs = fit_wcs(refcat, imcat, tpwcs, fitgeom="general")
    assert tpwcs.meta["fit_info"]["status"] == "FAILED: not enough matches"


def test_align_wcs_tpwcs_missing_cat(mock_fits_wcs):
    tpwcs = FITSWCSCorrector(mock_fits_wcs)
    with pytest.raises(ValueError) as e:
        align_wcs(tpwcs)
    assert e.value.args[0] == "Each object in 'wcscat' must have a valid catalog."


def test_align_wcs_tpwcs_type(mock_fits_wcs):
    imcat = Table([[1, 2, 3, 4], [0, 3, 1, 5]], names=("x", "y"))

    class WrongWCSCorrector:
        def __init__(self, wcs):
            self.wcs = wcs
            self.meta = {"catalog": imcat}

    err = (
        "Input 'wcscat' must be either a single WCSCorrector-derived "
        "object or a list of WCSCorrector-derived objects."
    )

    tpwcs = WrongWCSCorrector(mock_fits_wcs)

    with pytest.raises(TypeError) as e:
        align_wcs(tpwcs)
    assert e.value.args[0] == err

    with pytest.raises(TypeError) as e:
        align_wcs([tpwcs, tpwcs])
    assert e.value.args[0] == err


@pytest.mark.parametrize(
    "shift, rot, scale, fitgeom, weighted",
    [
        ((12, -34), 0, 1, "shift", True),
        ((12, -34), 15, 1.0123, "rscale", False),
        ((12, -34), (15, 17), (1.0123, 0.9876), "general", True),
    ],
)
def test_align_wcs_simple_ref_image_general(shift, rot, scale, fitgeom, weighted, mock_fits_wcs):
    xy = 1024 * np.random.random((100, 2))
    if weighted:
        w = np.ones((100, 1))
        xy = np.hstack((xy, w))
        names = ("x", "y", "weight")
        imcat = Table(xy, names=names, masked=True)
        imcat["x"].mask[16] = True
        imcat["y"].mask[48] = True
        imcat["weight"].mask[64] = True
    else:
        names = ("x", "y")
        imcat = Table(xy, names=names, masked=False)
    imcat_orig = copy.deepcopy(imcat)

    m = build_fit_matrix(rot, scale)
    xyr = np.dot(xy[:, :2], m.T) + shift
    radec = mock_fits_wcs.wcs_pix2world(xyr, 0)
    if weighted:
        radec = np.hstack((radec, w))
        names = ("RA", "DEC", "weight")
        refcat = Table(radec, names=names, masked=True)
        refcat["RA"].mask[15] = True
        refcat["DEC"].mask[25] = True
        refcat["weight"].mask[35] = True
    else:
        names = ("RA", "DEC")
        refcat = Table(radec, names=names, masked=False)

    refcat_orig = copy.deepcopy(refcat)

    tpwcs = FITSWCSCorrector(mock_fits_wcs, meta={"catalog": imcat})

    status = align_wcs(tpwcs, refcat, fitgeom=fitgeom, match=None)

    # Check that the reference catalog was not modified in place.
    _assert_unchanged(refcat, refcat_orig)
    # Check that the image catalog was not modified in place.
    _assert_unchanged(tpwcs.meta["catalog"], imcat_orig)

    assert status
    assert tpwcs.meta["fit_info"]["status"] == "SUCCESS"
    assert tpwcs.meta["fit_info"]["fitgeom"] == fitgeom
    assert np.allclose(tpwcs.meta["fit_info"]["shift"], shift)
    assert np.allclose(tpwcs.meta["fit_info"]["matrix"], m)
    assert np.allclose(tpwcs.meta["fit_info"]["rot"], rot)
    assert tpwcs.meta["fit_info"]["proper"]
    assert np.allclose(tpwcs.meta["fit_info"]["scale"], scale)
    assert tpwcs.meta["fit_info"]["rmse"] < 1.0e-8


@pytest.mark.parametrize("masked", [False, True])
def test_align_wcs_simple_twpwcs_ref(masked, mock_fits_wcs):
    shift = (12, -34)
    rot = (15, 17)
    scale = (1.0123, 0.9876)

    m = build_fit_matrix(rot, scale)

    if masked:
        xy = 1024 * np.random.random((100, 3))
        xy[:, 2] += 1.0  # ensure weights are not close to 0
        xyr = np.hstack((np.dot(xy[:, :2], m.T) + shift, xy[:, 2:3]))
        names = ("x", "y", "weight")
    else:
        xy = 1024 * np.random.random((100, 2))
        xyr = np.dot(xy[:, :2], m.T) + shift
        names = ("x", "y")

    # Rows to mask, per column. All different rows, so that the number of rows
    # valid in *both* catalogs is 100 - 4 = 96.
    bad_im = {"x": [3], "weight": [11]}
    bad_ref = {"y": [7], "weight": [20]}

    def _make_catalog(data, bad):
        if not masked:
            return Table(data, names=names)
        data = data.copy()
        cols = []
        for k, name in enumerate(names):
            mask = np.zeros(len(data), dtype=bool)
            for i in bad.get(name, ()):
                # garbage under the mask: using it would break the exact fit
                data[i, k] = 1e6
                mask[i] = True
            cols.append(MaskedColumn(data[:, k], mask=mask))
        return Table(cols, names=names)

    imcat = _make_catalog(xy, bad_im)
    refcat = _make_catalog(xyr, bad_ref)
    imcat_orig = copy.deepcopy(imcat)
    refcat_orig = copy.deepcopy(refcat)

    tpwcs = FITSWCSCorrector(mock_fits_wcs, meta={"catalog": imcat})
    reftpwcs = FITSWCSCorrector(mock_fits_wcs, meta={"catalog": refcat})

    eff_refcat = align_wcs(tpwcs, reftpwcs, ref_tpwcs=tpwcs, fitgeom="general", match=None)

    # Check that the reference WCSCorrector's catalog was not modified in place.
    _assert_unchanged(reftpwcs.meta["catalog"], refcat_orig)
    # Check that the image WCSCorrector's catalog was not modified in place.
    _assert_unchanged(tpwcs.meta["catalog"], imcat_orig)

    assert eff_refcat
    assert tpwcs.meta["fit_info"]["status"] == "SUCCESS"
    assert tpwcs.meta["fit_info"]["fitgeom"] == "general"
    assert np.allclose(tpwcs.meta["fit_info"]["shift"], shift)
    assert np.allclose(tpwcs.meta["fit_info"]["matrix"], m)
    assert np.allclose(tpwcs.meta["fit_info"]["rot"], rot)
    assert tpwcs.meta["fit_info"]["proper"]
    assert np.allclose(tpwcs.meta["fit_info"]["scale"], scale)
    assert tpwcs.meta["fit_info"]["rmse"] < 1.0e-8

    # Only sources valid in *both* catalogs are used in the fit:
    n_expected = 100 - 4 if masked else 100
    assert tpwcs.meta["fit_info"]["fitmask"].size == n_expected

    # The returned effective reference catalog keeps all rows but flags
    # the invalid reference sources:
    assert len(eff_refcat) == 100
    expected_invalid = sorted(set(bad_ref["y"] + bad_ref["weight"])) if masked else []
    assert np.flatnonzero(~np.asarray(eff_refcat["_valid"])).tolist() == expected_invalid


def test_align_wcs_tpwcs_refcat_must_have_catalog(mock_fits_wcs):
    xy = np.array([[1, 0], [2, 3], [3, 1], [4, 5]])
    imcat = Table(xy, names=("x", "y"))
    tpwcs = FITSWCSCorrector(mock_fits_wcs, meta={"catalog": imcat})
    reftpwcs = FITSWCSCorrector(mock_fits_wcs)

    with pytest.raises(ValueError) as e:
        align_wcs(tpwcs, refcat=reftpwcs)
    assert e.value.args[0] == "Reference 'WCSCorrector' must contain a catalog."


def test_align_wcs_unknown_fitgeom(mock_fits_wcs):
    xy = np.array([[1, 0], [2, 3], [3, 1], [4, 5]])
    imcat = Table(xy, names=("x", "y"))
    tpwcs = FITSWCSCorrector(mock_fits_wcs, meta={"catalog": imcat})

    with pytest.raises(ValueError) as e:
        align_wcs(tpwcs, fitgeom="unknown")
    assert (
        e.value.args[0] == "Unsupported 'fitgeom'. Valid values are: "
        "'shift', 'rshift', 'rscale', or 'general'"
    )


def test_align_wcs_no_radec_in_refcat(mock_fits_wcs):
    xy = np.array([[1, 0], [2, 3], [3, 1], [4, 5]])
    imcat = Table(xy, names=("x", "y"))
    refcat = Table(xy, names=("x", "y"))
    tpwcs = FITSWCSCorrector(mock_fits_wcs, meta={"catalog": imcat})

    with pytest.raises(KeyError) as e:
        align_wcs(tpwcs, refcat, fitgeom="shift", match=None)
    assert e.value.args[0] == "Reference catalogs *must* contain *both* 'RA' and 'DEC' columns."


def test_align_wcs_wrong_refcat_type(mock_fits_wcs):
    xy = np.array([[1, 0], [2, 3], [3, 1], [4, 5]])
    imcat = Table(xy, names=("x", "y"))
    tpwcs = FITSWCSCorrector(mock_fits_wcs, meta={"catalog": imcat})

    with pytest.raises(TypeError) as e:
        align_wcs(tpwcs, refcat=xy, fitgeom="shift", match=None)
    assert (
        e.value.args[0] == "Unsupported 'refcat' type. Supported 'refcat' "
        "types are 'tweakwcs.correctors.WCSCorrector' and "
        "'astropy.table.Table'"
    )


@pytest.mark.parametrize("enforce", [True, False])
def test_align_wcs_refcat_from_imcat(mock_fits_wcs, enforce):
    shift = (12, -34)
    rot = (15, 17)
    scale = (1.0123, 0.9876)

    crpix = mock_fits_wcs.wcs.crpix - 1
    xy = 1024 * np.random.random((100, 2))
    m = build_fit_matrix(rot, scale)
    xyr = np.dot(xy - crpix, m) + crpix + shift
    imcat = Table(xy, names=("x", "y"))
    refcat = Table(xyr, names=("x", "y"))
    tpwcs1 = FITSWCSCorrector(mock_fits_wcs, meta={"catalog": imcat})
    tpwcs2 = copy.deepcopy(tpwcs1)
    reftpwcs = FITSWCSCorrector(mock_fits_wcs, meta={"catalog": refcat})

    input_catalogs = [reftpwcs, tpwcs1, tpwcs2]
    status = align_wcs(
        input_catalogs,
        refcat=None,
        fitgeom="general",
        match=None,
        enforce_user_order=enforce,
        expand_refcat=True,
    )
    assert status

    for cat in input_catalogs:
        if cat.meta["fit_info"]["status"] == "REFERENCE":
            if enforce:
                assert cat is reftpwcs
            continue
        assert cat.meta["fit_info"]["status"] == "SUCCESS"


def test_align_wcs_minobj(mock_fits_wcs):
    shift = (12, -34)
    rot = (15, 17)
    scale = (1.0123, 0.9876)

    crpix = mock_fits_wcs.wcs.crpix - 1
    xy = 1024 * np.random.random((100, 2))
    m = build_fit_matrix(rot, scale)
    xyr = np.dot(xy - crpix, m) + crpix + shift
    imcat = Table(xy, names=("x", "y"))
    refcat = Table(xyr, names=("x", "y"))
    tpwcs1 = FITSWCSCorrector(mock_fits_wcs, meta={"catalog": imcat})
    reftpwcs = FITSWCSCorrector(mock_fits_wcs, meta={"catalog": refcat})

    status = align_wcs(
        [reftpwcs, tpwcs1],
        refcat=None,
        fitgeom="general",
        minobj=200,
        match=None,
        enforce_user_order=True,
        expand_refcat=False,
    )
    assert status
    assert reftpwcs.meta["fit_info"]["status"] == "REFERENCE"
    assert tpwcs1.meta["fit_info"]["status"] == "FAILED: not enough matches"


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
            pts += [v1 + (v2 - v1) * np.random.random((250 - len(corners), 2)), corners]
        return np.vstack(pts)

    # reference catalog sources:
    wcsref = copy.deepcopy(mock_fits_wcs)
    xyref = 512 * get_points()
    xyref[250:500, 0] += 512
    xyref[500:750, 1] += 512
    xyref[750:, :] += 512

    radec = wcsref.wcs_pix2world(xyref, 0)
    refcat = Table(radec, names=("RA", "DEC"))

    wcsref = copy.deepcopy(mock_fits_wcs)
    refcat = Table(xyref, names=("x", "y"))
    ref_img_tpwcs = FITSWCSCorrector(wcsref, meta={"catalog": refcat, "name": "ref_img_tpwcs"})

    # single overlap catalog sources:
    wcsim1 = copy.deepcopy(mock_fits_wcs)
    wcsim1.wcs.crval += 1e-5

    xyim1 = 512 + 512 * np.vstack((np.random.random((1000, 2)), corners))
    xyim1[:250, :] = xyref[750:, :]  # overlap
    xyim1[250:500, 0] += 512
    xyim1[500:750, 1] += 512
    xyim1[750:, :] += 512

    imcat = Table(xyim1, names=("x", "y"))
    im1_tpwcs = FITSWCSCorrector(wcsim1, meta={"catalog": imcat, "name": "im1_tpwcs"})

    # non-overlaping image:
    wcsim2 = copy.deepcopy(mock_fits_wcs)
    xyim2 = xyim1.copy()
    xyim2[:, 0] += 2000.0
    imcat = Table(xyim2, names=("x", "y"))
    im2_tpwcs = FITSWCSCorrector(wcsim2, meta={"catalog": imcat, "name": "im2_tpwcs"})

    # grouped images overlap reference catalog sources:
    wcsim3 = copy.deepcopy(mock_fits_wcs)

    xyim3 = 512 * np.vstack((np.random.random((1000, 2)), corners))
    xyim3[:250, :] = xyref[250:500, :]  # overlap
    xyim3[250:750, 0] += 1024
    xyim3[750:, 0] += 512
    xyim3[500:, 1] -= 512

    imcat = Table(xyim3, names=("x", "y"))
    im3_tpwcs = FITSWCSCorrector(
        wcsim3, meta={"catalog": imcat, "group_id": "group1", "name": "im3_tpwcs"}
    )

    wcsim4 = copy.deepcopy(mock_fits_wcs)
    xyim4 = (512, -512) + 1024 * np.vstack((np.random.random((1000, 2)), corners))
    imcat = Table(xyim4, names=("x", "y"))
    im4_tpwcs = FITSWCSCorrector(
        wcsim4,
        meta={
            "catalog": imcat,
            "group_id": "group1",
            "name": "im4_tpwcs",
        },
    )

    wcsim5 = copy.deepcopy(mock_fits_wcs)
    xyim5 = (512, -512 - 1024) + 1024 * np.vstack((np.random.random((1000, 2)), corners))
    imcat = Table(xyim5, names=("x", "y"))
    im5_tpwcs = FITSWCSCorrector(
        wcsim5, meta={"catalog": imcat, "group_id": "group1", "name": "im5_tpwcs"}
    )

    # Temporarily remove im4_tpwcs from imglist due to crashes in
    # spherical_geometry.
    imglist = [
        ref_img_tpwcs,
        im1_tpwcs,
        im2_tpwcs,
        im5_tpwcs,
        im3_tpwcs,  # im4_tpwcs
    ]

    status = align_wcs(
        imglist, None, fitgeom="general", enforce_user_order=False, expand_refcat=True
    )

    assert status
    assert im1_tpwcs.meta["fit_info"]["status"] == "SUCCESS"
    assert im1_tpwcs.meta["fit_info"]["fitgeom"] == "general"
    assert im1_tpwcs.meta["fit_info"]["rmse"] < 1e8 * _ATOL
    assert np.allclose(im1_tpwcs.wcs.wcs.crval, ref_img_tpwcs.wcs.wcs.crval, rtol=0, atol=1.0e-10)


def test_max_overlap_pair():
    assert max_overlap_pair([], True) == (None, None)
    assert max_overlap_pair(["test"], True) == ("test", None)


def test_align_wcs_1im_no_ref(mock_fits_wcs):
    xy = np.array([[1, 0], [2, 3], [3, 1], [4, 5]])
    imcat = Table(xy, names=("x", "y"))
    tpwcs = FITSWCSCorrector(mock_fits_wcs, meta={"catalog": imcat})

    with pytest.raises(NotEnoughCatalogs) as e:
        align_wcs(tpwcs, refcat=None, fitgeom="shift", match=None)
    assert e.value.args[0] == (
        "Too few input images (or groups of images) with non-empty catalogs."
    )

@pytest.mark.parametrize("masked_ra", [False, True])
def test_fit_wcs_masked_refcat(mock_fits_wcs, masked_ra):
    x, y, ra, dec = _generate_sources(mock_fits_wcs)
    n = len(x)

    # reference run: all sources valid
    ref_ok = Table([ra, dec], names=("RA", "DEC"))
    im_ok = Table([x, y], names=("x", "y"))
    c1 = FITSWCSCorrector(copy.deepcopy(mock_fits_wcs))
    r1 = fit_wcs(ref_ok, im_ok, c1, fitgeom="shift")

    # Masked run. Catalogs must be 1-to-1 in fit_wcs, so both get n + 1 rows:
    #  * row n: masked in the *reference* (image has a plausible value there)
    #  * row 3: masked in the *image* (reference has a valid value there)
    # Only the remaining n - 1 rows are valid in both catalogs.
    ref_m = _masked_refcat(ra, dec, bad_idx=n, masked_ra=masked_ra)

    xm = np.append(x, 50.0)
    ym = np.append(y, 50.0)
    xm[3] = 1e6  # garbage under the mask: using it would break the fit
    xmask = np.zeros(n + 1, dtype=bool)
    xmask[3] = True
    im_m = Table([MaskedColumn(xm, mask=xmask), MaskedColumn(ym)], names=("x", "y"))

    ref_m_orig = copy.deepcopy(ref_m)
    im_m_orig = copy.deepcopy(im_m)

    c2 = FITSWCSCorrector(copy.deepcopy(mock_fits_wcs))
    r2 = fit_wcs(ref_m, im_m, c2, fitgeom="shift")

    fi1 = r1.meta["fit_info"]
    fi2 = r2.meta["fit_info"]

    assert fi1["status"] == "SUCCESS"
    assert fi2["status"] == "SUCCESS"
    assert np.allclose(fi1["shift"], fi2["shift"], rtol=0, atol=1e4 * _ATOL)

    # 'fitmask' has one element per source valid in *both* catalogs:
    assert fi1["fitmask"].size == n
    assert fi2["fitmask"].size == n - 1

    # input catalogs were not modified in place:
    _assert_unchanged(ref_m, ref_m_orig)
    _assert_unchanged(im_m, im_m_orig)


@pytest.mark.parametrize("use_corrector_ref", [False, True])
def test_align_wcs_masked_refcat(mock_fits_wcs, use_corrector_ref):
    x, y, ra, dec = _generate_sources(mock_fits_wcs)
    imcat = Table([x + 0.5, y - 0.3], names=("x", "y"))

    def _align(refcat):
        corr = FITSWCSCorrector(copy.deepcopy(mock_fits_wcs), meta={"catalog": imcat.copy()})
        eff = align_wcs(corr, refcat=refcat, fitgeom="shift")
        return corr.meta["fit_info"], eff

    if use_corrector_ref:
        def _ref(masked):
            xx, yy = (np.append(x, 50.0), np.append(y, 50.0)) if masked else (x, y)
            mask = np.zeros(len(xx), dtype=bool)
            mask[-1] = masked
            cat = Table(
                [MaskedColumn(xx, mask=mask), MaskedColumn(yy, mask=mask)],
                names=("x", "y"),
            )
            return FITSWCSCorrector(copy.deepcopy(mock_fits_wcs), meta={"catalog": cat})

        ref_ok, ref_m = _ref(False), _ref(True)
        input_cat = lambda r: r.meta["catalog"]  # noqa: E731
    else:
        ref_ok = Table([ra, dec], names=("RA", "DEC"))
        ref_m = _masked_refcat(ra, dec, bad_idx=len(ra))
        input_cat = lambda r: r  # noqa: E731

    ref_m_orig = copy.deepcopy(input_cat(ref_m))

    info_ok, _ = _align(ref_ok)
    info_m, eff_m = _align(ref_m)

    assert info_ok["status"] == "SUCCESS"
    assert info_m["status"] == "SUCCESS"
    assert np.allclose(info_ok["shift"], info_m["shift"])

    # masked source is kept in the effective catalog but flagged invalid:
    assert len(eff_m) == len(x) + 1
    assert not eff_m["_valid"][-1]
    assert eff_m["_valid"][:-1].all()

    # input reference catalog was not modified:
    _assert_unchanged(input_cat(ref_m), ref_m_orig)


def test_align_wcs_expand_refcat_skips_invalid_sources(mock_fits_wcs):
    x, y, ra, dec = _generate_sources(mock_fits_wcs)
    mask = np.zeros(len(x), dtype=bool)
    # these image sources are invalid and should not be matched:
    mask[[2, 7]] = True
    imcat = Table([MaskedColumn(x, mask=mask), MaskedColumn(y, mask=mask)], names=("x", "y"))
    refcat = Table([ra, dec], names=("RA", "DEC"))
    corr = FITSWCSCorrector(copy.deepcopy(mock_fits_wcs), meta={"catalog": imcat})

    eff = align_wcs(corr, refcat=refcat, fitgeom="shift", match=None, expand_refcat=True)

    assert corr.meta["fit_info"]["status"] == "SUCCESS"
    # check that the two invalid sources were not appended (12 rows instead of 14):
    assert len(eff) == len(refcat)
    assert np.all(eff["_valid"])


def test_align_wcs_all_invalid_image_is_not_reference(mock_fits_wcs):
    x, y, _, _ = _generate_sources(mock_fits_wcs)
    all_masked = np.ones(len(x), dtype=bool)
    bad = Table([MaskedColumn(x, mask=all_masked), MaskedColumn(y, mask=all_masked)],
                names=("x", "y"))
    good1 = Table([x, y], names=("x", "y"))
    good2 = Table([x + 0.5, y - 0.3], names=("x", "y"))
    c_bad, c1, c2 = (
        FITSWCSCorrector(copy.deepcopy(mock_fits_wcs), meta={"catalog": c})
        for c in (bad, good1, good2)
    )

    # the all-masked image is first, so it would normally become the reference:
    align_wcs([c_bad, c1, c2], fitgeom="shift", match=None)

    assert c_bad.meta["fit_info"]["status"] == "FAILED: no valid sources"
    assert c1.meta["fit_info"]["status"] == "REFERENCE"
    assert c2.meta["fit_info"]["status"] == "SUCCESS"
