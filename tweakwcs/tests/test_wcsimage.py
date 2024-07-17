"""
A module containing unit tests for the `wcsimage` module.

Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
import copy
import pytest
import numpy as np
from astropy.table import Table, Column
from astropy.io import fits
from astropy import wcs
from tweakwcs import XYXYMatch, FITSWCSCorrector
from tweakwcs.wcsimage import (convex_hull, RefCatalog, WCSImageCatalog,
                               WCSGroupCatalog, _is_int)
from astropy.utils.data import get_pkg_data_filename


_ATOL = 100 * np.finfo(np.array([1.]).dtype).eps


def _same_spherical_polygons(p1, p2):
    p1 = np.array(list(p1.points), dtype=float)
    p2 = np.array(list(p2.points), dtype=float)

    if p1.shape != p2.shape:
        return False

    idx = np.argmin(np.linalg.norm(p1[0] - p2, axis=1))

    if idx > 0:
        p1 = p1[:-1]
        p2 = p2[:-1]
        p2 = np.roll(p2, -idx, axis=0)

    return np.allclose(p1, p2, atol=_ATOL, rtol=0)


@pytest.fixture
def rect_cat(scope='function'):
    x = np.array([0.0, 0.0, 10.0, 10.0, 0.0]) - 5
    y = np.array([0.0, 10.0, 10.0, 0.0, 0.0]) - 5
    imcat = Table([x, y], names=('x', 'y'))
    return imcat


@pytest.fixture(scope='function')
def rect_imcat(mock_fits_wcs):
    x = np.array([0.0, 0.0, 10.0, 10.0, 0.0]) - 5
    y = np.array([0.0, 10.0, 10.0, 0.0, 0.0]) - 5
    imcat = Table([x, y], names=('x', 'y'))
    corr = FITSWCSCorrector(mock_fits_wcs)
    w = WCSImageCatalog(imcat, corr)
    return w


@pytest.mark.parametrize('val, expected', [
    (1, True), (1.0, False), ('1', False), (True, False),
    (np.bool_(1), False), (np.float16(1.0), False), (np.int16(1), True),
])
def test_is_int(val, expected):
    assert _is_int(val) is expected


def test_wcsimcat_wrong_tpwcs_type(mock_fits_wcs):
    imcat = Table([[], []], names=('x', 'y'))
    with pytest.raises(TypeError) as e:
        WCSImageCatalog(imcat, None)
    assert (e.value.args[0] == "Unsupported 'corrector' type. 'corrector' "
            "must be a subtype of WCSCorrector.")


def test_wcsimcat_wrong_fit_status(mock_fits_wcs, rect_imcat):
    rect_imcat.fit_status = 'spectacular'
    assert rect_imcat.fit_status == 'spectacular'


def test_wcsimcat_set_none_catalog(mock_fits_wcs, rect_imcat):
    rect_imcat.catalog = None
    assert rect_imcat.catalog is None


def test_wcsimcat_wcs_transforms_roundtrip(mock_fits_wcs):
    x = np.arange(5)
    y = np.random.permutation(5)
    imcat = Table([x, y], names=('x', 'y'))
    corr = FITSWCSCorrector(mock_fits_wcs)
    w = WCSImageCatalog(imcat, corr)

    assert np.allclose(
        w.world_to_det(*w.det_to_world(x, y)), (x, y), rtol=0, atol=1.0e-5
    )

    assert np.allclose(
        w.tanp_to_det(*w.det_to_tanp(x, y)), (x, y), rtol=0, atol=1.0e-5
    )

    assert np.allclose(
        w.world_to_tanp(*w.tanp_to_world(1.0e-5 * x, 1.0e-5 * y)),
        (1.0e-5 * x, 1.0e-5 * y),
        rtol=0, atol=1.0e-5
    )


def test_wcsimcat_intersections(mock_fits_wcs, rect_imcat):
    pts1 = list(rect_imcat.polygon.points)[0]
    pts2 = list(rect_imcat.intersection(rect_imcat.polygon).points)[0]
    for pt1 in pts1:
        assert any(np.allclose(pt1, pt2) for pt2 in pts2)

    pts2 = list(rect_imcat.intersection(rect_imcat).points)[0]
    for pt1 in pts1:
        assert any(np.allclose(pt1, pt2) for pt2 in pts2)

    assert np.allclose(
        rect_imcat.intersection_area(rect_imcat), 2.9904967391303217e-12,
        atol=0.0, rtol=5.0e-4
    )


def test_wcsimcat_guarded_intersection_area(mock_fits_wcs, rect_imcat):
    assert np.allclose(
        rect_imcat._guarded_intersection_area(rect_imcat)[0],
        2.9904967391303217e-12, atol=0.0, rtol=5.0e-4
    )


def test_wcsimcat_no_wcs_bb(mock_fits_wcs, rect_cat):
    corr = FITSWCSCorrector(mock_fits_wcs)
    corr._owcs.pixel_bounds = None
    corr._owcs.pixel_shape = None

    assert corr.bounding_box is None

    WCSImageCatalog(rect_cat, corr)


def test_wcsimcat_calc_chip_bounding_polygon_custom_stepsize(mock_fits_wcs,
                                                             rect_imcat):
    rect_imcat._calc_chip_bounding_polygon(stepsize=None)
    pts1 = list(rect_imcat.polygon.points)[0]

    rect_imcat._calc_chip_bounding_polygon(stepsize=2)
    pts2 = list(rect_imcat.polygon.points)[0]

    for pt1 in pts1:
        assert any(np.allclose(pt1, pt2) for pt2 in pts2)

    assert rect_imcat._calc_chip_bounding_polygon


def test_wcsimcat_calc_cat_convex_hull_no_catalog(mock_fits_wcs, rect_imcat):
    rect_imcat.catalog = None
    assert rect_imcat._calc_cat_convex_hull() is None


def test_wcsimcat_calc_cat_convex_hull_adjacent():
    h = fits.Header.fromfile(get_pkg_data_filename('data/wfc3_uvis1.hdr'))
    w = wcs.WCS(h)
    corr = FITSWCSCorrector(w)
    cat = Table.read(
        get_pkg_data_filename('data/convex_hull_proximity.cat'),
        format='ascii.tab',
        delimiter='\t',
        names=['x', 'y']
    )
    w = WCSImageCatalog(cat, corr)
    assert len(list(w.polygon.points)[0]) == 14


def test_wcsimcat_bb_radec(mock_fits_wcs, rect_imcat):
    ra1, dec1 = rect_imcat.det_to_world(rect_imcat.catalog['x'],
                                        rect_imcat.catalog['y'])
    ra2, dec2 = rect_imcat.bb_radec
    assert np.allclose(ra1, ra2[::-1])
    assert np.allclose(dec1, dec2[::-1])


def test_wcsgroupcat_init(mock_fits_wcs, rect_imcat):
    g = WCSGroupCatalog(rect_imcat, 'name1')
    assert g.name == 'name1'

    for im in g:
        assert im.name == 'Unknown'

    g.name = 'name2'
    assert g.name == 'name2'
    assert g[0] is rect_imcat

    assert len(g) == 1

    # input list is empty:
    with pytest.raises(ValueError) as e:
        WCSGroupCatalog([])
    assert e.value.args[0] == "List of images cannot be empty."

    # wrong type for the WCSImageCatalog in a list
    with pytest.raises(TypeError) as e:
        WCSGroupCatalog([1])
    assert (e.value.args[0] == "Each element of the 'images' parameter "
            "must be an 'WCSImageCatalog' object.")

    # wrong type for the input catalog: it is not a WCSImageCatalog
    with pytest.raises(TypeError) as e:
        WCSGroupCatalog(1)
    assert (e.value.args[0] == "Parameter 'images' must be either a single "
            "'WCSImageCatalog' object or a list of 'WCSImageCatalog' objects")

    # input WCSImageCatalog with a missing catalog:
    with pytest.raises(ValueError) as e:
        rect_imcat.catalog = None
        WCSGroupCatalog(rect_imcat)
    assert (e.value.args[0] == "Each input WCS image catalog must have a "
            "valid catalog.")

    # input WCSImageCatalog  with a missing catalog in the input list:
    with pytest.raises(ValueError) as e:
        rect_imcat.catalog = None
        WCSGroupCatalog([rect_imcat])
    assert (e.value.args[0] == "Each input WCS image catalog must have a "
            "valid catalog.")


def test_wcsgroupcat_intersections(mock_fits_wcs, rect_imcat):
    g = WCSGroupCatalog(rect_imcat)

    pts1 = list(g.polygon.points)[0]
    pts2 = list(g.intersection(g.polygon).points)[0]
    for pt1 in pts1:
        assert any(np.allclose(pt1, pt2) for pt2 in pts2)

    pts2 = list(g.intersection(g).points)[0]
    for pt1 in pts1:
        assert any(np.allclose(pt1, pt2) for pt2 in pts2)

    assert np.allclose(
        g.intersection_area(g), 2.9904967391303217e-12, atol=0.0, rtol=5.0e-4
    )


def test_wcsgroupcat_guarded_intersection_area(mock_fits_wcs, rect_imcat):
    g = WCSGroupCatalog(rect_imcat)
    assert np.allclose(
        g._guarded_intersection_area(g)[0], 2.9904967391303217e-12,
        atol=0.0, rtol=5.0e-4
    )


def test_wcsgroupcat_update_bb_no_images(mock_fits_wcs, rect_imcat):
    g = WCSGroupCatalog(rect_imcat)
    g._images = []
    g.update_bounding_polygon()
    assert len(g.polygon) == 0


def test_wcsgroupcat_empty_cat(mock_fits_wcs, rect_imcat):
    imcat = Table([[], [], [], []], names=('x', 'y', 'TPx', 'TPy'))
    corr = FITSWCSCorrector(mock_fits_wcs)

    ra, dec = mock_fits_wcs.all_pix2world(rect_imcat.catalog['x'],
                                          rect_imcat.catalog['y'], 0)
    refcat = Table([ra, dec], names=('RA', 'DEC'))
    ref = RefCatalog(refcat)

    w = WCSImageCatalog(imcat, corr)
    ref.calc_tanp_xy(tanplane_wcs=rect_imcat.corrector)
    g = WCSGroupCatalog([w])
    g.calc_tanp_xy(tanplane_wcs=rect_imcat.corrector)

    nmatches, *_ = g.match2ref(ref, match=XYXYMatch())
    assert nmatches == 0


def test_wcsgroupcat_create_group_catalog(mock_fits_wcs, rect_imcat):
    w1 = copy.deepcopy(rect_imcat)
    w2 = copy.deepcopy(rect_imcat)
    g = WCSGroupCatalog([w1, w2])

    # catalogs with name set to None:
    names = []
    for im in g:
        names.append(im.name)
        im._name = None
    assert len(g.create_group_catalog()) == 2 * len(rect_imcat.catalog)
    for im, name in zip(g, names):
        im.name = name

    # Mixed catalogs: one has weights and another does not:
    g[0].catalog.add_column(Column(np.ones(5)), name='weight')
    with pytest.raises(KeyError) as e:
        g.create_group_catalog()
        assert False
    assert (e.value.args[0] == "Non-empty catalogs in a group must all "
            "either have or not have 'weight' column.")
    g[0].catalog.remove_column('weight')

    # artificially set catalog of one of the images to an empty table:
    g[0].catalog.remove_rows(slice(None, None))
    assert len(g.create_group_catalog()) == len(rect_imcat.catalog)

    # artificially set all catalogs to empty table:
    for im in g:
        im.catalog.remove_rows(slice(None, None))
    assert not g.create_group_catalog()

    # artificially set all catalogs to empty table:
    for im in g:
        im._name = None
    assert not g.create_group_catalog()


def test_wcsgroupcat_bb_policy(rect_imcat):
    w1 = copy.deepcopy(rect_imcat)
    w2 = copy.deepcopy(rect_imcat)
    g = WCSGroupCatalog([w1, w2], bb_policy='exact')

    # test approximation to bounding boxes using convex hull:
    g_approx_bb = WCSGroupCatalog([w1, w2], bb_policy='auto')
    assert _same_spherical_polygons(g.polygon, g_approx_bb.polygon)

    g_approx_bb = WCSGroupCatalog([w1, w2], bb_policy=0)
    assert _same_spherical_polygons(g.polygon, g_approx_bb.polygon)

    with pytest.raises(ValueError) as e:
        WCSGroupCatalog([w1, w2], bb_policy='wrong')
    assert e.value.args[0].startswith(
        "'bb_policy' must be either 'auto', 'exact'"
    )


def test_wcsgroupcat_recalc_catalog_radec(mock_fits_wcs, rect_imcat):
    ra, dec = mock_fits_wcs.all_pix2world(rect_imcat.catalog['x'],
                                          rect_imcat.catalog['y'], 0)
    refcat = Table([ra, dec], names=('RA', 'DEC'))
    ref = RefCatalog(refcat)

    wim1 = copy.deepcopy(rect_imcat)
    wim2 = copy.deepcopy(rect_imcat)
    g = WCSGroupCatalog(wim1)

    ra = g.catalog['RA']
    dec = g.catalog['DEC']

    # artificially add another image:
    g._images.append(wim2)

    g.recalc_catalog_radec()

    assert np.allclose(g.catalog['RA'], ra)
    assert np.allclose(g.catalog['DEC'], dec)

    g.align_to_ref(ref)
    mcat = g.get_matched_cat()
    assert len(mcat) == 5
    assert np.allclose(mcat['RA'], ra)
    assert np.allclose(mcat['DEC'], dec)

    del g.catalog['RA']
    del g.catalog['DEC']
    with pytest.raises(RuntimeError):
        g.calc_tanp_xy(None)


def test_wcsgroupcat_match2ref(mock_fits_wcs, rect_imcat):
    ra, dec = mock_fits_wcs.all_pix2world(rect_imcat.catalog['x'],
                                          rect_imcat.catalog['y'], 0)
    refcat = Table([ra[:-1], dec[:-1]], names=('RA', 'DEC'))
    ref = RefCatalog(refcat)

    # unequal catalog lengths
    g = WCSGroupCatalog(rect_imcat)
    with pytest.raises(ValueError):
        g.match2ref(ref, match=None)

    refcat = Table([ra, dec], names=('RA', 'DEC'))
    ref = RefCatalog(refcat)

    # call calc_tanp_xy before matching
    with pytest.raises(RuntimeError) as e:
        g.match2ref(ref, match=XYXYMatch())
    assert (e.value.args[0] == "'calc_tanp_xy()' should have been run "
            "prior to match2ref()")

    ref.calc_tanp_xy(rect_imcat.corrector)
    g.calc_tanp_xy(rect_imcat.corrector)
    g.catalog['matched_ref_id'] = np.ones(5, dtype=bool)
    g.catalog['_raw_matched_ref_idx'] = np.ones(5, dtype=bool)
    g.match2ref(ref, match=XYXYMatch())


def test_wcsgroupcat_fit2ref(mock_fits_wcs, caplog, rect_imcat):
    ra, dec = mock_fits_wcs.all_pix2world(rect_imcat.catalog['x'],
                                          rect_imcat.catalog['y'], 0)
    refcat = Table([ra, dec], names=('RA', 'DEC'))
    ref = RefCatalog(refcat)
    ref.calc_tanp_xy(rect_imcat.corrector)

    g = WCSGroupCatalog(rect_imcat)
    g.calc_tanp_xy(rect_imcat.corrector)
    g.match2ref(ref)

    g.fit2ref(ref, rect_imcat.corrector, fitgeom='shift')
    g.apply_affine_to_wcs(g[0].corrector, np.identity(2), np.zeros(2))

    g._images = []
    g.align_to_ref(ref)
    assert caplog.record_tuples[-1][-1].endswith("Nothing to align.")


def test_wcsrefcat_init_check_catalog(mock_fits_wcs):
    r = RefCatalog(None, name='refcat')

    assert r.name == 'refcat'
    r.name = 'refcat2'
    assert r.name == 'refcat2'

    assert r.poly_area is None

    with pytest.raises(ValueError) as e:
        r.catalog = None
    assert e.value.args[0] == "Reference catalogs cannot be None"

    with pytest.raises(KeyError) as e:
        r.catalog = Table([[1], [1]], names=('x', 'DEC'))
    assert (e.value.args[0] == "Reference catalogs *must* contain *both* 'RA' "
            "and 'DEC' columns.")

    r.expand_catalog(Table([[1], [1]], names=('RA', 'DEC')))


def test_wcsrefcat_intersections(mock_fits_wcs, rect_imcat):
    ra, dec = mock_fits_wcs.all_pix2world(10 * rect_imcat.catalog['x'],
                                          10 * rect_imcat.catalog['y'], 0)
    refcat = Table([ra, dec], names=('RA', 'DEC'))
    ref = RefCatalog(refcat)
    ref.calc_tanp_xy(rect_imcat.corrector)

    pts1 = list(ref.polygon.points)[0]
    pts2 = list(ref.intersection(ref.polygon).points)[0]
    for pt1 in pts1:
        assert any(np.allclose(pt1, pt2) for pt2 in pts2)

    pts2 = list(ref.intersection(ref).points)[0]
    for pt1 in pts1:
        assert any(np.allclose(pt1, pt2) for pt2 in pts2)

    assert np.allclose(
        ref.intersection_area(ref), 2.9902125220360176e-10, atol=0.0,
        rtol=0.005,
    )


def test_wcsrefcat_guarded_intersection_area(mock_fits_wcs, rect_imcat):
    ra, dec = mock_fits_wcs.all_pix2world(10 * rect_imcat.catalog['x'],
                                          10 * rect_imcat.catalog['y'], 0)
    refcat = Table([ra, dec], names=('RA', 'DEC'))
    ref = RefCatalog(refcat)
    ref.calc_tanp_xy(rect_imcat.corrector)

    assert np.allclose(
        ref._guarded_intersection_area(ref)[0],
        2.9902125220360176e-10,
        atol=0.0,
        rtol=0.005,
    )


def test_wcsrefcat_convex_hull(mock_fits_wcs):
    ref = RefCatalog(None)
    ref._calc_cat_convex_hull()  # catalog is None

    assert convex_hull([], []) == ([], [])
    assert all(isinstance(x, np.ndarray) and x.size == 0 for x in
               convex_hull(np.array([]), np.array([])))

    assert convex_hull([123], [456]) == ([123], [456])

    x = [0, 0, 10, 10]
    y = [0, 10, 10, 0]

    xc, yc = convex_hull(x + [10, 3, 6, 7], y + [0, 5, 1, 8])
    assert set(xc) == set(x) and set(yc) == set(y)

    xc, yc = convex_hull(np.array(x + [3, 6, 7]), np.array(y + [5, 1, 8]))
    assert set(xc) == set(x) and set(yc) == set(y)

    xc, yc = convex_hull(x + [3, 6, 7], y + [5, 1, 8], lambda x, y: (x, y))
    assert set(xc) == set(x) and set(yc) == set(y)

    xc, yc = convex_hull(np.array(x + [3, 6, 7]), np.array(y + [5, 1, 8]),
                         lambda x, y: (x, y))
    assert set(xc) == set(x) and set(yc) == set(y)


def test_convex_hull_invalid_min_separation():
    x = [0.0, 10.0, 10.0, 5.0, 0.0, 0.0]
    y = [0.0, 0.0, 10.0, 11.0, 10.0, 0.0]

    with pytest.raises(ValueError) as e:
        convex_hull(x, y, min_separation=-1)
    assert e.value.args[0] == "'min_separation' must be non-negative or None."


def test_convex_hull_adjacent():
    x = [0.0, 10.0, 10.0, 5.001, 5.0, 0.0, 0.0]
    y = [0.0, 0.0, 10.0, 11.000001, 11.0, 10.0, 0.0]

    xc, yc = convex_hull(x, y, min_separation=None)
    assert np.allclose([x, y], [xc, yc], rtol=0, atol=1e-14)

    xc, yc = convex_hull(x, y, min_separation=1e-2)
    x.pop(3)
    y.pop(3)
    assert np.allclose([x, y], [xc, yc], rtol=0, atol=1e-14)
