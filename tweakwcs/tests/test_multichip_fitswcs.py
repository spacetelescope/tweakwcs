from itertools import product

import numpy as np
from astropy.io import fits
from astropy import table
from astropy import wcs
from astropy.utils.data import get_pkg_data_filename
import pytest

import tweakwcs


def _match(x, y):
    lenx = len(x)
    leny = len(y)
    if lenx == leny:
        return (np.arange(lenx), np.arange(leny))
    elif lenx < leny:
        lenx, leny = leny, lenx
        x, y = y, x
    match = (np.arange(leny) + (0 if y.meta['name'] == 'ext1' else leny),
             np.arange(leny))
    return match


def test_multichip_fitswcs_alignment():
    h1 = fits.Header.fromfile(get_pkg_data_filename('data/wfc3_uvis1.hdr'))
    w1 = wcs.WCS(h1)
    imcat1 = tweakwcs.FITSWCS(w1)
    imcat1.meta['catalog'] = table.Table.read(
        get_pkg_data_filename('data/wfc3_uvis1.cat'),
        format='ascii.csv',
        delimiter=' ',
        names=['x', 'y']
    )
    imcat1.meta['group_id'] = 1
    imcat1.meta['name'] = 'ext1'

    h2 = fits.Header.fromfile(get_pkg_data_filename('data/wfc3_uvis2.hdr'))
    w2 = wcs.WCS(h2)
    imcat2 = tweakwcs.FITSWCS(w2)
    imcat2.meta['catalog'] = table.Table.read(
        get_pkg_data_filename('data/wfc3_uvis2.cat'),
        format='ascii.csv',
        delimiter=' ',
        names=['x', 'y']
    )
    imcat2.meta['group_id'] = 1
    imcat2.meta['name'] = 'ext4'

    refcat = table.Table.read(
        get_pkg_data_filename('data/ref.cat'),
        format='ascii.csv', delimiter=' ',
        names=['RA', 'DEC']
    )

    tweakwcs.align_wcs([imcat1, imcat2], refcat, match=_match, nclip=None,
                       sigma=3, fitgeom='general')

    fi1 = imcat1.meta['fit_info']
    fi2 = imcat2.meta['fit_info']

    w1 = imcat1.wcs
    w2 = imcat2.wcs

    assert np.allclose(w1.wcs.crval, (83.206917667519, -67.73275818507248), rtol=0)
    assert np.allclose(
        w1.wcs.cd,
        np.array(
            [
                [3.93222694902149e-06, -1.0106698270131359e-05],
                [-1.0377001075437075e-05, -4.577945148472431e-06]
            ]),
        atol=0.0,
        rtol=1e-8
    )

    assert np.allclose(w2.wcs.crval, (83.15167050722597, -67.74220306069903), rtol=0)
    assert np.allclose(
        w2.wcs.cd,
        np.array(
            [
                [3.834449806681195e-06, -9.996495217498745e-06],
                [-1.0348147451241423e-05, -4.503496019301529e-06]
            ]),
        atol=0.0,
        rtol=1e-8
    )

    assert np.allclose(fi1['<scale>'], 1.0025, rtol=0, atol=2e-8)
    assert np.allclose(fi2['<scale>'], 1.0025, rtol=0, atol=2e-8)

    assert fi1['rmse'] < 5e-5
    assert fi2['rmse'] < 5e-5

    cat1 = imcat1.meta['catalog']
    ra1, dec1 = w1.all_pix2world(cat1['x'], cat1['y'], 0)

    cat2 = imcat2.meta['catalog']
    ra2, dec2 = w2.all_pix2world(cat2['x'], cat2['y'], 0)

    ra = np.concatenate([ra1, ra2])
    dec = np.concatenate([dec1, dec2])

    rmse_ra = np.sqrt(np.mean((ra - refcat['RA'])**2))
    rmse_dec = np.sqrt(np.mean((dec - refcat['DEC'])**2))

    assert rmse_ra < 1e-9
    assert rmse_dec < 1e-9


@pytest.mark.parametrize('wcsno, refscale, dra, ddec', (
    x for x in product(
        [0, 1],
        [((15, 25), (1.0005, 0.9993))],
        [0, 0.05, -0.05, 4, -4],
        [0, 0.05, -0.05, 4, -4]
    )
))
def test_different_ref_tpwcs_fitswcs_alignment(wcsno, refscale, dra, ddec):
    # This test was designed to check that the results of alignment,
    # in particular and most importantly, the sky positions of sources in
    # aligned images do not depend on the tangent reference plane used
    # for alignment. [#125]
    h1 = fits.Header.fromfile(get_pkg_data_filename('data/wfc3_uvis1.hdr'))
    w1 = wcs.WCS(h1)
    imcat1 = tweakwcs.FITSWCS(w1)
    imcat1.meta['catalog'] = table.Table.read(
        get_pkg_data_filename('data/wfc3_uvis1.cat'),
        format='ascii.csv',
        delimiter=' ',
        names=['x', 'y']
    )
    imcat1.meta['group_id'] = 1
    imcat1.meta['name'] = 'ext1'

    h2 = fits.Header.fromfile(get_pkg_data_filename('data/wfc3_uvis2.hdr'))
    w2 = wcs.WCS(h2)
    imcat2 = tweakwcs.FITSWCS(w2)
    imcat2.meta['catalog'] = table.Table.read(
        get_pkg_data_filename('data/wfc3_uvis2.cat'),
        format='ascii.csv',
        delimiter=' ',
        names=['x', 'y']
    )
    imcat2.meta['group_id'] = 1
    imcat2.meta['name'] = 'ext4'

    refcat = table.Table.read(
        get_pkg_data_filename('data/ref.cat'),
        format='ascii.csv', delimiter=' ',
        names=['RA', 'DEC']
    )

    refwcses = [wcs.WCS(h1), wcs.WCS(h2)]

    refwcs = refwcses[wcsno]

    # change pointing of the reference WCS (alignment tangent plane):
    refwcs.wcs.crval = refwcses[1 - wcsno].wcs.crval + np.asarray([dra, ddec])

    rotm = tweakwcs.linearfit.build_fit_matrix(*refscale)
    refwcs.wcs.cd = np.dot(refwcs.wcs.cd, rotm)
    refwcs.wcs.set()
    ref_tpwcs = tweakwcs.FITSWCS(refwcs)

    tweakwcs.align_wcs([imcat1, imcat2], refcat, ref_tpwcs=ref_tpwcs,
                       match=_match, nclip=None, sigma=3, fitgeom='general')

    fi1 = imcat1.meta['fit_info']
    fi2 = imcat2.meta['fit_info']

    w1 = imcat1.wcs
    w2 = imcat2.wcs

    assert fi1['rmse'] < 1e-4
    assert fi2['rmse'] < 1e-4

    cat1 = imcat1.meta['catalog']
    ra1, dec1 = w1.all_pix2world(cat1['x'], cat1['y'], 0)

    cat2 = imcat2.meta['catalog']
    ra2, dec2 = w2.all_pix2world(cat2['x'], cat2['y'], 0)

    ra = np.concatenate([ra1, ra2])
    dec = np.concatenate([dec1, dec2])

    rmse_ra = np.sqrt(np.mean((ra - refcat['RA'])**2))
    rmse_dec = np.sqrt(np.mean((dec - refcat['DEC'])**2))

    assert rmse_ra < 5e-9
    assert rmse_dec < 5e-9
