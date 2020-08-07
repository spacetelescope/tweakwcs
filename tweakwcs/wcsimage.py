# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides support for working with image footprints on the sky and
source catalogs.

:Authors: Mihai Cara

:License: :doc:`../LICENSE`

"""
import os
import logging
import numbers
from copy import deepcopy
from distutils.version import LooseVersion

import numpy as np
from astropy import table
from astropy.utils.decorators import deprecated_renamed_argument
from spherical_geometry.polygon import SphericalPolygon

try:
    import gwcs
    if LooseVersion(gwcs.__version__) > '0.12.0':
        from gwcs.geometry import CartesianToSpherical, SphericalToCartesian
        _GWCS_VER_GT_0P12 = True
    else:
        _GWCS_VER_GT_0P12 = False

except ImportError:
    _GWCS_VER_GT_0P12 = False


if _GWCS_VER_GT_0P12:
    _S2C = SphericalToCartesian(name='s2c', wrap_lon_at=180)
    _C2S = CartesianToSpherical(name='c2s', wrap_lon_at=180)

else:
    def _S2C(phi, theta):
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)
        cs = np.cos(theta)
        x = cs * np.cos(phi)
        y = cs * np.sin(phi)
        z = np.sin(theta)
        return x, y, z

    def _C2S(x, y, z):
        h = np.hypot(x, y)
        phi = np.rad2deg(np.arctan2(y, x))
        theta = np.rad2deg(np.arctan2(z, h))
        return phi, theta

from .wcsutils import planar_rot_3d
from .tpwcs import TPWCS
from .linalg import inv
from .linearfit import iter_linear_fit

from . import __version__  # noqa: F401

__author__ = 'Mihai Cara'

__all__ = ['convex_hull', 'RefCatalog', 'WCSImageCatalog', 'WCSGroupCatalog']


# _S2C = SphericalToCartesian(name='s2c', wrap_lon_at=180)
# _C2S = CartesianToSpherical(name='c2s', wrap_lon_at=180)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def _is_int(n):
    return (
        (isinstance(n, numbers.Integral) and not isinstance(n, bool)) or
        (isinstance(n, np.generic) and np.issubdtype(n, np.integer))
    )


class WCSImageCatalog(object):
    """
    A class that holds information pertinent to an image WCS and a source
    catalog of the sources found in that image.

    .. warning::
        If ``tpwcs.meta`` dictionary contains any of the following
        keywords ``'catalog'``, ``'name'``, or ``'group_id'``, they
        will be ignored without warning.

    """

    def __init__(self, catalog, tpwcs, name=None, group_id=None, meta={}):
        """
        Parameters
        ----------
        catalog: astropy.table.Table
            Source catalog associated with an image. Must contain ``'x'`` and
            ``'y'`` columns which indicate source coordinates (in pixels) in
            the associated image.

        tpwcs: TPWCS
            ``TPWCS``-derived tangent-plane WCS corrector object associated
            with the image from which the catalog was derived.

        name: str, None, optional
            Image catalog's name. This is used to identify catalog during
            logging. If ``name`` is `None`, the ``name`` of this
            ``WCSImageCatalog`` object will be set to ``'Unknown'``.

        group_id: hashable, None, optional
            Group ID that may be used for identifying catalogs that need
            to be aligned together. ``group_id`` must be hashable.

        meta: dict, optional
            Additional information about image, catalog, and/or WCS to be
            stored (for convenience) within `WCSImageCatalog` object.

        """
        self.name = name
        self.group_id = group_id

        self._catalog = None
        self._bb_radec = None

        self.img_bounding_ra = None
        self.img_bounding_dec = None

        self.meta = dict(meta)
        self._fit_info = {'status': 'SKIPPED'}

        self.tpwcs = tpwcs
        self.catalog = catalog

    @property
    def tpwcs(self):
        """ Get :py:class:`TPWCS` WCS. """
        return self._tpwcs

    @tpwcs.setter
    def tpwcs(self, tpwcs):
        """ Get/Set catalog's WCS (a :py:class:`TPWCS`-derived object).

        .. note::
            Setting the WCS triggers automatic bounding polygon recalculation.

        Parameters
        ----------
        tpwcs: TPWCS
            ``TPWCS``-derived tangent-plane WCS corrector object associated
            with the image from which the catalog was extracted.

        """
        if not isinstance(tpwcs, TPWCS):
            raise TypeError("Unsupported 'tpwcs' type. "
                            "'tpwcs' must be a subtype of TPWCS.")
        self._tpwcs = tpwcs

        # create spherical polygon bounding the image
        self.calc_bounding_polygon()

    @property
    def name(self):
        """
        Get/set catalog's name. This is used to identify catalog during
        logging. Upon setting, the value will be converted to a `str`.
        When setting to `None`, the ``name`` will be set to ``'Unknown'``.

        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = 'Unknown' if name is None else str(name)

    @property
    def group_id(self):
        """
        Get/set :py:class:`WCSImageCatalog` object's group ID that may be used
        for identifying catalogs that need to be aligned together.
        ``group_id`` must be hashable.

        """
        return self._group_id

    @group_id.setter
    def group_id(self, group_id):
        # check if input is hashable:
        {group_id: None}
        self._group_id = group_id

    @property
    def fit_status(self):
        """
        Get/Set fit status. This property is a shortcut to the ``'status'``
        key value in the ``fit_info`` dictionary. When the
        :py:class:`WCSImageCatalog` object is created, ``fit_status`` is
        initially set to ``'SKIPPED'``. Alignment tools are reponsible for
        updating catalog's fit status.

        """
        return self._fit_info['status']

    @fit_status.setter
    def fit_status(self, fit_status):
        self._fit_info['status'] = fit_status

    @property
    def fit_info(self):
        """
        Get fit information - a dictionary. This class sets only the
        ``'status'`` field but fitting routines may set additional fields.

        """
        return self._fit_info

    @property
    def catalog(self):
        """ Get/set image's catalog. """
        return self._catalog

    @catalog.setter
    def catalog(self, catalog):
        if catalog is None:
            self._catalog = None
            return

        if 'x' not in catalog.colnames or 'y' not in catalog.colnames:
            raise ValueError("An image catalog must contain 'x' and 'y' "
                             "columns!")

        self._catalog = table.Table(catalog.copy(), masked=True)
        self._catalog.meta['name'] = self._name

        if 'id' not in self._catalog.colnames:  # pragma: no branch
            self._catalog['id'] = np.arange(1, len(self._catalog) + 1)

        # create spherical polygon bounding the image
        self.calc_bounding_polygon()

    def det_to_world(self, x, y):
        """
        Convert pixel coordinates to sky coordinates using full
        (i.e., including distortions) transformations.

        """
        return self._tpwcs.det_to_world(x, y)

    def world_to_det(self, ra, dec):
        """
        Convert sky coordinates to image's pixel coordinates using full
        (i.e., including distortions) transformations.

        """
        return self._tpwcs.world_to_det(ra, dec)

    def det_to_tanp(self, x, y):
        """
        Convert detector (pixel) coordinates to tangent plane coordinates.

        """
        return self._tpwcs.det_to_tanp(x, y)

    def tanp_to_det(self, x, y):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        return self._tpwcs.tanp_to_det(x, y)

    def tanp_to_world(self, x, y):
        """
        Convert tangent plane coordinates to world coordinates.

        """
        return self._tpwcs.tanp_to_world(x, y)

    def world_to_tanp(self, ra, dec):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        return self._tpwcs.world_to_tanp(ra, dec)

    @property
    def polygon(self):
        """ Get image's (or catalog's) bounding spherical polygon.
        """
        return self._polygon

    def intersection(self, wcsim):
        """
        Compute intersection of this `WCSImageCatalog` object and another
        `WCSImageCatalog`, `WCSGroupCatalog`, or
        :py:class:`~spherical_geometry.polygon.SphericalPolygon`
        object.

        Parameters
        ----------
        wcsim: WCSImageCatalog, WCSGroupCatalog, SphericalPolygon
            Another object that should be intersected with this
            `WCSImageCatalog`.

        Returns
        -------
        polygon: SphericalPolygon
            A :py:class:`~spherical_geometry.polygon.SphericalPolygon` that is
            the intersection of this `WCSImageCatalog` and `wcsim`.

        """
        if isinstance(wcsim, (WCSImageCatalog, WCSGroupCatalog)):
            return self._polygon.intersection(wcsim.polygon)
        else:
            return self._polygon.intersection(wcsim)

    # TODO: due to a bug in the sphere package, see
    #       https://github.com/spacetelescope/sphere/issues/74
    #       intersections with polygons formed as union does not work.
    #       For this reason I re-implement 'intersection_area' below with
    #       a workaround for the bug.
    #       The original implementation should be uncommented once the bug
    #       is fixed.
    #
    # def intersection_area(self, wcsim):
        # """ Calculate the area of the intersection polygon.
        # """
        # return np.fabs(self.intersection(wcsim).area())
    def intersection_area(self, wcsim):
        """ Calculate the area of the intersection polygon. """
        if isinstance(wcsim, (WCSImageCatalog, RefCatalog)):
            return np.fabs(self.intersection(wcsim).area())

        else:
            # this is bug workaround for image groups (multi-unions):
            area = 0.0
            for wim in wcsim:
                area += np.fabs(
                    self.polygon.intersection(wim.polygon).area()
                )
            return area

    def _calc_chip_bounding_polygon(self, stepsize=None):
        """
        Compute image's bounding polygon.

        Parameters
        ----------
        stepsize: int, None, optional
            Indicates the maximum separation between two adjacent vertices
            of the bounding polygon along each side of the image. Corners
            of the image are included automatically. If `stepsize` is `None`,
            bounding polygon will contain only vertices of the image.

        """
        if self.tpwcs is None or self.catalog is None:
            return

        if self.tpwcs.bounding_box is None:
            # just take max image coordinates from catalogs as bounds:
            lx = -0.5
            ly = -0.5
            hx = max(1, int(np.ceil(np.amax(self._catalog['x'])))) - 0.5
            hy = max(1, int(np.ceil(np.amax(self._catalog['y'])))) - 0.5

        else:
            ((lx, hx), (ly, hy)) = self.tpwcs.bounding_box

        if stepsize is None:
            nintx = 2
            ninty = 2
        else:
            nintx = max(2, int(np.ceil((hx - lx) / stepsize)))
            ninty = max(2, int(np.ceil((hy - ly) / stepsize)))

        xs = np.linspace(lx, hx, nintx, dtype=np.double)
        ys = np.linspace(ly, hy, ninty, dtype=np.double)[1:-1]
        nptx = xs.size
        npty = ys.size

        npts = 2 * (nptx + npty)

        borderx = np.empty((npts + 1,), dtype=np.double)
        bordery = np.empty((npts + 1,), dtype=np.double)

        # "bottom" points:
        borderx[:nptx] = xs
        bordery[:nptx] = ly
        # "right"
        sl = np.s_[nptx:nptx + npty]
        borderx[sl] = hx
        bordery[sl] = ys
        # "top"
        sl = np.s_[nptx + npty:2 * nptx + npty]
        borderx[sl] = xs[::-1]
        bordery[sl] = hy
        # "left"
        sl = np.s_[2 * nptx + npty:-1]
        borderx[sl] = lx
        bordery[sl] = ys[::-1]

        # close polygon:
        borderx[-1] = borderx[0]
        bordery[-1] = bordery[0]

        ra, dec = self.det_to_world(borderx, bordery)
        # TODO: for strange reasons, occasionally ra[0] != ra[-1] and/or
        #       dec[0] != dec[-1] (even though we close the polygon in the
        #       previous two lines). Then SphericalPolygon fails because
        #       points are not closed. Therefore we force it to be closed:
        ra[-1] = ra[0]
        dec[-1] = dec[0]

        self.img_bounding_ra = ra
        self.img_bounding_dec = dec
        self._bb_radec = (ra, dec)
        self._polygon = SphericalPolygon.from_radec(ra, dec)

    def _calc_cat_convex_hull(self):
        """
        Compute convex hull that bounds the sources in the catalog.

        """
        if self.tpwcs is None or self.catalog is None:
            return

        x = self.catalog['x']
        y = self.catalog['y']

        if len(x) == 0:
            # no points
            raise RuntimeError(  # pragma: no cover
                "Unexpected error: Contact software developer"
            )

        elif len(x) > 2:
            ra, dec = convex_hull(x, y, wcs=self.det_to_world)
            # else, for len(x) in [1, 2], use entire image footprint.
            # TODO: a more robust algorithm should be implemented to deal with
            #       len(x) in [1, 2] cases.

            # TODO: for strange reasons, occasionally ra[0] != ra[-1] and/or
            #       dec[0] != dec[-1] (even though we close the polygon in the
            #       previous two lines). Then SphericalPolygon fails because
            #       points are not closed. Threfore we force it to be closed:
            ra[-1] = ra[0]
            dec[-1] = dec[0]

            self._bb_radec = (ra, dec)
            self._polygon = SphericalPolygon.from_radec(ra, dec)
            self._poly_area = np.fabs(self._polygon.area())

    def calc_bounding_polygon(self):
        """
        Calculate bounding polygon of the image or of the sources in the
        catalog (if catalog was set).

        """
        # we need image's footprint for later:
        self._calc_chip_bounding_polygon()

        # create smallest convex spherical polygon bounding all sources:
        if self.catalog:
            self._calc_cat_convex_hull()

    @property
    def bb_radec(self):
        """
        Get a 2xN `numpy.ndarray` of RA and DEC of the vertices of the
        bounding polygon.

        """
        return self._bb_radec


class WCSGroupCatalog(object):
    """
    A class that holds together `WCSImageCatalog` image catalog objects
    whose relative positions are fixed and whose source catalogs should be
    fitted together to a reference catalog.

    """

    def __init__(self, images, name=None):
        """
        Parameters
        ----------
        images: list of WCSImageCatalog
            A list of `WCSImageCatalog` image catalogs.

        name: str, None, optional
            Name of the group.

        """
        self._catalog = None

        if isinstance(images, WCSImageCatalog):
            self._images = [images]
            if images.catalog is None:
                raise ValueError("Each input WCS image catalog must have a "
                                 "valid catalog.")

        elif hasattr(images, '__iter__'):
            if not images:
                raise ValueError("List of images cannot be empty.")

            self._images = []
            for im in images:
                if not isinstance(im, WCSImageCatalog):
                    raise TypeError("Each element of the 'images' parameter "
                                    "must be an 'WCSImageCatalog' object.")
                if im.catalog is None:
                    raise ValueError("Each input WCS image catalog must have "
                                     "a valid catalog.")
                self._images.append(im)

        else:
            raise TypeError("Parameter 'images' must be either a single "
                            "'WCSImageCatalog' object or a list of "
                            "'WCSImageCatalog' objects")

        self._name = name
        self.update_bounding_polygon()
        self._catalog = self.create_group_catalog()

    @property
    def name(self):
        """ Get/set :py:class:`WCSImageCatalog` object's name.
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def polygon(self):
        """ Get image's (or catalog's) bounding spherical polygon.
        """
        return self._polygon

    def intersection(self, wcsim):
        """
        Compute intersection of this `WCSGroupCatalog` object and another
        `WCSImageCatalog`, `WCSGroupCatalog`, or
        :py:class:`~spherical_geometry.polygon.SphericalPolygon`
        object.

        Parameters
        ----------
        wcsim: WCSImageCatalog, WCSGroupCatalog, SphericalPolygon
            Another object that should be intersected with this
            `WCSGroupCatalog`.

        Returns
        -------
        polygon: SphericalPolygon
            A :py:class:`~spherical_geometry.polygon.SphericalPolygon` that is
            the intersection of this `WCSGroupCatalog` and `wcsim`.

        """
        if isinstance(wcsim, (WCSGroupCatalog, WCSImageCatalog)):
            return self._polygon.intersection(wcsim.polygon)
        else:
            return self._polygon.intersection(wcsim)

    # TODO: due to a bug in the sphere package, see
    #       https://github.com/spacetelescope/sphere/issues/74
    #       intersections with polygons formed as union does not work.
    #       For this reason I re-implement 'intersection_area' below with
    #       a workaround for the bug.
    #       The original implementation should be uncommented once the bug
    #       is fixed.
    #
    # def intersection_area(self, wcsim):
        # """ Calculate the area of the intersection polygon.
        # """
        # return np.fabs(self.intersection(wcsim).area())
    def intersection_area(self, wcsim):
        """ Calculate the area of the intersection polygon.
        """
        return sum(im.intersection_area(wcsim) for im in self._images)

    def update_bounding_polygon(self):
        """ Recompute bounding polygons of the member images.
        """
        polygons = [im.polygon for im in self._images]
        if polygons:
            self._polygon = SphericalPolygon.multi_union(polygons)
        else:
            self._polygon = SphericalPolygon([])

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        return self._images[idx]

    def __iter__(self):
        for image in self._images:
            yield image

    @property
    def catalog(self):
        """ Get/set image's catalog.
        """
        return self._catalog

    def create_group_catalog(self):
        """
        Combine member's image catalogs into a single group's catalog.

        Returns
        -------
        group_catalog: astropy.table.Table
            Combined group catalog.

        """
        catalogs = []
        catno = 0
        has_weights = None
        cat_names = []
        for image in self._images:
            catlen = len(image.catalog)
            if catlen == 0:
                continue

            cat_name = image.catalog.meta.get(
                'name',
                image.name if image.name else 'Unnamed'
            )
            cat_names.append(cat_name)

            if has_weights is None:
                has_weights = 'weight' in image.catalog.colnames
            elif has_weights != ('weight' in image.catalog.colnames):
                raise KeyError("Non-empty catalogs in a group must all "
                               "either have or not have 'weight' column.")

            if image.name is None:
                catname = 'Catalog #{:d}'.format(catno)
            else:
                catname = image.name

            col_catname = table.MaskedColumn(catlen * [catname],
                                             name='cat_name')
            col_imcatidx = table.MaskedColumn(catlen * [catno],
                                              name='_imcat_idx')
            col_id = table.MaskedColumn(image.catalog['id'])
            col_x = table.MaskedColumn(image.catalog['x'], dtype=np.double)
            col_y = table.MaskedColumn(image.catalog['y'], dtype=np.double)
            ra, dec = image.det_to_world(
                image.catalog['x'], image.catalog['y']
            )
            col_ra = table.MaskedColumn(ra, dtype=np.double, name='RA')
            col_dec = table.MaskedColumn(dec, dtype=np.double, name='DEC')

            if has_weights:
                col_wght = table.MaskedColumn(image.catalog['weight'],
                                              dtype=np.double)

                cat = table.Table(
                    [col_imcatidx, col_catname, col_id, col_x,
                     col_y, col_ra, col_dec, col_wght],
                    masked=True
                )

            else:
                cat = table.Table(
                    [col_imcatidx, col_catname, col_id, col_x,
                     col_y, col_ra, col_dec],
                    masked=True
                )

            catalogs.append(cat)
            catno += 1

        catname = os.path.commonprefix(cat_names) if cat_names else None

        if catno:
            cat = table.vstack(catalogs, join_type='exact')

        else:
            # no catalogs with sources. Create an empty table with required
            # columns and types:
            image = self._images[0]
            if image.name is None:
                catname = 'Catalog #{:d}'.format(catno)
            else:
                catname = image.name

            col_catname = table.MaskedColumn([catname], name='cat_name')
            col_catname = col_catname[[False]]
            col_imcatidx = table.MaskedColumn([], dtype=np.int,
                                              name='_imcat_idx')
            col_id = table.MaskedColumn(image.catalog['id'])
            col_x = table.MaskedColumn([], name='x', dtype=np.double)
            col_y = table.MaskedColumn([], name='y', dtype=np.double)
            col_ra = table.MaskedColumn([], name='RA', dtype=np.double)
            col_dec = table.MaskedColumn([], name='DEC', dtype=np.double)

            cat = table.Table(
                [col_imcatidx, col_catname, col_id, col_x,
                 col_y, col_ra, col_dec],
                masked=True
            )

        if catname:
            cat.meta['name'] = catname

        return cat

    def get_unmatched_cat(self):
        """
        Retrieve only those sources from the catalog that have **not** been
        matched to the sources in the reference catalog.

        """
        mask = self._catalog['matched_ref_id'].mask
        return self._catalog[mask]

    def get_matched_cat(self):
        """
        Retrieve only those sources from the catalog that **have been**
        matched to the sources in the reference catalog.

        """
        mask = np.logical_not(self._catalog['matched_ref_id'].mask)
        return self._catalog[mask]

    def recalc_catalog_radec(self):
        """ Recalculate RA and DEC of the sources in the catalog.
        """
        for k, image in enumerate(self._images):

            idx = (self._catalog['_imcat_idx'] == k)
            if not np.any(idx):
                continue

            ra, dec = image.det_to_world(
                self._catalog['x'][idx], self._catalog['y'][idx]
            )
            self._catalog['RA'][idx] = ra
            self._catalog['DEC'][idx] = dec

    def calc_tanp_xy(self, tanplane_wcs):
        """
        Compute x- and y-positions of the sources from the image catalog
        in the tangent plane. This creates the following
        columns in the catalog's table: ``'TPx'`` and ``'TPy'``.

        Parameters
        ----------
        tanplane_wcs: TPWCS
            A `TPWCS` object that will provide transformations to
            the tangent plane to which sources of this catalog a should be
            "projected".

        """
        if 'RA' not in self._catalog.colnames or \
           'DEC' not in self._catalog.colnames:
            raise RuntimeError("'recalc_catalog_radec()' should have been run "
                               "prior to calc_tanp_xy()")

        # compute x & y in the reference WCS:
        xtp, ytp = tanplane_wcs.world_to_tanp(self.catalog['RA'],
                                              self.catalog['DEC'])
        self._catalog['TPx'] = table.MaskedColumn(
            xtp, name='TPx', dtype=np.double, mask=False
        )
        self._catalog['TPy'] = table.MaskedColumn(
            ytp, name='TPy', dtype=np.double, mask=False
        )

    def match2ref(self, refcat, match=None):
        """ Uses ``xyxymatch`` to cross-match sources between this catalog and
            a reference catalog.

        Parameters
        ----------
        refcat: RefCatalog
            A `RefCatalog` object that contains a catalog of reference sources
            as well as a valid reference WCS.

        match: MatchCatalogs, function, None, optional
            A callable that takes two arguments: a reference catalog and an
            image catalog. Both catalogs will have columns ``'TPx'`` and
            ``'TPy'`` that represent the source coordinates in some common
            (to both catalogs) coordinate system.

        """
        colnames = self._catalog.colnames
        catlen = len(self._catalog)

        if match is None:
            if catlen != len(refcat.catalog):
                raise ValueError("When matching is not requested, catalogs "
                                 "should have been matched previously and "
                                 "have equal lengths.")
            log.info("No matching of sources from '{:}' catalog with sources "
                     "from the reference '{:}' catalog was requested."
                     .format(self.name, refcat.name))
            log.info("Catalogs are assumed matched with 1-to-1 "
                     "correspondence.")
            mref_idx = np.arange(catlen)
            minput_idx = np.arange(catlen)
            nmatches = catlen

        else:
            if 'TPx' not in colnames or 'TPy' not in colnames:
                raise RuntimeError("'calc_tanp_xy()' should have been run "
                                   "prior to match2ref()")

            mref_idx, minput_idx = match(refcat.catalog, self._catalog)
            nmatches = len(mref_idx)

        # matched_ref_id:
        if 'matched_ref_id' in colnames:
            self._catalog['matched_ref_id'].mask[:] = True
        else:
            c = table.MaskedColumn(name='matched_ref_id', dtype=int,
                                   length=catlen, mask=True)
            self._catalog.add_column(c)

        self._catalog['matched_ref_id'].mask[minput_idx] = False
        self._catalog['matched_ref_id'][minput_idx] = refcat.catalog['id'][mref_idx]  # noqa: E501

        # this is needed to index reference catalog directly without using
        # astropy table indexing which, at this moment, is experimental:
        if '_raw_matched_ref_idx' in colnames:
            self._catalog['_raw_matched_ref_idx'].mask = True
        else:
            c = table.MaskedColumn(name='_raw_matched_ref_idx',
                                   dtype=int, length=catlen, mask=True)
            self._catalog.add_column(c)
        self._catalog['_raw_matched_ref_idx'][minput_idx] = mref_idx
        self._catalog['_raw_matched_ref_idx'].mask[minput_idx] = False

        log.info("Found {:d} matches for '{}'...".format(nmatches, self.name))

        # TODO: revisit this once the bug described in
        # https://github.com/spacetelescope/stsci.stimage/issues/8
        # is fixed.
        #
        # Due to this bug minput_idx may contain duplicate values.
        # Because of this, the above logic for masking, saving ids, and indices
        # does not work reliably. As a workaround, we save matched array
        # indices within the image catalog so that we can use them in
        # `fit2ref()`. For this to work, reference and image catalog must not
        # change between this function return and `fit2ref()` call.
        #
        self._mref_idx = mref_idx
        self._minput_idx = minput_idx

        return nmatches, mref_idx, minput_idx

    def fit2ref(self, refcat, tanplane_wcs, fitgeom='general', nclip=3,
                sigma=(3.0, 'rmse')):
        """
        Perform linear fit of this group's combined catalog to the reference
        catalog. When either/both group's catalog or/and the reference catalog
        contain ``'weight'`` column, weigted fitting will be performed.
        See ``Notes`` section for further details.

        Parameters
        ----------
        refcat: RefCatalog
            A `RefCatalog` object that contains a catalog of reference sources.

        tanplane_wcs: TPWCS
            A `TPWCS` object that will provide transformations to
            the tangent plane to which sources of this catalog a should be
            "projected".

        fitgeom: {'shift', 'rscale', 'general'}, optional
            The fitting geometry to be used in fitting the matched object
            lists. This parameter is used in fitting the offsets, rotations
            and/or scale changes from the matched object lists. The 'general'
            fit geometry allows for independent scale and rotation for
            each axis.

        nclip: int, None, optional
            Number (a non-negative integer) of clipping iterations in fit.
            Clipping will be turned off if ``nclip`` is either `None` or 0.

        sigma: float, tuple of the form (float, str), optional
            When a tuple is provided, first value (a positive number)
            indicates the number of "fit error estimates" to use for clipping.
            The second value (a string) indicates the statistic to be
            used for "fit error estimate". Currently the following values are
            supported: ``'rmse'``, ``'mae'``, and ``'std'``
            - see `~tweakwcs.linearfit.iter_linear_fit` for more details.

            When ``sigma`` is a single number, it must be a positive number and
            the default error estimate ``'rmse'`` is assumed.

            This parameter is ignored when ``nclip`` is either `None` or 0.

        Notes
        -----
        When fitting image sources to reference catalog sources, we can
        specify which sources have higher weights. This can be done by
        assigning a "weight" to each source by specifying these values
        in the optional ``'weight'`` column of either the reference catalog,
        image catalog, or both.

        When weights are not provided, all sources are weighed equally. When
        only one of image or reference catalog weights are provided,
        the sources will be weighted with the specified weights.
        When *both* image *and* reference catalogs specify weights for
        the same sources, the two weights will be combined into a single
        weight as:

        .. math::
            1/w = 1/w_i + 1/w_r

        .. warning::
            Keep in mind that when a group catalog is created from individual
            catalogs, weights of the group catalog are created by
            *concatenating* weights of individual catalogs. Therefore,
            for the weighting of groups of catalogs to work correctly,
            the weights of individual catalogs should be scaled in such a way
            that when individual catalogs are combined into a single
            "group catalog", weights preserve their relative values.

            For example, let's say a group is formed from two individual
            catalogs. Let's say first catalog contains four sources with equal
            weights ``[1,1,1,1]`` and the second catalog contains two sources
            with weights ``[1,1]`` then the group's catalogs sources will
            also have equal weights ``[1,1,1,1,1,1]``. However, if each
            individual catalog's weights were normalized such that sum of
            all weights is 1, then group's sources will be weighed unequally:
            ``[0.25,0.25,0.25,0.25,0.5,0.5]``.

        """
        im_xyref = np.asarray([self._catalog['TPx'],
                               self._catalog['TPy']]).T
        refxy = np.asarray([refcat.catalog['TPx'],
                            refcat.catalog['TPy']]).T

        # mask = np.logical_not(self._catalog['matched_ref_id'].mask)
        # im_xyref = im_xyref[mask]
        # ref_idx = self._catalog['_raw_matched_ref_idx'][mask]

        # TODO: revisit this once the bug described in
        # https://github.com/spacetelescope/stsci.stimage/issues/8
        # is fixed.
        #
        # Due to this bug minput_idx may contain duplicate values.
        # For now we bypass the above commented code by accessing indices
        # stored in the image's catalog.
        minput_idx = self._minput_idx
        im_xyref = im_xyref[minput_idx]
        ref_idx = self._mref_idx

        refxy = refxy[ref_idx]

        # process weights:
        if 'weight' in self._catalog.colnames:
            im_weight = np.asarray(self._catalog['weight'])[minput_idx]
        else:
            im_weight = None

        if 'weight' in refcat.catalog.colnames:
            ref_weight = np.asarray(refcat.catalog['weight'])[ref_idx]
        else:
            ref_weight = None

        fit = iter_linear_fit(
            refxy, im_xyref, ref_weight, im_weight,
            fitgeom=fitgeom, nclip=nclip, sigma=sigma, center=None
        )

        # re-compute shifts for the center at (0, 0):
        fit['shift_ld'] += fit['center_ld'] - np.dot(fit['center_ld'], fit['matrix_ld'].T)
        fit['shift'] = fit['shift_ld'].astype(np.double)

        xy_fit = fit['shift'] + np.dot(im_xyref[fit['fitmask']], fit['matrix'].T)

        fit['fit_xy'] = xy_fit
        fit['fit_RA'], fit['fit_DEC'] = tanplane_wcs.tanp_to_world(*(xy_fit.T))

        log.info("Computed '{:s}' fit for {}:".format(fitgeom, self.name))
        if fitgeom == 'shift':
            log.info("XSH: {:.6g}  YSH: {:.6g}".format(*fit['shift']))
        elif fitgeom == 'rscale' and fit['proper']:
            log.info(
                "XSH: {:.6g}  YSH: {:.6g}    ROT: {:.6g}    SCALE: {:.6g}"
                .format(*fit['shift'], fit['proper_rot'], fit['<scale>'])
            )
        elif fitgeom == 'general' or (fitgeom == 'rscale' and not
                                      fit['proper']):
            log.info("XSH: {:.6g}  YSH: {:.6g}    PROPER ROT: {:.6g}    "
                     .format(*fit['shift'], fit['proper_rot']))
            log.info("<ROT>: {:.6g}  SKEW: {:.6g}    ROT_X: {:.6g}  "
                     "ROT_Y: {:.6g}".format(fit['<rot>'], fit['skew'],
                                            *fit['rot']))
            log.info("<SCALE>: {:.6g}  SCALE_X: {:.6g}  SCALE_Y: {:.6g}"
                     .format(fit['<scale>'], *fit['scale']))
        else:
            raise ValueError("Unsupported fit geometry.")  # pragma: no cover

        log.info("")
        log.info("FIT RMSE: {:.6g}   FIT MAE: {:.6g}"
                 .format(fit['rmse'], fit['mae']))
        log.info("Final solution based on {:d} objects."
                 .format(fit['resids'].shape[0]))

        return fit

    @deprecated_renamed_argument('tanplane_wcs', 'ref_tpwcs', since='0.6.5')
    def apply_affine_to_wcs(self, ref_tpwcs, matrix, shift, meta=None):
        """ Applies a general affine transformation to the WCS. """
        for imcat in self:
            imcat.tpwcs.set_correction(matrix, shift, ref_tpwcs=ref_tpwcs, meta=meta)

    def align_to_ref(self, refcat, ref_tpwcs=None, match=None, minobj=None,
                     fitgeom='rscale', nclip=3, sigma=(3.0, 'rmse')):
        """
        Matches sources from the image catalog to the sources in the
        reference catalog, finds the affine transformation between matched
        sources, and adjusts images' WCS according to this fit.

        Upon successful return, this function will also set the following
        fields of the ``fit_info`` attribute of each member
        `WCSImageCatalog` object:

            * **'fitgeom'**: the value of the ``fitgeom`` argument
            * **'eff_minobj'**: effective value of the ``minobj`` parameter
            * **'matrix'**: computed rotation matrix
            * **'shift'**: shift (offset) along X- and Y-axis
            * **'rot'**: A tuple of ``(rotx, roty)`` - the rotation angles with
              regard to the ``X`` and ``Y`` axes.
            * **'<rot>'**: *Arithmetic mean* of the angles of rotation around
              ``X`` and ``Y`` axes.
            * **'proper_rot'**: rotation angle as if rotation is a proper
              rotation.
            * **'proper'**: Indicates whether the rotation is a proper rotation
              (boolean)
            * **'scale'**: A tuple of ``(sx, sy)`` - scale change in the
              direction of the ``X`` and ``Y`` axes.
            * **'<scale>'**: *Geometric mean* of scales ``sx`` and ``sy``.
            * **'skew'**: Computed skew - an angle in the range [-180, 180).
            * **'center'**: Center of rotation in the *tangent plane* of the
              computed linear transformations.
            * **'fitmask'**: boolean array indicating (with `True`) sources
              **used** for fitting
            * **'nmatches'** [when ``match`` is not `None`]: number of matched
              sources
            * **'matched_ref_idx'** [when ``match`` is not `None`]: indices of
              the matched sources in the reference catalog
            * **'matched_input_idx'** [when ``match`` is not `None`]: indices
              of the matched sources in the "input" catalog (the catalog from
              image to be aligned)
            * **'fit_ref_idx'**: indices of the sources from the reference
              catalog used for fitting (a subset of 'matched_ref_idx' indices,
              when ``match`` is not `None`, left after clipping iterations
              performed during fitting)
            * **'fit_input_idx'**: indices of the sources from the "input"
              (image) catalog used for fitting (a subset of
              'matched_input_idx' indices, when ``match`` is not `None`,
              left after clipping iterations performed during fitting)
            * **'rmse'**: fit Root-Mean-Square Error in *tangent plane*
              coordinates of corrected image source positions from reference
              source positions.
            * **'mae'**: fit Mean Absolute Error in *tangent plane*
              coordinates of corrected image source positions from reference
              source positions.
            * **'std'**: Norm of the STandard Deviation of the residuals
              in *tangent plane* along each axis.
            * **'fit_RA'**: first (corrected) world coordinate of input source
              positions used in fitting.
            * **'fit_DEC'**: second (corrected) world coordinate of input
              source positions used in fitting.
            * **'status'**: Alignment status. Currently two possible status are
              possible ``'SUCCESS'`` or ``'FAILED: reason for failure'``.
              When alignment failed, the reason for failure is provided after
              alignment status.

        .. note::
            A ``'SUCCESS'`` status does not indicate a "good" alignment. It
            simply indicates that alignment algortithm has completed without
            errors. Use other fields to evaluate alignment: fit ``RMSE``
            and ``MAE`` values, number of matched sources, etc.

        Parameters
        ----------
        refcat: RefCatalog
            A `RefCatalog` object that contains a catalog of reference sources.

        ref_tpwcs : TPWCS
            A `TPWCS` object that defines a projection tangent plane to be
            used for matching and fitting during alignment.

        match: MatchCatalogs, function, None, optional
            A callable that takes two arguments: a reference catalog and an
            image catalog.

        minobj: int, None, optional
            Minimum number of identified objects from each input image to use
            in matching objects from other images. If the default `None` value
            is used then `align` will automatically deternmine the minimum
            number of sources from the value of the `fitgeom` parameter.
            This parameter is used to interrupt alignment process (catalog
            fitting, ``WCS`` "tweaking") when the number of matched sources
            is smaller than the value of ``minobj`` in which case this
            method will return `False`.

        fitgeom: {'shift', 'rscale', 'general'}, optional
            The fitting geometry to be used in fitting the matched object
            lists. This parameter is used in fitting the offsets, rotations
            and/or scale changes from the matched object lists. The 'general'
            fit geometry allows for independent scale and rotation for each
            axis. This parameter is ignored if ``match`` is `False`.

        nclip: int, None, optional
            Number (a non-negative integer) of clipping iterations in fit.
            Clipping will be turned off if ``nclip`` is either `None` or 0.

            This parameter is ignored if ``match`` is `False`.

        sigma: float, tuple of the form (float, str), optional
            When a tuple is provided, first value (a positive number)
            indicates the number of "fit error estimates" to use for clipping.
            The second value (a string) indicates the statistic to be
            used for "fit error estimate". Currently the following values are
            supported: ``'rmse'``, ``'mae'``, and ``'std'``
            - see `~tweakwcs.linearfit.iter_linear_fit` for more details.

            When ``sigma`` is a single number, it must be a positive number and
            the default error estimate ``'rmse'`` is assumed.

            This parameter is ignored when ``nclip`` is either `None` or 0
            or when ``match`` is `False`.

        Returns
        -------
        bool
            Returns `True` if the number of matched sources is larger or equal
            to ``minobj`` and all steps have been performed, including catalog
            fitting and ``WCS`` "tweaking". If the number of matched sources is
            smaller than ``minobj``, this function will return `False`.

        """
        if not self._images:
            name = 'Unnamed' if self.name is None else self.name
            log.warning("WCSGroupCatalog '{:s}' is empty. Nothing to align."
                        .format(name))
            return False

        # set initial status to 'FAILED':
        for imcat in self:
            imcat.fit_status = "FAILED: Unknown error"

        if minobj is None:
            if fitgeom == 'general':
                minobj = 3
            elif fitgeom == 'rscale':
                minobj = 2
            else:
                minobj = 1

        if ref_tpwcs is None:
            ref_tpwcs = deepcopy(self._images[0].tpwcs)

        self.calc_tanp_xy(tanplane_wcs=ref_tpwcs)
        refcat.calc_tanp_xy(tanplane_wcs=ref_tpwcs)

        nmatches, mref_idx, minput_idx = self.match2ref(
            refcat=refcat,
            match=match
        )

        if nmatches < minobj:
            name = 'Unnamed' if self.name is None else self.name
            log.warning("Not enough matches (< {:d}) found for image "
                        "catalog '{:s}'.".format(nmatches, name))
            for imcat in self:
                imcat.fit_status = 'FAILED: not enough matches'
            return False

        fit = self.fit2ref(refcat=refcat, tanplane_wcs=ref_tpwcs,
                           fitgeom=fitgeom, nclip=nclip, sigma=sigma)

        fit_info = {
            'fitgeom': fitgeom,
            'eff_minobj': minobj,
            'matrix': fit['matrix'],
            'shift': fit['shift'],
            'center': fit['center'],  # center of rotation in geom. transforms
            'fitmask': fit['fitmask'],  # sources was used for fitting
            'proper_rot': fit['proper_rot'],  # proper rotation
            'proper': fit['proper'],  # is a proper rotation? True/False
            'rot': fit['rot'],  # rotx, roty
            '<rot>': fit['<rot>'],  # Arithmetic mean of rotx and roty
            'scale': fit['scale'],  # sx, sy
            '<scale>': fit['<scale>'],  # Geometric mean of sx, sy
            'skew': fit['skew'],  # skew
            'rmse': fit['rmse'],  # fit RMSE in tangent plane coords
            'mae': fit['mae'],  # fit MAE in tangent plane coords
            'fit_RA': fit['fit_RA'],
            'fit_DEC': fit['fit_DEC'],
            'status': 'SUCCESS',
        }

        if match is not None:
            fit_info.update({
                'nmatches': nmatches,
                'matched_ref_idx': mref_idx,
                'matched_input_idx': minput_idx
            })

        self.apply_affine_to_wcs(
            ref_tpwcs=ref_tpwcs,
            matrix=fit['matrix'],
            shift=fit['shift'],
            # meta=meta
        )

        for imcat in self:
            imcat.fit_info.update(deepcopy(fit_info))

        self.recalc_catalog_radec()

        return True


class RefCatalog(object):
    """
    An object that holds a reference catalog and provides
    tools for coordinate convertions using reference WCS as well as
    catalog manipulation and expansion.

    """

    def __init__(self, catalog, name=None, footprint_tol=1.0):
        """
        Parameters
        ----------
        catalog: astropy.table.Table
            Reference catalog.

            .. note::
                Reference catalogs (:py:class:`~astropy.table.Table`)
                *must* contain *both* ``'RA'`` and ``'DEC'`` columns.

        name: str, None, optional
            Name of the reference catalog.

        footprint_tol: float, optional
            Matching tolerance in arcsec. This is used to estimate catalog's
            footprint when catalog contains only one or two sources.

        """
        self._name = name
        self._catalog = None
        self._footprint_tol = footprint_tol
        self._poly_area = None

        # make sure catalog has RA & DEC
        if catalog is not None:
            self.catalog = catalog

    def _check_catalog(self, catalog):
        if catalog is None:
            raise ValueError("Reference catalogs cannot be None")

        if 'RA' not in catalog.colnames or 'DEC' not in catalog.colnames:
            raise KeyError("Reference catalogs *must* contain *both* 'RA' "
                           "and 'DEC' columns.")

    @property
    def name(self):
        """ Get/set :py:class:`RefCatalog` object's name.
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def catalog(self):
        """ Get/set image's catalog.
        """
        return self._catalog

    @catalog.setter
    def catalog(self, catalog):
        self._check_catalog(catalog)

        if not catalog:
            raise ValueError("Reference catalog must contain at least one "
                             "source.")

        self._catalog = catalog.copy()

        if 'id' not in self._catalog.colnames:
            self._catalog['id'] = np.arange(1, len(self._catalog) + 1)

        # create spherical polygon bounding the sources
        self.calc_bounding_polygon()

    @property
    def poly_area(self):
        """ Area of the bounding polygon (in srad).
        """
        return self._poly_area

    @property
    def polygon(self):
        """ Get image's (or catalog's) bounding spherical polygon.
        """
        return self._polygon

    def intersection(self, wcsim):
        """
        Compute intersection of this `WCSImageCatalog` object and another
        `WCSImageCatalog`, `WCSGroupCatalog`, `RefCatalog`, or
        :py:class:`~spherical_geometry.polygon.SphericalPolygon`
        object.

        Parameters
        ----------
        wcsim: WCSImageCatalog, WCSGroupCatalog, RefCatalog, SphericalPolygon
            Another object that should be intersected with this
            `WCSImageCatalog`.

        Returns
        -------
        polygon: SphericalPolygon
            A :py:class:`~spherical_geometry.polygon.SphericalPolygon` that is
            the intersection of this `WCSImageCatalog` and `wcsim`.

        """
        if isinstance(wcsim, (WCSImageCatalog, WCSGroupCatalog, RefCatalog)):
            return self._polygon.intersection(wcsim.polygon)
        else:
            return self._polygon.intersection(wcsim)

    # TODO: due to a bug in the sphere package, see
    #       https://github.com/spacetelescope/sphere/issues/74
    #       intersections with polygons formed as union does not work.
    #       For this reason I re-implement 'intersection_area' below with
    #       a workaround for the bug.
    #       The original implementation should be uncommented once the bug
    #       is fixed.
    #
    # def intersection_area(self, wcsim):
        # """ Calculate the area of the intersection polygon.
        # """
        # return np.fabs(self.intersection(wcsim).area())
    def intersection_area(self, wcsim):
        """ Calculate the area of the intersection polygon.
        """
        if isinstance(wcsim, (WCSImageCatalog, RefCatalog)):
            return np.fabs(self.intersection(wcsim).area())

        else:
            # this is bug workaround:
            area = 0.0
            for wim in wcsim:
                area += np.fabs(
                    self.polygon.intersection(wim.polygon).area()
                )
            return area

    def _calc_cat_convex_hull(self):
        """
        Calculate spherical polygon corresponding to the convex hull bounding
        the sources in the catalog.

        """
        if self.catalog is None:
            return

        # Find an "optimal" tangent plane to the catalog points based on the
        # mean point and then construct a WCS based on the mean point.
        # Compute x, y coordinates in this tangent plane based on the
        # previously computed WCS and return the set of x, y coordinates and
        # "reference WCS".
        x, y, z = _S2C(self.catalog['RA'], self.catalog['DEC'])
        ra_ref, dec_ref = _C2S(
            x.mean(dtype=np.double),
            y.mean(dtype=np.double),
            z.mean(dtype=np.double)
        )

        rotm = [planar_rot_3d(np.deg2rad(alpha), 2 - axis)
                for axis, alpha in enumerate([ra_ref, dec_ref])]
        euler_rot = np.linalg.multi_dot(rotm)
        inv_euler_rot = inv(euler_rot)
        xr, yr, zr = np.dot(euler_rot, (x, y, z))
        x = yr / xr
        y = zr / xr

        xv, yv = convex_hull(x, y)

        if len(xv) == 0:
            # no points
            raise RuntimeError(  # pragma: no cover
                "Unexpected error: Contact software developer"
            )

        elif len(xv) == 1:
            # one point. build a small box around it:
            x, y = convex_hull(x, y, wcs=None)
            tol = 0.5 * self._footprint_tol

            xv = [x[0] - tol, x[0] - tol, x[0] + tol, x[0] + tol, x[0] - tol]
            yv = [y[0] - tol, y[0] + tol, y[0] + tol, y[0] - tol, y[0] - tol]

        elif len(xv) == 2 or len(xv) == 3:
            # two points. build a small box around them:
            x, y = convex_hull(x, y, wcs=None)
            tol = 0.5 * self._footprint_tol

            vx = y[1] - y[0]
            vy = x[1] - x[0]
            norm = np.sqrt(vx * vx + vy * vy)
            vx /= norm
            vy /= norm

            xv = [
                x[0] - (vx - vy) * tol,
                x[0] - (vx + vy) * tol,
                x[1] + (vx - vy) * tol,
                x[1] + (vx + vy) * tol,
                x[0] - (vx - vy) * tol
            ]
            yv = [
                y[0] - (vy + vx) * tol,
                y[0] - (vy - vx) * tol,
                y[1] + (vy + vx) * tol,
                y[1] + (vy - vx) * tol,
                y[0] - (vy + vx) * tol
            ]

        # "unrotate" cartezian coordinates back to their original
        # ra_ref and dec_ref "positions":
        xt = np.ones_like(xv)
        xcr, ycr, zcr = np.dot(inv_euler_rot, (xt, xv, yv)).astype(np.double)
        # convert cartesian to spherical coordinates:
        ra, dec = _C2S(xcr, ycr, zcr)

        # TODO: for strange reasons, occasionally ra[0] != ra[-1] and/or
        #       dec[0] != dec[-1] (even though we close the polygon in the
        #       previous two lines). Then SphericalPolygon fails because
        #       points are not closed. Threfore we force it to be closed:
        ra[-1] = ra[0]
        dec[-1] = dec[0]

        self._radec = [(ra, dec)]
        self._polygon = SphericalPolygon.from_radec(ra, dec)
        self._poly_area = np.fabs(self._polygon.area())

    def calc_bounding_polygon(self):
        """ Calculate bounding polygon of the sources in the catalog.
        """
        # create spherical polygon bounding the sources
        self._calc_cat_convex_hull()

    def expand_catalog(self, catalog):
        """
        Expand current reference catalog with sources from another catalog.

        If current catalog is empty, then the catalog being added will become
        the new reference catalog. In this case if the ``catalog`` does have
        ``id`` column, those ID values will be preserved. If the ``catalog``
        does not contain an ID column, then the new IDs will be assigned
        in increasing order starting with 1.

        If the existing reference catalog is not empty, then the IDs from the
        ``catalog`` being added will be discarded and new IDs will be assigned
        in the increasing order such as to continue the numbering of existing
        source positions in the reference catalog.

        Parameters
        ----------
        catalog: astropy.table.Table
            A catalog of new sources to be added to the existing reference
            catalog. `catalog` *must* contain *both* ``'RA'`` and ``'DEC'``
            columns.

        """
        self._check_catalog(catalog)
        cat = catalog.copy()
        if self._catalog is None:
            self._catalog = cat
            if 'id' not in self._catalog.colnames:  # pragma: no branch
                self._catalog['id'] = np.arange(1, len(self._catalog) + 1)
        else:
            maxid = self.catalog['id'].max()
            oldlen = len(self.catalog)
            self._catalog = table.vstack(
                [self.catalog, cat],
                join_type='outer',
                metadata_conflicts='silent'
            )

            # assign ids to the newly added source positions in consecutive
            # order above the largest id in the already existing catalog:
            self._catalog['id'][oldlen:] = np.arange(maxid + 1,
                                                     maxid + len(cat) + 1)

        self.calc_bounding_polygon()

    def calc_tanp_xy(self, tanplane_wcs):
        """
        Compute x- and y-positions of the sources from the reference catalog
        in the tangent plane provided by the `tanplane_wcs`.
        This creates the following columns in the catalog's table:
        ``'TPx'`` and ``'TPy'``.

        Parameters
        ----------
        tanplane_wcs: TPWCS
            A `TPWCS` object that will provide transformations to
            the tangent plane to which sources of this catalog a should be
            "projected".

        """
        # compute x & y in the reference WCS:
        xtp, ytp = tanplane_wcs.world_to_tanp(self.catalog['RA'],
                                              self.catalog['DEC'])
        self._catalog['TPx'] = table.MaskedColumn(
            xtp, name='TPx', dtype=np.double, mask=False
        )
        self._catalog['TPy'] = table.MaskedColumn(
            ytp, name='TPy', dtype=np.double, mask=False
        )


def convex_hull(x, y, wcs=None):
    """Computes the convex hull of a set of 2D points.

    Implements `Andrew's monotone chain algorithm <http://en.wikibooks.org\
/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain>`_.
    The algorithm has O(n log n) complexity.

    Credit: `<http://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/\
Convex_hull/Monotone_chain>`_

    Parameters
    ----------

    points: list of tuples
        An iterable sequence of (x, y) pairs representing the points.

    Returns
    -------
    Output: list
        A list of vertices of the convex hull in counter-clockwise order,
        starting from the vertex with the lexicographically smallest
        coordinates.

    """
    ndarray = isinstance(x, np.ndarray) or isinstance(y, np.ndarray)

    # Sort the points lexicographically (tuples are compared
    # lexicographically). Remove duplicates to detect the case we have
    # just one unique point.
    points = sorted(set(zip(x, y)))

    # Boring case: no points or a single point,
    # possibly repeated multiple times.
    if len(points) == 0:
        if ndarray:
            return (np.array([]), np.array([]))
        else:
            return ([], [])

    elif len(points) == 1:
        if ndarray:
            return (np.array([points[0][0]]), np.array([points[0][1]]))
        else:
            return ([points[0][0]], [points[0][1]])

    # 2D cross product of OA and OB vectors, i.e. z-component of their
    # 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the
    # beginning of the other list.
    total_hull = np.asanyarray(lower[:-1] + upper)

    ptx = total_hull[:, 0]
    pty = total_hull[:, 1]

    if wcs is None:
        if ndarray:
            return (ptx, pty)
        else:
            return (ptx.tolist(), pty.tolist())

    # convert x, y vertex coordinates to RA & DEC:
    ra, dec = wcs(ptx, pty)
    ra[-1] = ra[0]
    dec[-1] = dec[0]

    if ndarray:
        return (ra, dec)
    else:
        return (ra.tolist(), dec.tolist())
