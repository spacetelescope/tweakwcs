# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides support for working with image footprints on the sky and
source catalogs.

:Authors: Mihai Cara (contact: help@stsci.edu)

:License: :doc:`../LICENSE`

"""
# STDLIB
import logging
import sys
from copy import deepcopy

# THIRD-PARTY
import numpy as np
import gwcs
from astropy import table
from spherical_geometry.polygon import SphericalPolygon
from stsci.stimage import xyxymatch

# LOCAL
from .wcsutils import cartesian2spherical, spherical2cartesian, planar_rot_3D
from .tpwcs import TPWCS
from .matchutils import TPMatch
from .linearfit import iter_linear_fit

from . import __version__, __version_date__

__author__ = 'Mihai Cara'

__all__ = ['convex_hull', 'RefCatalog', 'WCSImageCatalog', 'WCSGroupCatalog']


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


_INT_TYPE = (int, long,) if sys.version_info < (3,) else (int,)


def _is_int(n):
    return (
        (isinstance(n, _INT_TYPE) and not isinstance(n, bool)) or
        (isinstance(n, np.generic) and np.issubdtype(n, np.integer))
    )


class WCSImageCatalog(object):
    """
    A class that holds information pertinent to an image WCS and a source
    catalog of the sources found in that image.

    """
    def __init__(self, catalog, imwcs, shape=None, name=None, meta={}):
        """
        Parameters
        ----------

        catalog : astropy.table.Table
            Source catalog associated with an image. Must contain ``'x'`` and
            ``'y'`` columns which indicate source coordinates (in pixels) in
            the associated image.

        imwcs : TPWCS
            WCS associated with the image from which the catalog was derived.

        shape : tuple, None, optional
            A tuple of two integer values indicating the size of the image
            along each axis. Must follow the same convention as the shape of
            a :py:class:`numpy.ndarray` objects. Specifically,
            first size should be indicate the number of rows in the image and
            second size should indicate the number of columns in the image.
            When ``shape`` has the default value of `None`, image shape is
            estimated as :math:`(\max(y, 1), \max(x, 1))`.

        name : str, None, optional
            Image catalog's name.

        meta : dict, optional
            Additional information about image, catalog, and/or WCS to be
            stored (for convenience) within `WCSImageCatalog` object.

        """
        self._shape = shape
        self._name = name
        self._catalog = None
        self._bb_radec = None

        self.img_bounding_ra = None
        self.img_bounding_dec = None

        self.meta = {}
        self.meta.update(meta)

        self.imwcs = imwcs
        self.catalog = catalog

    @property
    def imwcs(self):
        """ Get :py:class:`TPWCS` WCS. """
        return self._imwcs

    @imwcs.setter
    def imwcs(self, imwcs):
        """ Get/Set catalog's WCS (a :py:class:`TPWCS` object).

        .. note::
            Setting the WCS triggers automatic bounding polygon recalculation.

        Parameters
        ----------

        imwcs : TPWCS
            WCS associated with the image from which the catalog was derived.

        """
        if not isinstance(imwcs, TPWCS):
            raise TypeError("Unsupported 'imwcs' type. "
                            "'imwcs' must be a subtype of TPWCS.")
        self._imwcs = imwcs

        # create spherical polygon bounding the image
        self.calc_bounding_polygon()

    @property
    def name(self):
        """ Get/set :py:class:`WCSImageCatalog` object's name.
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        if hasattr(self, '_catalog'):
            if self._catalog is not None:
                self._catalog.meta['name'] = name

    @property
    def shape(self):
        """
        Get/set image's shape. This must be a tuple of two dimensions
        following the same convention as the shape of `numpy.ndarray`.

        """
        return self._shape

    @shape.setter
    def shape(self, shape):
        if shape is None:
            self._shape = None
            return

        try:
            is_int = all(map(_is_int, shape))
            if not is_int:
                raise TypeError
        except TypeError:
            raise TypeError("'shape' must be a 1D list/tuple/array with "
                            "exactly two integer elements or None.")

        if not all(npix > 0 for npix in shape):
            raise ValueError("Null image: Image dimension must be positive.")

        self._shape = (int(shape[0]), int(shape[1]))

    @property
    def catalog(self):
        """ Get/set image's catalog.
        """
        return self._catalog

    @catalog.setter
    def catalog(self, catalog):
        if catalog is None:
            self._catalog = None
            return

        if 'x' not in catalog.colnames or 'y' not in catalog.colnames:
            raise ValueError("An image catalog must contain 'x' and 'y' "
                             "columns!")

        if len(catalog) < 1:
            raise ValueError("Image catalog must contain at least one entry.")

        self._catalog = table.Table(catalog.copy(), masked=True)
        self._catalog.meta['name'] = self._name

        if 'id' not in self._catalog.colnames:
            self._catalog['id'] = np.arange(1, len(self._catalog) + 1)

        # create spherical polygon bounding the image
        self.calc_bounding_polygon()

    def det_to_world(self, x, y):
        """
        Convert pixel coordinates to sky coordinates using full
        (i.e., including distortions) transformations.

        """
        if self._imwcs is None:
            raise RuntimeError("WCS has not been set")
        return self._imwcs.det_to_world(x, y)

    def world_to_det(self, ra, dec):
        """
        Convert sky coordinates to image's pixel coordinates using full
        (i.e., including distortions) transformations.

        """
        if self._imwcs is None:
            raise RuntimeError("WCS has not been set")
        return self._imwcs.world_to_det(ra, dec)

    def det_to_tanp(self, x, y):
        """
        Convert detector (pixel) coordinates to tangent plane coordinates.

        """
        if self._imwcs is None:
            raise RuntimeError("WCS has not been set")
        return self._imwcs.det_to_tanp(x, y)

    def tanp_to_det(self, x, y):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        if self._imwcs is None:
            raise RuntimeError("WCS has not been set")
        return self._imwcs.tanp_to_det(x, y)

    def tanp_to_world(self, x, y):
        """
        Convert tangent plane coordinates to world coordinates.

        """
        if self._imwcs is None:
            raise RuntimeError("WCS has not been set")
        return self._imwcs.tanp_to_world(x, y)

    def world_to_tanp(self, ra, dec):
        """
        Convert tangent plane coordinates to detector (pixel) coordinates.

        """
        if self._imwcs is None:
            raise RuntimeError("WCS has not been set")
        return self._imwcs.world_to_tanp(ra, dec)

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
        wcsim : WCSImageCatalog, WCSGroupCatalog, SphericalPolygon
            Another object that should be intersected with this
            `WCSImageCatalog`.

        Returns
        -------
        polygon : SphericalPolygon
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
    #def intersection_area(self, wcsim):
        #""" Calculate the area of the intersection polygon.
        #"""
        #return np.fabs(self.intersection(wcsim).area())
    def intersection_area(self, wcsim):
        """ Calculate the area of the intersection polygon.
        """
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
        stepsize : int, None, optional
            Indicates the maximum separation between two adjacent vertices
            of the bounding polygon along each side of the image. Corners
            of the image are included automatically. If `stepsize` is `None`,
            bounding polygon will contain only vertices of the image.

        """
        if self.imwcs is None or (self.shape is None and self.catalog is None):
            return

        if self.shape is None:
            nx = max(1, int(np.ceil(np.amax(self._catalog['x']))))
            ny = max(1, int(np.ceil(np.amax(self._catalog['y']))))
        else:
            ny, nx = self.shape

        if stepsize is None:
            nintx = 2
            ninty = 2
        else:
            nintx = max(2, int(np.ceil((nx + 1.0) / stepsize)))
            ninty = max(2, int(np.ceil((ny + 1.0) / stepsize)))

        xs = np.linspace(-0.5, nx - 0.5, nintx, dtype=np.float)
        ys = np.linspace(-0.5, ny - 0.5, ninty, dtype=np.float)[1:-1]
        nptx = xs.size
        npty = ys.size

        npts = 2 * (nptx + npty)

        borderx = np.empty((npts + 1,), dtype=np.float)
        bordery = np.empty((npts + 1,), dtype=np.float)

        # "bottom" points:
        borderx[:nptx] = xs
        bordery[:nptx] = -0.5
        # "right"
        sl = np.s_[nptx:nptx + npty]
        borderx[sl] = nx - 0.5
        bordery[sl] = ys
        # "top"
        sl = np.s_[nptx + npty:2 * nptx + npty]
        borderx[sl] = xs[::-1]
        bordery[sl] = ny - 0.5
        # "left"
        sl = np.s_[2 * nptx + npty:-1]
        borderx[sl] = -0.5
        bordery[sl] = ys[::-1]

        # close polygon:
        borderx[-1] = borderx[0]
        bordery[-1] = bordery[0]

        ra, dec = self.det_to_world(borderx, bordery)
        # TODO: for strange reasons, occasionally ra[0] != ra[-1] and/or
        #       dec[0] != dec[-1] (even though we close the polygon in the
        #       previous two lines). Then SphericalPolygon fails because
        #       points are not closed. Threfore we force it to be closed:
        ra[-1] = ra[0]
        dec[-1] = dec[0]

        self.img_bounding_ra = ra
        self.img_bounding_dec = dec
        self._polygon = SphericalPolygon.from_radec(ra, dec)

    def _calc_cat_convex_hull(self):
        """
        Compute convex hull that bounds the sources in the catalog.

        """
        if self.imwcs is None or self.catalog is None:
            return

        x = self.catalog['x']
        y = self.catalog['y']

        if len(x) == 0:
            # no points
            raise RuntimeError("Unexpected error: Contact software developer")

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
        if self._catalog is not None and len(self.catalog) > 0:
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
        images : list of WCSImageCatalog
            A list of `WCSImageCatalog` image catalogs.

        name : str, None, optional
            Name of the group.

        """

        self._catalog = None

        if isinstance(images, WCSImageCatalog):
            self._images = [images]

        elif hasattr(images, '__iter__'):
            self._images = []
            for im in images:
                if not isinstance(im, WCSImageCatalog):
                    raise TypeError("Each element of the 'images' parameter "
                                    "must be an 'WCSImageCatalog' object.")
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
        wcsim : WCSImageCatalog, WCSGroupCatalog, SphericalPolygon
            Another object that should be intersected with this
            `WCSGroupCatalog`.

        Returns
        -------
        polygon : SphericalPolygon
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
    #def intersection_area(self, wcsim):
        #""" Calculate the area of the intersection polygon.
        #"""
        #return np.fabs(self.intersection(wcsim).area())
    def intersection_area(self, wcsim):
        """ Calculate the area of the intersection polygon.
        """
        area = 0.0
        for im in self._images:
            area += im.intersection_area(wcsim)
        return area

    def update_bounding_polygon(self):
        """ Recompute bounding polygons of the member images.
        """
        polygons = [im.polygon for im in self._images]
        if len(polygons) == 0:
            self._polygon = SphericalPolygon([])
        else:
            self._polygon = SphericalPolygon.multi_union(polygons)

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

        group_catalog : astropy.table.Table
            Combined group catalog.

        """
        catalogs = []
        catno = 0
        for image in self._images:
            catlen = len(image.catalog)

            if image.name is None:
                catname = 'Catalog #{:d}'.format(catno)
            else:
                catname = image.name

            col_catname = table.MaskedColumn(catlen * [catname],
                                             name='cat_name')
            col_imcatidx = table.MaskedColumn(catlen * [catno],
                                              name='_imcat_idx')
            col_id = table.MaskedColumn(image.catalog['id'])
            col_x = table.MaskedColumn(image.catalog['x'], dtype=np.float64)
            col_y = table.MaskedColumn(image.catalog['y'], dtype=np.float64)
            ra, dec = image.det_to_world(
                image.catalog['x'], image.catalog['y']
            )
            col_ra = table.MaskedColumn(ra, dtype=np.float64, name='RA')
            col_dec = table.MaskedColumn(dec, dtype=np.float64, name='DEC')

            cat = table.Table(
                [col_imcatidx, col_catname, col_id, col_x,
                 col_y, col_ra, col_dec],
                masked=True
            )

            catalogs.append(cat)
            catno += 1

        return table.vstack(catalogs, join_type='exact')

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
        tanplane_wcs : ImageGWCS
            A `ImageGWCS` object that will provide transformations to
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
            xtp, name='TPx', dtype=np.float64, mask=False
        )
        self._catalog['TPy'] = table.MaskedColumn(
            ytp, name='TPy', dtype=np.float64, mask=False
        )

    def match2ref(self, refcat, match=None):
        """ Uses ``xyxymatch`` to cross-match sources between this catalog and
            a reference catalog.

        Parameters
        ----------
        refcat : RefCatalog
            A `RefCatalog` object that contains a catalog of reference sources
            as well as a valid reference WCS.

        match : MatchCatalogs, function, None, optional
            A callable that takes two arguments: a reference catalog and an
            image catalog. Both catalogs will have columns ``'TPx'`` and
            ``'TPy'`` that represent the source coordinates in some common
            (to both catalogs) coordinate system.

        """
        colnames = self._catalog.colnames
        catlen = len(self._catalog)

        if match is None:
            if catlen != len(self._catalog):
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
        if 'matched_ref_id' not in colnames:
            c = table.MaskedColumn(name='matched_ref_id', dtype=int,
                                   length=catlen, mask=True)
            self._catalog.add_column(c)
        else:
            self._catalog['matched_ref_id'].mask[:] = True

        self._catalog['matched_ref_id'].mask[minput_idx] = False
        self._catalog['matched_ref_id'][minput_idx] = refcat.catalog['id'][mref_idx]

        # this is needed to index reference catalog directly without using
        # astropy table indexing which, at this moment, is experimental:
        if '_raw_matched_ref_idx' not in colnames:
            c = table.MaskedColumn(name='_raw_matched_ref_idx',
                                   dtype=int, length=catlen, mask=True)
            self._catalog.add_column(c)
        else:
            self._catalog['_raw_matched_ref_idx'].mask = True
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
                sigma=3.0):
        """
        Perform linear fit of this group's combined catalog to the reference
        catalog.


        Parameters
        ----------

        refcat : RefCatalog
            A `RefCatalog` object that contains a catalog of reference sources.

        tanplane_wcs : ImageGWCS
            A `ImageGWCS` object that will provide transformations to
            the tangent plane to which sources of this catalog a should be
            "projected".

        fitgeom : {'shift', 'rscale', 'general'}, optional
            The fitting geometry to be used in fitting the matched object
            lists. This parameter is used in fitting the offsets, rotations
            and/or scale changes from the matched object lists. The 'general'
            fit geometry allows for independent scale and rotation for
            each axis.

        nclip : int, optional
            Number (a non-negative integer) of clipping iterations in fit.

        sigma : float, optional
            Clipping limit in sigma units.

        """
        im_xyref = np.asanyarray([self._catalog['TPx'],
                                  self._catalog['TPy']]).T
        refxy = np.asanyarray([refcat.catalog['TPx'],
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

        fit = iter_linear_fit(
            refxy, im_xyref, xyindx=ref_idx, uvindx=minput_idx, fitgeom=fitgeom,
            nclip=nclip, sigma=sigma, center=(0, 0)
        )

        xy_fit = fit['img_coords'] + fit['resids']
        fit['fit_xy'] = xy_fit
        fit['fit_RA'], fit['fit_DEC'] = tanplane_wcs.tanp_to_world(*(xy_fit.T))

        log.info("Computed '{:s}' fit for {}:".format(fitgeom, self.name))
        if fitgeom == 'shift':
            log.info("XSH: {:.6g}  YSH: {:.6g}"
                     .format(fit['offset'][0], fit['offset'][1]))
        elif fitgeom == 'rscale' and fit['proper']:
            log.info("XSH: {:.6g}  YSH: {:.6g}    ROT: {:.6g}    SCALE: {:.6g}"
                     .format(fit['offset'][0], fit['offset'][1],
                             fit['rot'], fit['scale'][0]))
        elif fitgeom == 'general' or (fitgeom == 'rscale' and not
                                      fit['proper']):
            log.info("XSH: {:.6g}  YSH: {:.6g}    PROPER ROT: {:.6g}    "
                     .format(fit['offset'][0], fit['offset'][1], fit['rot']))
            log.info("<ROT>: {:.6g}  SKEW: {:.6g}    ROT_X: {:.6g}  "
                     "ROT_Y: {:.6g}".format(fit['rotxy'][2], fit['skew'],
                                            fit['rotxy'][0], fit['rotxy'][1]))
            log.info("<SCALE>: {:.6g}  SCALE_X: {:.6g}  SCALE_Y: {:.6g}"
                     .format(fit['scale'][0], fit['scale'][1],
                             fit['scale'][2]))
        else:
            raise ValueError("Unsupported fit geometry.")

        log.info("")
        log.info("XRMS: {:.6g}    YRMS: {:.6g}".format(fit['rms'][0],
                                                       fit['rms'][1]))
        log.info("Final solution based on {:d} objects."
                 .format(fit['resids'].shape[0]))

        return fit

    def apply_affine_to_wcs(self, tanplane_wcs, matrix, shift, meta=None):
        """ Applies a general affine transformation to the WCS.
        """
        # compute the matrix for the scale and rotation correction
        matrix = matrix.T
        shift = -np.dot(np.linalg.inv(matrix), shift)

        for imcat in self:
            # compute linear transformation from the tangent plane used for
            # alignment to the tangent plane of another image in the group:
            if imcat.imwcs == tanplane_wcs:
                m = matrix.copy()
                s = shift.copy()
            else:
                r1, t1 = _tp2tp(imcat.imwcs, tanplane_wcs)
                r2, t2 = _tp2tp(tanplane_wcs, imcat.imwcs)
                m = np.linalg.multi_dot([r2, matrix, r1])
                s = t1 + np.dot(np.linalg.inv(r1), shift) + \
                    np.dot(np.linalg.inv(np.dot(matrix, r1)), t2)

            imcat.imwcs.set_correction(m, s, meta=meta)

    def align_to_ref(self, refcat, match=None, minobj=None,
                     fitgeom='rscale', nclip=3, sigma=3.0):
        """
        Matches sources from the image catalog to the sources in the
        reference catalog, finds the affine transformation between matched
        sources, and adjusts images' WCS according to this fit.

        Upon successful return, this function will also set the following
        fields of the ``meta`` attribute of the tangent-plane ``WCS``
        (a `TPWCS`-derived object) of each member `WCSImageCatalog` object:

            * **'fitgeom'**: the value of the ``fitgeom`` argument
            * **'matrix'**: computed rotation matrix
            * **'shift'**: offset along X- and Y-axis
            * **'eff_minobj'**: effective value of the ``minobj`` parameter
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
            * **'rot'**: rotation angle as if rotation is a proper rotation
            * **'proper'**: Indicates whether the rotation is a proper rotation
              (boolean)
            * **'rotxy'**: a tuple of (rotation of the X-axis, rotation of the
              Y-axis, mean rotation, computed skew)
            * **'scale'**: a tuple of (mean scale, scale along X-axis, scale
              along Y-axis)
            * **'skew'**: computed skew
            * **'rms'**: fit RMS in *image* coordinates as a tuple of two
              values: (RMS_X, RMS_Y)
            * **'status'**: Alignment status. Currently two possible status are
              possible ``'SUCCESS'`` or ``'FAILED: reason for failure'``.
              When alignment failed, the reason for failure is provided after
              alignment status.

        .. note::
            A ``'SUCCESS'`` status does not indicate a "good" alignment. It
            simply indicates that alignment algortithm has completed without
            errors. Use other fields to evaluate alignment: residual RMS
            values, number of matched sources, etc.


        Parameters
        ----------

        refcat : RefCatalog
            A `RefCatalog` object that contains a catalog of reference sources
            as well as a valid reference WCS.

        match : MatchCatalogs, function, None, optional
            A callable that takes two arguments: a reference catalog and an
            image catalog.

        minobj : int, None, optional
            Minimum number of identified objects from each input image to use
            in matching objects from other images. If the default `None` value
            is used then `align` will automatically deternmine the minimum
            number of sources from the value of the `fitgeom` parameter.
            This parameter is used to interrupt alignment process (catalog
            fitting, ``WCS`` "tweaking") when the number of matched sources
            is smaller than the value of ``minobj`` in which case this
            method will return `False`.

        fitgeom : {'shift', 'rscale', 'general'}, optional
            The fitting geometry to be used in fitting the matched object
            lists. This parameter is used in fitting the offsets, rotations
            and/or scale changes from the matched object lists. The 'general'
            fit geometry allows for independent scale and rotation for each
            axis. This parameter is ignored if ``match`` is `False`.

        nclip : int, optional
            Number (a non-negative integer) of clipping iterations in fit.
            This parameter is ignored if ``match`` is `False`.

        sigma : float, optional
            Clipping limit in sigma units. This parameter is ignored if
            ``match`` is `False`.


        Returns
        -------

        bool
            Returns `True` if the number of matched sources is larger or equal
            to ``minobj`` and all steps have been performed, including catalog
            fitting and ``WCS`` "tweaking". If the number of matched sources is
            smaller than ``minobj``, this function will return `False`.

        """
        if len(self._images) == 0:
            name = 'Unnamed' if self.name is None else self.name
            log.warning("WCSGroupCatalog '{:s}' is empty. Nothing to align."
                        .format(name))
            return False

        # set initial status to 'FAILED':
        for imcat in self:
            imcat.imwcs.meta['status'] = "FAILED: Unknown error"

        if minobj is None:
            if fitgeom == 'general':
                minobj = 3
            elif fitgeom == 'rscale':
                minobj = 2
            else:
                minobj = 1

        tanplane_wcs = deepcopy(self._images[0].imwcs)

        self.calc_tanp_xy(tanplane_wcs=tanplane_wcs)
        refcat.calc_tanp_xy(tanplane_wcs=tanplane_wcs)

        nmatches, mref_idx, minput_idx = self.match2ref(
            refcat=refcat,
            match=match
        )

        if nmatches < minobj:
            name = 'Unnamed' if self.name is None else self.name
            log.warning("Not enough matches (< {:d}) found for image "
                        "catalog '{:s}'.".format(nmatches, name))
            for imcat in self:
                imcat.imwcs.meta['status'] = 'FAILED: not enough matches'
            return False

        fit = self.fit2ref(refcat=refcat, tanplane_wcs=tanplane_wcs,
                           fitgeom=fitgeom, nclip=nclip, sigma=sigma)

        meta = {
            'fitgeom' : fitgeom,
            'matrix': fit['fit_matrix'],
            'shift': fit['offset'],
            'eff_minobj': minobj,
            'fit_ref_idx': fit['ref_indx'],  # indices after fit clipping
            'fit_input_idx': fit['img_indx'],  # indices after fit clipping
            'rot': fit['rot'],  # proper rotation
            'proper': fit['proper'],  # is a proper rotation? True/False
            'rotxy': fit['rotxy'],  # rotx, roty, <rot>, skew
            'scale': fit['scale'],  # <s>, sx, sy
            'skew': fit['skew'],  # skew
            'rms': fit['rms'],  # fit RMS in image coords (RMS_X, RMS_Y)
            'status': 'SUCCESS'
        }

        if match is not None:
            meta.update({
                'nmatches': nmatches,
                'matched_ref_idx': mref_idx,
                'matched_input_idx': minput_idx
            })

        self.apply_affine_to_wcs(
            tanplane_wcs=tanplane_wcs,
            matrix=fit['fit_matrix'],
            shift=fit['offset'],
            meta=meta
        )

        self.recalc_catalog_radec()

        return True


def _tp2tp(imwcs1, imwcs2):
    x = np.array([0.0, 1.0, 0.0], dtype=np.float)
    y = np.array([0.0, 0.0, 1.0], dtype=np.float)
    xrp, yrp = imwcs2.world_to_tanp(*imwcs1.tanp_to_world(x, y))

    matrix = np.array([(xrp[1:] - xrp[0]), (yrp[1:] - yrp[0])])
    shift = -np.dot(np.linalg.inv(matrix), [xrp[0], yrp[0]])

    return matrix, shift


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
        catalog : astropy.table.Table
            Reference catalog.

            ..note::
                Reference catalogs (:py:class:`~astropy.table.Table`)
                *must* contain *both* ``'RA'`` and ``'DEC'`` columns.

        name : str, None, optional
            Name of the reference catalog.

        footprint_tol : float, optional
            Matching tolerance in arcsec. This is used to estimate catalog's
            footprint when catalog contains only one or two sources.

        """
        self._name = name
        self._catalog = None
        self._footprint_tol = footprint_tol

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
        """ Get/set :py:class:`WCSImageCatalog` object's name.
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

        if len(catalog) == 0:
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
        wcsim : WCSImageCatalog, WCSGroupCatalog, RefCatalog, SphericalPolygon
            Another object that should be intersected with this
            `WCSImageCatalog`.

        Returns
        -------
        polygon : SphericalPolygon
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
    #def intersection_area(self, wcsim):
        #""" Calculate the area of the intersection polygon.
        #"""
        #return np.fabs(self.intersection(wcsim).area())
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
        x, y, z = spherical2cartesian(
            self.catalog['RA'], self.catalog['DEC']
        )
        ra_ref, dec_ref = cartesian2spherical(
            x.mean(dtype=np.float64),
            y.mean(dtype=np.float64),
            z.mean(dtype=np.float64)
        )
        rotm = [planar_rot_3D(np.deg2rad(alpha), 2 - axis)
                for axis, alpha in enumerate([ra_ref, dec_ref])]
        euler_rot = np.linalg.multi_dot(rotm)
        inv_euler_rot = np.linalg.inv(euler_rot)
        xr, yr, zr = np.dot(euler_rot, (x, y, z))
        x = yr / xr
        y = zr / xr

        xv, yv = convex_hull(x, y)

        if len(xv) == 0:
            # no points
            raise RuntimeError("Unexpected error: Contact software developer")

        elif len(xv) == 1:
            # one point. build a small box around it:
            x, y = convex_hull(x, y, wcs=None)
            tol = 0.5 * self._footprint_tol

            xv = [x[0] - tol, x[0] - tol, x[0] + tol, x[0] + tol, x[0] - tol]
            yv = [y[0] - tol, y[0] + tol, y[0] + tol, y[0] - tol, y[0] - tol]

        elif len(xv) == 2:
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
        xcr, ycr, zcr = np.dot(inv_euler_rot, (xt, xv, yv))
        # convert cartesian to spherical coordinates:
        ra, dec = cartesian2spherical(xcr, ycr, zcr)

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

        Parameters
        ----------
        catalog : astropy.table.Table
            A catalog of new sources to be added to the existing reference
            catalog. `catalog` *must* contain *both* ``'RA'`` and ``'DEC'``
            columns.

        """
        self._check_catalog(catalog)
        cat = catalog.copy()
        if self._catalog is None:
            self._catalog = cat
            if 'id' not in self._catalog.colnames:
                self._catalog['id'] = np.arange(1, len(self._catalog) + 1)
        else:
            self._catalog = table.vstack([self.catalog, cat],
                                         join_type='outer')
            # overwrite source ID since when expanding the catalog,
            # there could be duplicates in source ID:
            self._catalog['id'] = np.arange(1, len(self._catalog) + 1)

        self.calc_bounding_polygon()

    def calc_tanp_xy(self, tanplane_wcs):
        """
        Compute x- and y-positions of the sources from the reference catalog
        in the tangent plane provided by the `tanplane_wcs`.
        This creates the following columns in the catalog's table:
        ``'TPx'`` and ``'TPy'``.

        Parameters
        ----------
        tanplane_wcs : ImageGWCS
            A `ImageGWCS` object that will provide transformations to
            the tangent plane to which sources of this catalog a should be
            "projected".

        """
        # compute x & y in the reference WCS:
        xtp, ytp = tanplane_wcs.world_to_tanp(self.catalog['RA'],
                                              self.catalog['DEC'])
        self._catalog['TPx'] = table.MaskedColumn(
            xtp, name='TPx', dtype=np.float64, mask=False
        )
        self._catalog['TPy'] = table.MaskedColumn(
            ytp, name='TPy', dtype=np.float64, mask=False
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

    points : list of tuples
        An iterable sequence of (x, y) pairs representing the points.

    Returns
    -------
    Output : list
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
        if not ndarray:
            return (np.array([]), np.array([]))
        else:
            return ([], [])
    elif len(points) == 1:
        if not ndarray:
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
        if not ndarray:
            return (ptx.tolist(), pty.tolist())
        else:
            return (ptx, pty)

    # convert x, y vertex coordinates to RA & DEC:
    ra, dec = wcs(ptx, pty)
    ra[-1] = ra[0]
    dec[-1] = dec[0]

    if not ndarray:
        return (ra.tolist(), dec.tolist())
    else:
        return (ra, dec)
