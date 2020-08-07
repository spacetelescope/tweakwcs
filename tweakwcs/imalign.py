# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A module that provides functions for "aligning" images: specifically, it
provides functions for computing corrections to image ``WCS`` so that
image catalogs "align" to the reference catalog *on the sky*.

:Authors: Mihai Cara

:License: :doc:`../LICENSE`

"""
# STDLIB
import logging
from datetime import datetime
import collections

# THIRD PARTY
import numpy as np
import astropy

# LOCAL
from . wcsimage import RefCatalog, WCSImageCatalog, WCSGroupCatalog
from . tpwcs import TPWCS
from . matchutils import TPMatch

from . import __version__

__author__ = 'Mihai Cara'

__all__ = ['fit_wcs', 'align_wcs']


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def fit_wcs(refcat, imcat, tpwcs, ref_tpwcs=None, fitgeom='general', nclip=3,
            sigma=(3.0, 'rmse')):
    """ "Tweak" **a single** image's ``WCS`` by fitting image catalog to a
    reference catalog. This is a simplified version of `align_wcs` that does
    not perform matching and is limited to the fitting part.

    .. note::
        Both reference and image catalogs must have been **matched**
        *prior to* calling ``fit_wcs()``. This means that the lengths of both
        ``refcat`` and ``imcat`` catalogs must be equal *and* that coordinates
        with the same indices in both catalogs correspond to the same source.

    .. warning::
        If ``tpwcs.meta`` dictionary contains ``'catalog'`` keyword,
        it will be ignored.

    Parameters
    ----------
    refcat: astropy.table.Table
        A reference source catalog. The catalog must contain ``'RA'`` and
        ``'DEC'`` columns which indicate reference source world
        coordinates (in degrees). An optional column in the catalog is
        the ``'weight'`` column, which when present, will be used in fitting.
        See ``Notes`` section for further details.

    imcat: astropy.table.Table
        Source catalog associated with an image whose WCS needs to be aligned
        by fitting a linear transformation to ``imcat`` source positions so as
        to align them to the same sources from the ``refcat`` catalog.
        Must contain ``'x'`` and ``'y'`` columns which indicate source
        coordinates (in pixels) in the associated image. An optional column in
        the catalog is the ``'weight'`` column, which when present, will be
        used in fitting. See ``Notes`` section for further details.

    tpwcs: TPWCS
        A ``WCS`` associated with the image from which the catalog was derived.
        This ``TPWCS``-subclassed WCS corrector object must also define
        a tangent plane that will be used for fitting the two catalogs'
        sources and in which WCS corrections will be applied.

    ref_tpwcs: TPWCS, None, optional
        A reference WCS of the type ``TPWCS`` that provides the tangent
        plane in which matching will be performed and corrections will be
        defined. When not provided (i.e., set to `None`), reference tangent
        plane will be the same as defined by ``tpwcs`` argument.

    fitgeom: {'shift', 'rscale', 'general'}, optional
        The fitting geometry to be used in fitting the matched object lists.
        This parameter is used in fitting the offsets, rotations and/or scale
        changes from the matched object lists. The 'general' fit geometry
        allows for independent scale and rotation for each axis.

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

    Returns
    -------
    twwcs: TPWCS
        "Tweaked" (aligned) ``WCS`` that contains tangent-plane corrections
        so that reference and image catalog sources better align in the tangent
        plane and therefore on the sky as well.

    Notes
    -----
    When fitting image sources to reference catalog sources, we can specify
    which sources have higher weights. This can be done by assigning a "weight"
    to each source by specifying these values in the optional ``'weight'``
    column of either the reference catalog, image catalog, or both.

    When weights are not provided, all sources are weighed equally. When
    only either image or reference catalog weights are provided, the sources
    will be weighted with the specified weights. When *both* image *and*
    reference catalogs specify weights for the same sources, the two weights
    will be combined into a single weight as:

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

    Upon **successful** completion, this function will set the ``'fit_info'``
    key value of the ``meta`` attribute of the returned ``TPWCS`` object.
    ``'fit_info'`` is a dictionary with the following items:

        * **'shift'**: A ``numpy.ndarray`` with two components of the
          computed shift.

        * **'matrix'**: A ``2x2`` ``numpy.ndarray`` with the computed
          generalized rotation matrix.

        * **'proper_rot'**: Rotation angle (degree) as if the rotation is
          proper.

        * **'rot'**: A tuple of ``(rotx, roty)`` - the rotation angles with
          regard to the ``X`` and ``Y`` axes.

        * **'<rot>'**: *Arithmetic mean* of the angles of rotation around
          ``X`` and ``Y`` axes.

        * **'scale'**: A tuple of ``(sx, sy)`` - scale change in the direction
          of the ``X`` and ``Y`` axes.

        * **'<scale>'**: *Geometric mean* of scales ``sx`` and ``sy``.

        * **'skew'**: Computed skew.

        * **'proper'**: a boolean indicating whether the rotation is proper.

        * **'fitgeom'**: Fit geometry (allowed transformations) used for
          fitting data (to minimize residuals). This is copy of the input
          argument ``fitgeom``.

        * **'center'**: Center of rotation in the *tangent plane* of the
          computed linear transformations.

        * **'fitmask'**: A boolean array indicating which source positions
          where used for fitting (`True`) and which were clipped out
          (`False`). **NOTE:** For weighted fits, positions with zero
          weights are automatically excluded from the fits.

        * **'eff_nclip'**: Effective number of clipping iterations

        * **'rmse'**: fit Root-Mean-Square Error in *tangent plane*
          coordinates of corrected image source positions from reference
          source positions.

        * **'mae'**: fit Mean Absolute Error in *tangent plane*
          coordinates of corrected image source positions from reference
          source positions.

        * **'std'**: Norm of the standard deviation of the residuals
          in *tangent plane* along each axis.

        * **'resids'**: An array of residuals of the fit in the
          *tangent plane*.

          **NOTE:** Only the residuals for the "valid" points are reported
          here. Therefore the length of this array may be smaller than the
          length of input arrays of positions.

        * **'fit_RA'**: first (corrected) world coordinate of input source
          positions used in fitting.

        * **'fit_DEC'**: second (corrected) world coordinate of input
          source positions used in fitting.

        * **'status'**: Alignment status. Currently two possible status are
          possible ``'SUCCESS'`` or ``'FAILED: reason for failure'``.
          When alignment failed, the reason for failure is provided after
          alignment status.

    """
    function_name = fit_wcs.__name__

    # Time it
    runtime_begin = datetime.now()

    log.info(" ")
    log.info("***** {:s}.{:s}() started on {}"
             .format(__name__, function_name, runtime_begin))
    log.info("      Version {}".format(__version__))
    log.info(" ")

    try:
        # Attempt to set initial status to FAILED.
        tpwcs.meta['fit_info'] = {'status': 'FAILED: Unknown error'}
    except Exception:
        raise AttributeError("Unable to set/modify tpwcs.meta attribute.")

    # check fitgeom:
    fitgeom = fitgeom.lower()
    if fitgeom not in ['shift', 'rscale', 'general']:
        raise ValueError("Unsupported 'fitgeom'. Valid values are: "
                         "'shift', 'rscale', or 'general'")

    wimcat = WCSImageCatalog(imcat, tpwcs,
                             name=imcat.meta.get('name', 'Unnamed'))
    wgcat = WCSGroupCatalog(wimcat, name=imcat.meta.get('name', 'Unnamed'))
    wrefcat = RefCatalog(refcat, name=imcat.meta.get('name', 'Unnamed'))

    succes = wgcat.align_to_ref(
        refcat=wrefcat,
        ref_tpwcs=ref_tpwcs,
        match=None,
        minobj=None,
        fitgeom=fitgeom,
        nclip=nclip,
        sigma=sigma
    )

    tpwcs.meta['fit_info'] = wimcat.fit_info
    if not succes:
        log.warning("Failed to align catalog '{}'.".format(wgcat.name))

    # log running time:
    runtime_end = datetime.now()
    log.info(" ")
    log.info("***** {:s}.{:s}() ended on {}"
             .format(__name__, function_name, runtime_end))
    log.info("***** {:s}.{:s}() TOTAL RUN TIME: {}"
             .format(__name__, function_name, runtime_end - runtime_begin))
    log.info(" ")

    return wgcat[0].tpwcs


def align_wcs(wcscat, refcat=None, ref_tpwcs=None, enforce_user_order=True,
              expand_refcat=False, minobj=None, match=TPMatch(),
              fitgeom='general', nclip=3, sigma=(3.0, 'rmse')):
    r"""
    Align (groups of) image catalogs by adjusting the parameters of their
    WCS based on fits between matched sources in these catalogs and a reference
    catalog which may be automatically created from one of the input ``wcscat``
    catalogs.

    .. warning::
        This function modifies the ``wcs`` attribute of each item
        in the input ``wcscat`` list!

    Upon completion, this function will add a field ``'fit_info'``
    to the ``meta`` attribute of the input WCS correctors (except of the one
    chosen as a reference catalog when ``refcat`` is `None`) containing
    a dictionary describing matching and fit results. For a description
    of the items in this dictionary, see
    :meth:`tweakwcs.wcsimage.WCSGroupCatalog.align_to_ref`. In addition to the
    status set by :meth:`~tweakwcs.wcsimage.WCSGroupCatalog.align_to_ref`,
    this function may set status to ``'REFERENCE'`` for an input image used
    as a reference image when a reference catalog is not provided.
    In this case no other fields in the ``'fit_info'`` will be present
    because a reference image is not being aligned. When alignment failed,
    the reason for failure is provided after alignment status.

    .. warning::

        Unless status in ``'fit_info'`` is ``'SUCCESS'``, there is no
        guarantee that other fields in ``'fit_info'`` are present or
        valid. Therefore, it is advisable verify that status is ``'SUCCESS'``
        before attempting to access other items, for example:

        >>> fit_info = wcscat[0].meta.get('fit_info')  # noqa
        >>> if fit_info['status'] == 'SUCCESS':
        ...     print("shifts: [{}, {}]".format(*fit_info['shift']))
        ... else:
        ...     print("tweak info not available for this image")

    Parameters
    ----------
    wcscat: tweakwcs.tpwcs.TPWCS, list of tweakwcs.tpwcs.TPWCS
        A list of all `~tweakwcs.tpwcs.TPWCS`-derived WCS correctors whose
        ``meta`` dictionary **must** contain ``'catalog'``
        item with a non-empty table value of type `astropy.table.Table`.
        This catalog must contain ``'x'`` and ``'y'`` columns which indicate
        source coordinates (in pixels) in the associated image. An optional
        column in the catalog is the ``'weight'`` column, which when present,
        will be used in fitting. See ``Notes`` section for further details.
        In addition to ``'catalog'``, the following items in the ``meta``
        dictionary are recognized/supported: ``'name'`` and ``'group_id'``.
        ``'name'`` is catalog's name and it used to identify catalog during
        logging. If ``'name'`` value is `None` or not present at all in the
        ``meta`` of a catalog, the name of that catalog will reported as
        ``'Unknown'``. Group ID that may be used for identifying catalogs
        that need to be aligned together. ``group_id`` must be hashable.
        If ``'group_id'`` is `None` or not provided, each input WCS/catalog
        will be aligned individually.

        .. note::
            Upon completion this function will add ``'fit_info'``
            item (a dictionary) to input object's ``meta`` dictionary.
            See **Notes** section for more details.

        .. warning::
            This function modifies the WCS of ``TPWCS`` objects by calling
            their :py:meth:`~tweakwcs.tpwcs.TPWCS.set_correction` method.

    refcat: astropy.table.Table, optional
        A reference source catalog. The catalog must contain ``'RA'`` and
        ``'DEC'`` columns which indicate reference source world
        coordinates (in degrees). An optional column in the catalog is
        the ``'weight'`` column, which when present, will be used in fitting.
        See ``Notes`` section for further details.

    ref_tpwcs: TPWCS, None, optional
        A reference WCS of the type ``TPWCS`` that provides the tangent
        plane in which matching will be performed and corrections will be
        defined. When not provided (i.e., set to `None`), reference tangent
        plane will be defined from the first ``TPWCS`` object
        *in the re-ordered* (if ``enforce_user_order`` was
        set to `True`) input list ``wcscat``.

    enforce_user_order: bool, optional
        Specifies whether images should be aligned in the order specified in
        the `file` input parameter or `align` should optimize the order
        of alignment by intersection area of the images. Default value (`True`)
        will align images in the user specified order, except when some images
        cannot be aligned in which case `align` will optimize the image
        alignment order. Alignment order optimization is available *only*
        when ``expand_refcat`` is `True`.

    expand_refcat: bool, optional
        Specifies whether to add new sources from just matched images to
        the reference catalog to allow next image to be matched against an
        expanded reference catalog. By delault, the reference catalog is not
        being expanded.

        If ``refcat`` is not `None` and contains an ``'id'`` column, then
        sources being added to the reference catalog will be assigned
        consecutive IDs that continue maximum ID in the ``refcat``.

        If one desires to uniquely associate source in the expanded catalog
        to their original catalogs, it is recommended that one assign unique
        IDs to all sources in all input catalogs **and** in the reference
        catalog in a separate column such as ``'uuid'``.

    minobj: int, None, optional
        Minimum number of identified objects from each input image to use
        in matching objects from other images. If the default `None` value is
        used then `align` will automatically deternmine the minimum number
        of sources from the value of the ``fitgeom`` parameter.

    match: MatchCatalogs, function, None, optional
        A callable that takes two arguments: a reference catalog and an
        image catalog. Both catalogs will have columns ``'TPx'`` and
        ``'TPy'`` that represent the source coordinates in some common
        (to both catalogs) coordinate system.

    fitgeom: {'shift', 'rscale', 'general'}, optional
        The fitting geometry to be used in fitting the matched object lists.
        This parameter is used in fitting the offsets, rotations and/or scale
        changes from the matched object lists. The 'general' fit geometry
        allows for independent scale and rotation for each axis.

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

    Returns
    -------
    eff_refcat: astropy.table.Table
        Effective reference catalog used for aligning all images. Depending
        on the values of the input parameters ``refcat``,
        ``enforce_user_order``, and ``expand_refcat``, effective
        reference catalog may be one of the input image catalogs, the original
        ``refcat`` catalog, an expanded ``refcat`` with a combination of
        source positions from all input images.

    Notes
    -----
    **1. Weights:**

    When fitting image sources to reference catalog sources, we can specify
    which sources have higher weights. This can be done by assigning a "weight"
    to each source by specifying these values in the optional ``'weight'``
    column of either the reference catalog, image catalog, or both.

    When weights are not provided, all sources are weighed equally. When
    only either image or reference catalog weights are provided, the sources
    will be weighted with the specified weights. When *both* image *and*
    reference catalogs specify weights for the same sources, the two weights
    will be combined into a single weight as:

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

    .. warning::
        When image catalogs contain optional ``'weight'`` column, then
        all image catalogs in a group must contain this column.

    **2.** ``'fit_info'``:

    Upon completion, this function will add ``'fit_info'``
    item (itself a dictionary) to input object's ``meta`` dictionary.
    If input objects are `~tweakwcs.tpwcs.TPWCS` WCS correctors, then
    ``TPWCS.meta['fit_info']`` will be set to a dictionary containing
    fit information.

    .. note::
        For `~tweakwcs.tpwcs.TPWCS` that are aligned in a group,
        the ``'matrix'`` and ``'shift'`` items in the ``'fit_info'``
        dictionary may differ from
        the values of the same items in ``TPWCS.meta`` dictionary. This is
        normal since WCS corrections (in `~tweakwcs.tpwcs.TPWCS`) are applied
        in the image's WCS plane while fit may be performed in a slightly
        different tangent plane.

    """
    function_name = align_wcs.__name__

    # Time it
    runtime_begin = datetime.now()

    log.info(" ")
    log.info("***** {:s}.{:s}() started on {}"
             .format(__name__, function_name, runtime_begin))
    log.info("      Version {}".format(__version__))
    log.info(" ")

    # Check that type of `wcscat` is correct and set initial status to FAILED:
    if isinstance(wcscat, TPWCS):
        wcscat = [wcscat]
        start = 1
    else:
        start = 0

    if not (hasattr(wcscat, '__iter__') and all(isinstance(wcat, TPWCS)
                                                for wcat in wcscat[start:])):
        raise TypeError("Input 'wcscat' must be either a single TPWCS-derived "
                        " object or a list of TPWCS-derived objects.")

    wcs_im_cats = []
    for wcat in wcscat:
        if wcat.meta.get('catalog', None) is None:
            raise ValueError("Each object in 'wcscat' must have a valid "
                             "catalog.")

        wcs_im_cat = WCSImageCatalog(
            catalog=wcat.meta['catalog'],
            tpwcs=wcat,
            name=wcat.meta.get('name', 'Unknown'),
            group_id=wcat.meta.get('group_id', None)
        )
        wcs_im_cat.fit_status = "FAILED: Unknown error"

        wcs_im_cats.append(wcs_im_cat)

    # check fitgeom:
    fitgeom = fitgeom.lower()
    if fitgeom not in ['shift', 'rscale', 'general']:
        raise ValueError("Unsupported 'fitgeom'. Valid values are: "
                         "'shift', 'rscale', or 'general'")

    if minobj is None:  # pragma: no branch
        if fitgeom == 'general':
            minobj = 3
        elif fitgeom == 'rscale':
            minobj = 2
        else:
            minobj = 1

    # process reference catalog or image if provided:
    if refcat is not None:
        if isinstance(refcat, TPWCS):
            if 'catalog' not in refcat.meta:
                raise ValueError("Reference 'TPWCS' must contain a "
                                 "catalog.")

            rcat = refcat.meta['catalog'].copy()

            if not ('RA' in rcat.colnames and 'DEC' in rcat.colnames):  # pragma: no branch
                # convert image x & y to world coordinates:
                ra, dec = refcat.det_to_world(rcat['x'], rcat['y'])
                rcat['RA'] = ra
                rcat['DEC'] = dec

            refcat_name = refcat.meta.get(
                'name', rcat.meta.get('name', 'Unnamed')
            )

            refcat = RefCatalog(rcat, name=refcat_name)

        elif isinstance(refcat, astropy.table.Table):
            if 'RA' not in refcat.colnames or 'DEC' not in refcat.colnames:
                raise KeyError("Reference catalogs *must* contain *both* 'RA' "
                               "and 'DEC' columns.")
            refcat = RefCatalog(
                refcat, name=refcat.meta.get('name', 'Unnamed')
            )

        else:
            raise TypeError("Unsupported 'refcat' type. Supported 'refcat' "
                            "types are 'tweakwcs.tpwcs.TPWCS' and "
                            "'astropy.table.Table'")

    # find group ID and assign images to groups:
    grouped_images = collections.defaultdict(list)
    for wcat in wcs_im_cats:
        grouped_images[wcat.group_id].append(wcat)

    # create WCSImageCatalog and WCSGroupCatalog:
    wcs_gcat = []
    for group_id, wcatalogs in grouped_images.items():
        if group_id is None:
            for wcat in wcatalogs:
                wcs_gcat.append(
                    WCSGroupCatalog(wcat, name='GROUP ID: None')
                )

        else:
            gcat = WCSGroupCatalog(
                wcatalogs,
                name='GROUP ID: {}'.format(group_id)
            )
            if not len(gcat.catalog):
                log.warning("Group with ID '{}' will not be aligned: empty "
                            "source catalog".format(group_id))

                for wcat in wcatalogs:
                    wcat.tpwcs.meta['fit_info'] = {
                        'status': 'FAILED: empty source catalog'
                    }

                continue

            wcs_gcat.append(gcat)

    # check that we have enough input images:
    if (refcat is None and len(wcs_gcat) < 2) or len(wcs_gcat) == 0:
        raise ValueError("Too few input images (or groups of images) with "
                         "non-empty catalogs.")

    # get the first image to be aligned and
    # create reference catalog if needed:
    if refcat is None:
        # create reference catalog:
        ref_imcat, current_wcat = max_overlap_pair(
            images=wcs_gcat,
            enforce_user_order=enforce_user_order or not expand_refcat
        )
        log.info("Selected image '{}' as reference image"
                 .format(ref_imcat.name))

        refcat = RefCatalog(ref_imcat.catalog, name=ref_imcat[0].name)

        for wcat in ref_imcat:
            wcat.tpwcs.meta['fit_info'] = {'status': 'REFERENCE'}

    else:
        # find the first image to be aligned:
        current_wcat = max_overlap_image(
            refimage=refcat,
            images=wcs_gcat,
            enforce_user_order=enforce_user_order or not expand_refcat
        )

    while current_wcat is not None:
        log.info("Aligning image catalog '{}' to the reference catalog."
                 .format(current_wcat.name))

        current_wcat.align_to_ref(
            refcat=refcat,
            ref_tpwcs=ref_tpwcs,
            match=match,
            minobj=minobj,
            fitgeom=fitgeom,
            nclip=nclip,
            sigma=sigma
        )

        for wcat in current_wcat:
            wcat.tpwcs.meta['fit_info'] = wcat.fit_info

        # add unmatched sources to the reference catalog:
        if expand_refcat:
            unmatched_src = current_wcat.get_unmatched_cat()
            refcat.expand_catalog(unmatched_src)
            log.info("Added {:d} unmatched sources from '{}' to the reference "
                     "catalog.".format(len(unmatched_src), current_wcat.name))

        # find the next image to be aligned:
        current_wcat = max_overlap_image(
            refimage=refcat,
            images=wcs_gcat,
            enforce_user_order=enforce_user_order or not expand_refcat
        )

    # log running time:
    runtime_end = datetime.now()
    log.info(" ")
    log.info("***** {:s}.{:s}() ended on {}"
             .format(__name__, function_name, runtime_end))
    log.info("***** {:s}.{:s}() TOTAL RUN TIME: {}"
             .format(__name__, function_name, runtime_end - runtime_begin))
    log.info(" ")

    eff_refcat = refcat.catalog
    return eff_refcat


def overlap_matrix(images):
    """
    Compute overlap matrix: non-diagonal elements (i,j) of this matrix are
    absolute value of the area of overlap on the sky between i-th input image
    and j-th input image.

    .. note::
        The diagonal of the returned overlap matrix is set to ``0.0``, i.e.,
        this function does not compute the area of the footprint of a single
        image on the sky.

    Parameters
    ----------

    images: list of WCSImageCatalog, WCSGroupCatalog, or RefCatalog
        A list of catalogs that implement :py:meth:`intersection_area` method.

    Returns
    -------
    m: numpy.ndarray
        A `numpy.ndarray` of shape ``NxN`` where ``N`` is equal to the
        number of input images. Each non-diagonal element (i,j) of this matrix
        is the absolute value of the area of overlap on the sky between i-th
        input image and j-th input image. Diagonal elements are set to ``0.0``.

    """
    nimg = len(images)
    m = np.zeros((nimg, nimg), dtype=np.double)
    for i in range(nimg):
        for j in range(i + 1, nimg):
            area = images[i].intersection_area(images[j])
            m[j, i] = area
            m[i, j] = area
    return m


def max_overlap_pair(images, enforce_user_order):
    """
    Return a pair of images with the largest overlap.

    .. warning::
        Returned pair of images is "poped" from input ``images`` list and
        therefore on return ``images`` will contain a smaller number of
        elements.

    Parameters
    ----------

    images: list of WCSImageCatalog, WCSGroupCatalog, or RefCatalog
        A list of catalogs that implement :py:meth:`intersection_area` method.

    enforce_user_order: bool
        When ``enforce_user_order`` is `True`, a pair of images will be
        returned **in the same order** as they were arranged in the ``images``
        input list. That is, image overlaps will be ignored.

    Returns
    -------
    (im1, im2)
        Returns a tuple of two images - elements of input ``images`` list.
        When ``enforce_user_order`` is `True`, images are returned in the
        order in which they appear in the input ``images`` list. When the
        number of input images is smaller than two, ``im1`` and ``im2`` may
        be `None`.

    """
    nimg = len(images)

    if nimg == 0:
        return None, None

    elif nimg == 1:
        return images[0], None

    elif nimg == 2 or enforce_user_order:
        # for the special case when only two images are provided
        # return (refimage, image) in the same order as provided in 'images'.
        # Also, when ref. catalog is static - revert to old tweakreg behavior
        im1 = images.pop(0)  # reference image
        im2 = images.pop(0)
        return im1, im2

    m = overlap_matrix(images)
    i, j = np.unravel_index(m.argmax(), m.shape)
    si = np.sum(m[i])
    sj = np.sum(m[:, j])

    if si < sj:  # pragma: no branch
        i, j = j, i

    if i < j:  # pragma: no branch
        j -= 1

    im1 = images.pop(i)  # reference image
    im2 = images.pop(j)

    # Sort the remaining of the input list of images by overlap area
    # with the reference image (in decreasing order):
    row = m[i]
    row = np.delete(row, i)
    row = np.delete(row, j)
    sorting_indices = np.argsort(row)[::-1]
    sorted_images = [images[k] for k in sorting_indices]  # apply argsort
    del images[:]
    for im in sorted_images:
        images.append(im)

    return im1, im2


def max_overlap_image(refimage, images, enforce_user_order):
    """
    Return the image from the input ``images`` list that has the largest
    overlap with the ``refimage`` image.

    .. warning::
        Returned image of images is "poped" from input ``images`` list and
        therefore on return ``images`` will contain a smaller number of
        elements.

    Parameters
    ----------
    refimage: RefCatalog
        Reference catalog.

    images: list of WCSImageCatalog, or WCSGroupCatalog
        A list of catalogs that implement :py:meth:`intersection_area` method.

    enforce_user_order: bool
        When ``enforce_user_order`` is `True`, returned image is the first
        image from the ``images`` input list regardless ofimage overlaps.

    Returns
    -------
    image: WCSImageCatalog, WCSGroupCatalog, or None
        Returns an element of input ``images`` list. When input list is
        empty - `None` is returned.

    """
    if not images:
        return None

    if enforce_user_order:
        # revert to old tweakreg behavior
        return images.pop(0)

    idx = np.argmax([refimage.intersection_area(im) for im in images])

    return images.pop(idx)
