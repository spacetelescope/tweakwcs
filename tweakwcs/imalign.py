# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A module that provides functions for "aligning" images: specifically, it
provides functions for computing corrections to image ``WCS`` so that
image catalogs "align" to the reference catalog *on the sky*.

:Authors: Mihai Cara (contact: help@stsci.edu)

:License: :doc:`../LICENSE`

"""
# STDLIB
import logging
from datetime import datetime
import collections
from copy import deepcopy

# THIRD PARTY
import numpy as np
import astropy
from astropy.nddata import NDDataBase
import gwcs

# We need JWST DataModel so that we can detect this type and treat it
# differently from astropy.nddata.NDData because JWST's WCS is stored in
# DataModel.meta.wcs:
try:
    from jwst.datamodels import DataModel
except:
    DataModel = None

# LOCAL
from . wcsimage import *
from . tpwcs import *
from . matchutils import *

from . import __version__, __version_date__

__author__ = 'Mihai Cara'

__all__ = ['tweak_wcs', 'tweak_image_wcs']


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def tweak_wcs(refcat, imcat, imwcs, fitgeom='general', nclip=3, sigma=3.0):
    """ "Tweak" image's ``WCS`` by fitting image catalog to a reference
    catalog.

    .. note::
        Both reference and image catalogs must have been matched *prior to*
        calling ``tweak_wcs()``. This means that the lengths of both
        ``refcat`` and ``imcat`` catalogs must be equal *and* that coordinates
        with the same indices in both catalogs correspond to the same source.

    Parameters
    ----------

    refcat : astropy.table.Table
        A reference source catalog. The catalog must contain ``'RA'`` and
        ``'DEC'`` columns which indicate reference source world
        coordinates (in degrees).

    imcat : astropy.table.Table
        Source catalog associated with an image. Must contain ``'x'`` and
        ``'y'`` columns which indicate source coordinates (in pixels) in
        the associated image.

    imwcs : TPWCS
        A ``WCS`` associated with the image from which the catalog was derived.
        This ``TPWCS`` must also define a tangent plane that will be used
        for fitting the two catalogs' sources and in which WCS corrections
        will be applied.

    fitgeom : {'shift', 'rscale', 'general'}, optional
        The fitting geometry to be used in fitting the matched object lists.
        This parameter is used in fitting the offsets, rotations and/or scale
        changes from the matched object lists. The 'general' fit geometry
        allows for independent scale and rotation for each axis.

    nclip : int, optional
        Number (a non-negative integer) of clipping iterations in fit.

    sigma : float, optional
        Clipping limit in sigma units.

    Returns
    -------

    twwcs : TPWCS
        "Tweaked" (aligned) ``WCS`` that contains tangent-plane corrections
        so that reference and image catalog sources better align in the tangent
        plane and therefore on the sky as well.

    Notes
    -----

    Upon successful completion, this function will set the following ``meta``
    fields of the ``meta`` attribute of the returned ``TPWCS`` object:

        * **'fitgeom'**: the value of the ``fitgeom`` argument
        * **'matrix'**: computed rotation matrix
        * **'shift'**: offset along X- and Y-axis
        * **'eff_minobj'**: effective value of the ``minobj`` parameter
        * **'fit_ref_idx'**: indices of the sources from the reference catalog
          used for fitting
        * **'fit_input_idx'**: indices of the sources from the "input" (image)
          catalog used for fitting
        * **'rot'**: rotation angle as if rotation is a proper rotation
        * **'proper'**: Indicates whether the rotation is a proper rotation
          (boolean)
        * **'rotxy'**: a tuple of (rotation of the X-axis, rotation of the
          Y-axis, mean rotation, computed skew)
        * **'scale'**: a tuple of (mean scale, scale along X-axis, scale along
          Y-axis)
        * **'skew'**: computed skew
        * **'rms'**: fit RMS in *image* coordinates as a tuple of two values:
          (RMS_X, RMS_Y)
        * **'status'**: Alignment status. Currently two possible status are
          possible ``'SUCCESS'`` or ``'FAILED: reason for failure'``.
          When alignment failed, the reason for failure is provided after
          alignment status.

    """
    function_name = tweak_wcs.__name__

    # Time it
    runtime_begin = datetime.now()

    log.info(" ")
    log.info("***** {:s}.{:s}() started on {}"
             .format(__name__, function_name, runtime_begin))
    log.info("      Version {} ({})".format(__version__, __version_date__))
    log.info(" ")

    try:
        # Attempt to set initial status to FAILED.
        imwcs.meta['status'] = "FAILED: Unknown error"
    except:
        # Most likely the code will fail later with a more specific exception
        pass

    # check fitgeom:
    fitgeom = fitgeom.lower()
    if fitgeom not in ['shift', 'rscale', 'general']:
        raise ValueError("Unsupported 'fitgeom'. Valid values are: "
                         "'shift', 'rscale', or 'general'")

    wimcat = WCSImageCatalog(imcat, imwcs, shape=None,
                             name=imcat.meta.get('name', None))
    wgcat = WCSGroupCatalog(wimcat, name=imcat.meta.get('name', None))
    wrefcat = RefCatalog(refcat, name=imcat.meta.get('name', None))

    succes = wgcat.align_to_ref(refcat=wrefcat, match=None, minobj=None,
                                fitgeom=fitgeom, nclip=nclip, sigma=sigma)

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

    return wgcat[0].imwcs


def tweak_image_wcs(images, refcat=None, enforce_user_order=True,
                    expand_refcat=False, minobj=None, match=TPMatch(),
                    fitgeom='general', nclip=3, sigma=3.0):
    """
    Align (groups of) images by adjusting the parameters of their WCS based on
    fits between matched sources in these images and a reference catalog which
    may be automatically created from one of the input images.

    .. warning::
        This function modifies the ``image.wcs`` or ``image.meta.wcs``
        (in the case of JWST ``DataModel``s) attribute of each item
        in the input ``images`` list!

    Upon completion, this function will add a field ``'tweakwcs_info'``
    to the ``meta`` attribute of the input images (except of the image
    chosen as a reference image when ``refcat`` is `None`) containing
    a dictionary describing matching and fit results. For a description
    of the items in this dictionary, see
    :meth:`tweakwcs.wcsimage.WCSGroupCatalog.align_to_ref`. In addition to the
    status set by :meth:`~tweakwcs.wcsimage.WCSGroupCatalog.align_to_ref`,
    this function may set status to ``'REFERENCE'`` for an input image used
    as a reference image when a reference catalog is not provided.
    In this case no other fields in the ``'tweakwcs_info'`` will be present
    because a reference image is not being aligned. When alignment failed,
    the reason for failure is provided after alignment status.

    .. warning::

        Unless status in ``'tweakwcs_info'`` is ``'SUCCESS'``, there is no
        guarantee that other fields in ``'tweakwcs_info'`` are present or
        valid. Therefore, it is advisable verify that status is ``'SUCCESS'``
        before attempting to access other items, for example:

        >>> tweak_info = images[0].meta.get('tweakwcs_info')
        >>> if tweak_info['status'] == 'SUCCESS':
        ...     print("shifts: [{}, {}]".format(*tweak_info['shift']))
        ... else:
        ...     print("tweak info not available for this image")

    Parameters
    ----------
    images : list of astropy.nddata.NDDataBase
        A list of `astropy.nddata.NDDataBase` objects whose WCS (provided
        through its ``meta['wcs']`` attribute) should be adjusted.

        .. warning::
            This function modifies the WCS of the input images provided
            through the ``images`` parameter. On return, each input image WCS
            will be replaced with a "tweaked" WCS version.

    refcat : astropy.table.Table, astropy.nddata.NDDataBase, optional
        A reference source catalog. When ``refcat`` is an
        `astropy.table.Table`, the catalog must contain ``'RA'`` and
        ``'DEC'`` columns which indicate reference source world
        coordinates (in degrees).

        When ``refcat`` is an  `astropy.nddata.NDDataBase`, its ``meta``
        attribute must contain at least ``'catalog'`` item that is an
        `astropy.table.Table`. There are two different scenarios regarding
        the information (columns) that this catalog should contain:

        - If ``refcat``'s ``meta`` attribute **does not** contain a ``'wcs'``
          item, then the catalog must contain ``'RA'`` and ``'DEC'`` columns
          which indicate reference source world coordinates (in degrees).

        - If ``refcat``'s ``meta`` attribute contains a ``'wcs'`` item with
          a valid ``WCS`` then the catalog *can* contain *either*
          ``'RA'`` and ``'DEC'`` columns and/or ``'x'`` and ``'y'``
          columns which indicate reference source image coordinates
          (in pixels). The ``'x'`` and ``'y'`` columns are used only when
          ``'RA'`` and ``'DEC'`` are not provided. Image coordinates are
          converted (if necessary) to world coordinates using ``refcat``'s
          WCS object.

    enforce_user_order : bool, optional
        Specifies whether images should be aligned in the order specified in
        the `file` input parameter or `align` should optimize the order
        of alignment by intersection area of the images. Default value (`True`)
        will align images in the user specified order, except when some images
        cannot be aligned in which case `align` will optimize the image
        alignment order. Alignment order optimization is available *only*
        when ``expand_refcat`` is `True`.

    expand_refcat : bool, optional
        Specifies whether to add new sources from just matched images to
        the reference catalog to allow next image to be matched against an
        expanded reference catalog. By delault, the reference catalog is not
        being expanded.

    minobj : int, None, optional
        Minimum number of identified objects from each input image to use
        in matching objects from other images. If the default `None` value is
        used then `align` will automatically deternmine the minimum number
        of sources from the value of the ``fitgeom`` parameter.

    match : MatchCatalogs, function, None, optional
        A callable that takes two arguments: a reference catalog and an
        image catalog. Both catalogs will have columns ``'TPx'`` and
        ``'TPy'`` that represent the source coordinates in some common
        (to both catalogs) coordinate system.

    fitgeom : {'shift', 'rscale', 'general'}, optional
        The fitting geometry to be used in fitting the matched object lists.
        This parameter is used in fitting the offsets, rotations and/or scale
        changes from the matched object lists. The 'general' fit geometry
        allows for independent scale and rotation for each axis.

    nclip : int, optional
        Number (a non-negative integer) of clipping iterations in fit.

    sigma : float, optional
        Clipping limit in sigma units.

    """
    function_name = tweak_image_wcs.__name__

    # Time it
    runtime_begin = datetime.now()

    log.info(" ")
    log.info("***** {:s}.{:s}() started on {}"
             .format(__name__, function_name, runtime_begin))
    log.info("      Version {} ({})".format(__version__, __version_date__))
    log.info(" ")

    # Check that type of `images` is correct and set initial status to FAILED:
    if isinstance(images, NDDataBase):
        images.meta['tweakwcs_info'] = {'status': "FAILED: Unknown error"}
        images = [images]
    else:
        try:
            imtype_ok = all([isinstance(i, NDDataBase) for i in images])
        except:
            imtype_ok = False
        finally:
            if imtype_ok:
                for im in images:
                    # initially set a "bad" status and update later
                    # if successful:
                    im.meta['tweakwcs_info'] = {
                        'status': "FAILED: Unknown error"
                    }
            else:
                raise TypeError("Input 'images' must be either a single "
                                "'NDDataBase' object or a list of "
                                "'NDDataBase' objects.")

    # check fitgeom:
    fitgeom = fitgeom.lower()
    if fitgeom not in ['shift', 'rscale', 'general']:
        raise ValueError("Unsupported 'fitgeom'. Valid values are: "
                         "'shift', 'rscale', or 'general'")

    if minobj is None:
        if fitgeom == 'general':
            minobj = 3
        elif fitgeom == 'rscale':
            minobj = 2
        else:
            minobj = 1

    # process reference catalog or image if provided:
    if refcat is not None:
        if isinstance(refcat, NDDataBase):
            if not 'catalog' in refcat.meta:
                raise ValueError("Reference 'NDDataBase' must contain a "
                                 "catalog.")

            rcat = refcat.meta['catalog'].copy()

            if 'RA' not in rcat.colnames or 'DEC' not in rcat.colnames:
                # convert image x & y to world coordinates:
                if refcat.wcs is None:
                    raise ValueError("A valid WCS is required to convert "
                                     "image coordinates in the reference "
                                     "catalog to world coordinates.")

                #TODO: Need to implement astropy APE-14 support.
                if (hasattr(refcat.meta, 'wcs') and
                    isinstance(refcat.meta.wcs, gwcs.WCS)):

                    # most likely we are dealing with JWST models.
                    # In any case, the WCS is a gWCS => use forward transform:
                    ra, dec = refcat.meta.wcs(rcat['x'], rcat['y'])

                elif isinstance(refcat.wcs, astropy.wcs.WCS):
                    ra, dec = refcat.wcs(rcat['x'], rcat['y'], 0)

                elif isinstance(refcat.wcs, gwcs.WCS):
                    # we are dealing with a gWCS:
                    ra, dec = refcat.wcs(rcat['x'], rcat['y'])

                else:
                    raise TypeError("Unsupported WCS type for the reference "
                                    "catalog.")

                rcat['RA'] = ra
                rcat['DEC'] = dec
                if 'name' not in rcat.meta and 'name' in refcat.meta:
                    rcat.meta['name'] = refcat.meta['name']

            refcat = rcat

        elif isinstance(refcat,  astropy.table.Table):
            if 'RA' not in refcat.colnames or 'DEC' not in refcat.colnames:
                raise KeyError("Reference catalogs *must* contain *both* 'RA' "
                               "and 'DEC' columns.")

        else:
            raise TypeError("Unsupported 'refcat' type. Supported 'refcat' "
                            "types are 'astropy.nddata.NDDataBase' and "
                            "'astropy.table.Table'")

        refcat = RefCatalog(refcat, name=refcat.meta.get('name', None))

    # find group ID and assign images to groups:
    grouped_images = collections.defaultdict(list)
    for img in images:
        grouped_images[img.meta.get('group_id', None)].append(img)

    # create WCSImageCatalog and WCSGroupCatalog:
    imcat = []
    for group_id, imlist in grouped_images.items():
        if group_id is None:
            for img in imlist:
                if 'catalog' in img.meta:
                    catalog = img.meta['catalog'].copy()
                else:
                    raise ValueError("Each image must have a valid catalog.")

                #TODO: Currently code works only for JWST gWCS and FITS WCS!
                #      What is needed is a way to let users to
                #      specify a corrector class for images.
                #
                if (hasattr(img.meta, 'wcs') and
                    isinstance(img.meta.wcs, gwcs.WCS)):
                    # we need this special check because jwst package is
                    # an *optional* dependency:

                    if DataModel is None:
                        # most likely we've got a JWST DataModel but we
                        # do not have the required `jwst` package installed
                        # to deal with jwst DataModel
                        raise ImportError(
                            "Suspected jwst.datamodels.DataModel input image "
                            "but the required 'jwst' package is not installed."
                        )

                    # We are dealing with JWST ImageModel
                    wcsinfo = img.meta.get('wcsinfo', None)
                    if wcsinfo is not None:
                        wcsinfo = wcsinfo._instance
                    wcs_corr = JWSTgWCS(deepcopy(img.meta.wcs), wcsinfo)

                else:
                    img_wcs = img.wcs

                    if isinstance(img.wcs, astropy.wcs.WCS):
                        wcs_corr = FITSWCS(deepcopy(img.wcs))

                    elif isinstance(img.wcs, gwcs.WCS):
                        #TODO: Currently only JWST gWCS is supported
                        raise NotImplementedError(
                            "Currently only alignment of JWST gWCS is "
                            "supported. Support for alignment of arbitrary "
                            "image gWCS has not yet been implemented."
                        )

                    else:
                        raise TypeError("Unsupported WCS type for image "
                                        "catalog.")

                imcat.append(
                    WCSGroupCatalog(
                        WCSImageCatalog(
                            catalog=catalog,
                            imwcs=wcs_corr,
                            shape=img.data.shape,
                            name=img.meta.get('name', None),
                            meta={'orig_image_nddata': img}
                        ),
                        name='GROUP ID: None'
                    )
                )

        else:
            wcsimlist = []
            for img in imlist:
                if 'catalog' in img.meta:
                    catalog = img.meta['catalog'].copy()
                else:
                    raise ValueError("Each image must have a valid catalog.")

                #TODO: this works only for JWST gWCS and FITS WCS!
                #      What is needed is a way to let users to
                #      specify a corrector class for images.
                if hasattr(img.meta, 'wcs'):
                    # We are dealing with JWST ImageModel
                    img_wcs = img.meta.wcs
                else:
                    img_wcs = img.wcs

                if isinstance(img_wcs, astropy.wcs.WCS):
                    wcs_corr = FITSWCS(deepcopy(img_wcs))
                else:
                    wcsinfo = img.meta.get('wcsinfo', None)
                    if wcsinfo is not None:
                        wcsinfo = wcsinfo._instance
                    wcs_corr = JWSTgWCS(deepcopy(img_wcs), wcsinfo)

                wcsimlist.append(
                    WCSImageCatalog(
                        catalog=catalog,
                        imwcs=wcs_corr,
                        shape=img.data.shape,
                        name=img.meta.get('name', None),
                        meta={'orig_image_nddata': img}
                    )
                )

            imcat.append(WCSGroupCatalog(wcsimlist,
                                         name='GROUP ID: {}'.format(group_id)))

    # check that we have enough input images:
    if (refcat is None and len(imcat) < 2) or len(imcat) == 0:
        raise ValueError("Too few input images (or groups of images).")

    # get the first image to be aligned and
    # create reference catalog if needed:
    if refcat is None:
        # create reference catalog:
        ref_imcat, current_imcat = max_overlap_pair(
            images=imcat,
            enforce_user_order=enforce_user_order or not expand_refcat
        )
        log.info("Selected image '{}' as reference image"
                 .format(ref_imcat.name))

        refcat = RefCatalog(ref_imcat.catalog, name=ref_imcat[0].name)

        for im in ref_imcat:
            nddata_obj = im.meta['orig_image_nddata']
            nddata_obj.meta['tweakwcs_info'] = {'status': 'REFERENCE'}

        # aligned_imcat = [img.meta['orig_image_nddata'] for img in ref_imcat]

    else:
        # find the first image to be aligned:
        current_imcat = max_overlap_image(
            refimage=refcat,
            images=imcat,
            enforce_user_order=enforce_user_order or not expand_refcat
        )
        # aligned_imcat = []

    while current_imcat is not None:
        log.info("Aligning image catalog '{}' to the reference catalog."
                 .format(current_imcat.name))

        current_imcat.align_to_ref(
            refcat=refcat,
            match=match,
            minobj=minobj,
            fitgeom=fitgeom,
            nclip=nclip,
            sigma=sigma
        )
        for image in current_imcat:
            img = image.meta['orig_image_nddata']
            if DataModel is not None and isinstance(img, DataModel):
                # We are dealing with JWST ImageModel
                img.meta.wcs = image.imwcs.wcs
            else:
                #TODO: find an alternative way of asigning the WCS
                #      without using private members of NDData
                #      See https://github.com/astropy/astropy/issues/8192
                img._wcs = image.imwcs.wcs
            # aligned_imcat.append(image)
            img.meta['tweakwcs_info'] = deepcopy(image.imwcs.meta)

        # add unmatched sources to the reference catalog:
        if expand_refcat:
            unmatched_src = current_imcat.get_unmatched_cat()
            refcat.expand_catalog(unmatched_src)
            log.info("Added {:d} unmatched sources from '{}' to the reference "
                     "catalog.".format(len(unmatched_src), current_imcat.name))

        # find the next image to be aligned:
        current_imcat = max_overlap_image(
            refimage=refcat,
            images=imcat,
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

    # return aligned_imcat # aligned_imcat may be out of order wrt to input


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

    images : list of WCSImageCatalog, WCSGroupCatalog, or RefCatalog
        A list of catalogs that implement :py:meth:`intersection_area` method.

    Returns
    -------
    m : numpy.ndarray
        A `numpy.ndarray` of shape ``NxN`` where ``N`` is equal to the
        number of input images. Each non-diagonal element (i,j) of this matrix
        is the absolute value of the area of overlap on the sky between i-th
        input image and j-th input image. Diagonal elements are set to ``0.0``.

    """
    nimg = len(images)
    m = np.zeros((nimg, nimg), dtype=np.float)
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

    images : list of WCSImageCatalog, WCSGroupCatalog, or RefCatalog
        A list of catalogs that implement :py:meth:`intersection_area` method.

    enforce_user_order : bool
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

    if si < sj:
        c = j
        j = i
        i = c

    if i < j:
        j -= 1

    im1 = images.pop(i)  # reference image
    im2 = images.pop(j)

    # Sort the remaining of the input list of images by overlap area
    # with the reference image (in decreasing order):
    row = m[i]
    row = np.delete(row, i)
    row = np.delete(row, j)
    sorting_indices = np.argsort(row)[::-1]
    images_arr = np.asarray(images)[sorting_indices]
    while len(images) > 0:
        del images[0]
    for k in range(images_arr.shape[0]):
        images.append(images_arr[k])

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
    refimage : RefCatalog
        Reference catalog.

    images : list of WCSImageCatalog, or WCSGroupCatalog
        A list of catalogs that implement :py:meth:`intersection_area` method.

    enforce_user_order : bool
        When ``enforce_user_order`` is `True`, returned image is the first
        image from the ``images`` input list regardless ofimage overlaps.

    Returns
    -------
    image: WCSImageCatalog, WCSGroupCatalog, or None
        Returns an element of input ``images`` list. When input list is
        empty - `None` is returned.

    """
    if len(images) < 1:
        return None

    if enforce_user_order:
        # revert to old tweakreg behavior
        return images.pop(0)

    area = [refimage.intersection_area(im) for im in images]
    idx = np.argmax(area)
    return images.pop(idx)
