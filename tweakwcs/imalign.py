"""
A module that provides functions for "aligning" images: specifically, it
provides functions for computing corrections to image ``WCS`` so that
image catalogs "align" to the reference catalog *on the sky*.

:Authors: Mihai Cara (contact: help@stsci.edu)

:License: :doc:`../LICENSE`

"""
from __future__ import (absolute_import, division, unicode_literals,
                        print_function)

# STDLIB
import logging
from datetime import datetime

# THIRD PARTY
import numpy as np

# LOCAL
from . wcsimage import *

from . import __version__, __version_date__

__author__ = 'Mihai Cara'

__all__ = ['tweak_wcs']


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def tweak_wcs(refcat, imcat, imwcs, fitgeom='general', nclip=3, sigma=3.0):
    """ "Tweak" image's ``WCS`` by fitting image catalog to a reference
    catalog.

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

    """

    function_name = tweak_wcs.__name__

    # Time it
    runtime_begin = datetime.now()

    log.info(" ")
    log.info("***** {:s}.{:s}() started on {}"
             .format(__name__, function_name, runtime_begin))
    log.info("      Version {} ({})".format(__version__, __version_date__))
    log.info(" ")

    # check fitgeom:
    fitgeom = fitgeom.lower()
    if fitgeom not in ['shift', 'rscale', 'general']:
        raise ValueError("Unsupported 'fitgeom'. Valid values are: "
                         "'shift', 'rscale', or 'general'")

    wimcat = WCSImageCatalog(imcat, imwcs, shape=None,
                             name=imcat.meta.get('name', None))
    print(wimcat.catalog)
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
