# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides support for manipulating tangent-plane corrections
of ``WCS``.

:Authors: Mihai Cara

:License: :doc:`LICENSE`

"""
import logging
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

from .correctors import TPWCS, JWSTgWCS, FITSWCS
from . import __version__  # noqa: F401

__author__ = 'Mihai Cara'

__all__ = ['TPWCS', 'JWSTgWCS', 'FITSWCS']

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


warnings.warn(
    "Module 'tweakwcs.tpwcs' has been deprecated since version 0.8.0. "
    "Please use corrector classes from the 'tweakwcs.correctors' module.",
    AstropyDeprecationWarning
)
