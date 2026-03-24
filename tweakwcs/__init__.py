"""
This package provides support for image alignment.

"""
__docformat__ = 'restructuredtext'

__taskname__ = 'tweakwcs'
__author__ = 'Mihai Cara'


from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = ''


from .correctors import (
    WCSCorrector,
    FITSWCSCorrector,
    ST_V2V3_WCSCorrector,
    JWSTWCSCorrector,
    RomanWCSCorrector,
)
from .matchutils import XYXYMatch, MatchCatalogs, MatchSourceConfusionError
from .imalign import fit_wcs, align_wcs
from .wcsimage import (
    convex_hull,
    RefCatalog,
    WCSImageCatalog,
    WCSGroupCatalog,
)
from .linalg import inv
from .linearfit import iter_linear_fit, build_fit_matrix

# import deprecated classes:
from .correctors import TPWCS, JWSTgWCS, FITSWCS  # noqa: F401
from .matchutils import TPMatch  # noqa: F401


__all__ = [
    'FITSWCSCorrector',
    'JWSTWCSCorrector',
    'MatchCatalogs',
    'MatchSourceConfusionError',
    'RefCatalog',
    'RomanWCSCorrector',
    'ST_V2V3_WCSCorrector',
    'WCSCorrector',
    'WCSGroupCatalog',
    'WCSImageCatalog',
    'XYXYMatch',
    'align_wcs',
    'build_fit_matrix',
    'convex_hull',
    'fit_wcs',
    'inv',
    'iter_linear_fit',
]
