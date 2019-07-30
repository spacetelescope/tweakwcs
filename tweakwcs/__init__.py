"""
This package provides support for image alignment.

"""
__docformat__ = 'restructuredtext'

__taskname__ = 'tweakwcs'
__author__ = 'Mihai Cara'

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = 'UNKNOWN'

from .tpwcs import TPWCS, JWSTgWCS, FITSWCS  # noqa: F401
from .matchutils import MatchCatalogs, TPMatch  # noqa: F401
from .imalign import fit_wcs, align_wcs  # noqa: F401
from .wcsimage import (convex_hull, RefCatalog, WCSImageCatalog,  # noqa: F401
                       WCSGroupCatalog)  # noqa: F401
from .linalg import inv  # noqa: F401
from .linearfit import iter_linear_fit, build_fit_matrix  # noqa: F401
