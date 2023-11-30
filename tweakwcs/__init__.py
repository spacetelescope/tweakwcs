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


from .correctors import (WCSCorrector, JWSTWCSCorrector,  # noqa: F401
                         FITSWCSCorrector,  # noqa: F401
                         TPWCS, JWSTgWCS, FITSWCS)  # noqa: F401
from .matchutils import MatchCatalogs, XYXYMatch, TPMatch  # noqa: F401
from .imalign import fit_wcs, align_wcs  # noqa: F401
from .wcsimage import (convex_hull, RefCatalog, WCSImageCatalog,  # noqa: F401
                       WCSGroupCatalog)  # noqa: F401
from .linalg import inv  # noqa: F401
from .linearfit import iter_linear_fit, build_fit_matrix  # noqa: F401
