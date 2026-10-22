Welcome to tweakwcs documentation!
==================================

``tweakwcs`` is a package that provides core algorithms for computing and
applying corrections to ``WCS`` objects such as to minimize mismatch between
image and reference catalogs. Currently only aligning images with ``FITS WCS``
and ``JWST GWCS`` are supported.

.. attention::

	Starting December 1st 2026,	STScI will no longer test macOS x86 versions of software (Intel processors, including all Mac machines earlier than 2020).
	macOS x86 builds will still be available on PyPI and installable with ``pip install``, but will not be tested for numerical accuracy.

	Going forward, software testing for macOS will only be ARM64 (`Apple Silicon <https://support.apple.com/en-us/116943>`_).

.. moduleauthor:: Mihai Cara

Content
=======

.. toctree::
   :maxdepth: 2

   source/imalign
   source/correctors
   source/matchutils
   source/wcsimage
   source/linearfit
   source/wcsutils
   source/linalg
   source/LICENSE.rst


Development Notes
=================

.. toctree::
   :maxdepth: 1
   :titlesonly:

   source/CHANGELOG.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
