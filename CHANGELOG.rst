.. _release_notes:

=============
Release Notes
=============


.. tweakwcs (DEVELOPMENT)
   ======================



tweakwcs v0.3.2 (15-January-2019)
=================================

- Fixed the formula for computing ``RMSD`` of non-weighted fit. [#46]


tweakwcs v0.3.1 (14-January-2019)
=================================

- Fixed Read-The-Docs build failure. [#45]


tweakwcs v0.3.0 (14-January-2019)
=================================

- Implemented higher-accuracy matrix inversion. [#42]

- Bug fix related to not switching to using ``bounding_box`` instead of
  ``pixel_shape``. [#41]

- Added support for optional ``'weight'`` column in catalogs indicating
  the weight of each source in fitting linear transformations. [#41]

- Add support for weights to the linear fitting routines. [#40]

- Replaced the use of RMS for each axis with a single RMSD value, see
  [Root-Mean-Square Deviation]\
  (https://en.wikipedia.org/wiki/Root-mean-square_deviation). [#40]

- Rely on ``pixel_bounds``
  [see APE 14](https://github.com/astropy/astropy-APEs/blob/master/APE14.rst)
  when available for computation of image's bounding box. [#39]

- Fix a bug in the computation of the world coordinates of the fitted
  (*aligned*) sources. [#36]


tweakwcs v0.2.0 (20-December-2018)
==================================

- Fix swapped reported reference and input indices of sources used for
  fitting. [#34]

- Fix for non-initialized C arrays. [#34]

- Changelog correction. [#33]


tweakwcs v0.1.1 (11-December-2018)
==================================

- Fixeded a bug due to which ``'fit_ref_idx'`` and ``'fit_input_idx'``
  fields in the ``fit`` dictionary were never updated. [#31]

- ``jwst`` (pipeline) package is no longer a hard dependency. [#30]

- Removed unnecessary install dependencies. [#30]

- Documentation improvements. [#30, #32]

- Corrected 'RA', 'DEC' units used to compute bounding polygon for the
  reference catalog. [#30]

- Updated ``C`` code to avoid ``numpy`` deprecation warnings. [#30]


tweakwcs v0.1.0 (08-December-2018)
==================================

- Added support for aligning FITS WCS. [#15, #16]

- Added keywords to ``meta`` attributes of the ``TPWCS`` and ``NDData``
  to allow easy access to the match and fit information. [#20, #21, #28]

- Package and setup re-design. Support for ``readthedocs``. [#23]

- Documentation improvements. [#17, #18]

- Numerous other bug fixes, code clean-up, documentation improvements
  and enhancements. [#2, #3, #4, #5, #6, #7, #8, #9, #10, #11, #12, #13, #14, \
  #19, #22, #24, #25, #26, #27, #28, #29]


tweakwcs v0.0.1 (25-April-2018)
===============================

Initial release. [#1]
