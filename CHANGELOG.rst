.. _release_notes:

=============
Release Notes
=============


tweakwcs (DEVELOPMENT)
======================

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
