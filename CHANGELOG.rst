.. _release_notes:

=============
Release Notes
=============

.. 0.4.1 (unreleased)
   ==================


0.4.0 (08-February-2019)
========================

- Matched indices, linear fit results and fit residuals are now set in the
  input "WCS catalogs" ``meta['fit_info']`` instead of
  ``meta['tweakwcs_info']``. [#57]

- Updated example notebook to reflect changes to API. [#57]

- Allow ``TPWCS`` classes to set ``meta`` during object instantiation.
  This allows attaching, for example, a source catalog to the tangent-plane
  WCS corrector object. [#57]

- ``align_wcs`` no longer supports ``NDData`` input. Instead catalogs can be
  provided directly in the ``meta`` attribute of ``TPWCS``-derived WCS
  "correctors". This fundamentally transfers the responsibility of
  instantiating the correct tangent-plane WCS to the caller. This, in turn,
  will allow future WCS to be supported by providing a custom ``TPWCS``-derived
  corrector defined externally to ``tweakwcs`` package. Second benefit is that
  image data no longer need to be kept in memory in ``NDData`` objects as
  image data are not needed for image alignment once catalogs have been
  created. [#57]

- Renamed ``tweak_wcs`` to ``fit_wcs`` and ``tweak_image_wcs`` to
  ``align_wcs``. [#57]

- Fixed a bug due to which the code might crash due to an undefined ``ra``
  variable, see issue #55. [#56]

- ``tweak_image_wcs()`` now returns effective reference catalog used for
  image alignment. [#54]

- Modified how IDs are assigned to the reference catalog source positions when
  ``expand_refcat`` is `True`: instead of having all sources numbered
  consecutively starting with 1, now the code will attempt to preserve
  the original IDs (if any) of the input reference catalog (``refcat``)
  or an input image used as a reference catalog and consecutively number only
  the sources being added to the ``refcat``. [#54]

- Modified the clipping algorithm to start with all valid sources at each
  iteration. In other words, clippings do not accumulate by default.
  Old behavior can be replicated by setting ``clip_accum`` to `True`. [#53]

- Cleaned-up ``iter_linear_fit`` interface as well as simplified the
  ``fit`` dictionary returned by ``iter_linear_fit``. [#53]

- Added option to specify statistics used for clipping. [#51, #52]


0.3.3 (21-January-2019)
=======================

- Corrected a bug in the non-weighted ``rscale`` fit. [#49]

- Corrected a bug in the computation of ``RMSE`` for the "general" fit. [#47]

- Added computation of ``MAE`` of the fit (in addition to ``RMSE``), see
  [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error).
  [#47]

- Renamed ``RMSD`` to ``RMSE`` (Root-Mean-Square Error). [#47]


0.3.2 (15-January-2019)
=======================

- Fixed the formula for computing ``RMSD`` of non-weighted fit. [#46]


0.3.1 (14-January-2019)
=======================

- Fixed Read-The-Docs build failure. [#45]


0.3.0 (14-January-2019)
=======================

- Implemented higher-accuracy matrix inversion. [#42]

- Bug fix related to not switching to using ``bounding_box`` instead of
  ``pixel_shape``. [#41]

- Added support for optional ``'weight'`` column in catalogs indicating
  the weight of each source in fitting linear transformations. [#41]

- Add support for weights to the linear fitting routines. [#40]

- Replaced the use of ``RMS`` for each axis with a single ``RMSD`` value, see
  [Root-Mean-Square Deviation]\
  (https://en.wikipedia.org/wiki/Root-mean-square_deviation). [#40]

- Rely on ``pixel_bounds``
  [see APE 14](https://github.com/astropy/astropy-APEs/blob/master/APE14.rst)
  when available for computation of image's bounding box. [#39]

- Fix a bug in the computation of the world coordinates of the fitted
  (*aligned*) sources. [#36]


0.2.0 (20-December-2018)
========================

- Fix swapped reported reference and input indices of sources used for
  fitting. [#34]

- Fix for non-initialized C arrays. [#34]

- Changelog correction. [#33]


0.1.1 (11-December-2018)
========================

- Fixeded a bug due to which ``'fit_ref_idx'`` and ``'fit_input_idx'``
  fields in the ``fit`` dictionary were never updated. [#31]

- ``jwst`` (pipeline) package is no longer a hard dependency. [#30]

- Removed unnecessary install dependencies. [#30]

- Documentation improvements. [#30, #32]

- Corrected 'RA', 'DEC' units used to compute bounding polygon for the
  reference catalog. [#30]

- Updated ``C`` code to avoid ``numpy`` deprecation warnings. [#30]


0.1.0 (08-December-2018)
========================

- Added support for aligning FITS WCS. [#15, #16]

- Added keywords to ``meta`` attributes of the ``TPWCS`` and ``NDData``
  to allow easy access to the match and fit information. [#20, #21, #28]

- Package and setup re-design. Support for ``readthedocs``. [#23]

- Documentation improvements. [#17, #18]

- Numerous other bug fixes, code clean-up, documentation improvements
  and enhancements. [#2, #3, #4, #5, #6, #7, #8, #9, #10, #11, #12, #13, #14, \
  #19, #22, #24, #25, #26, #27, #28, #29]


0.0.1 (25-April-2018)
=====================

Initial release. [#1]
