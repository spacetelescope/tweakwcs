.. _release_notes:

=============
Release Notes
=============

.. 0.7.0 (unreleased)
   ==================



0.6.5 (09-September-2020)
=========================

- Added ``ref_tpwcs`` parameter to ``imalign.fit_wcs()``,
  ``imalign.align_wcs()``, ``wcsimage.WCSGroupCatalog.align_to_ref()`` to allow
  alignment to be performed in user-specified distortion-free tangent
  plane. [#125]

- Renamed ``tanplane_wcs`` parameter in
  ``wcsimage.WCSGroupCatalog.apply_affine_to_wcs()`` to ``ref_tpwcs``.
  ``tanplane_wcs`` parameter was deprecated and will be removed in a future
  release. [#125]


0.6.4 (14-May-2020)
===================

- Bug fix: Unable to initialize ``JWSTgWCS`` tangent-plane corrector from an
  already corrected WCS. [#122]

- Fix a bug in how corrections are applied to a previously corrected
  JWST WCS. [#120]

- Do not attempt to extract center of linear transformation when not available
  in ``'fit_info'``. [#119]


0.6.3 (14-April-2020)
=====================

- Fixed a bug due to which reprojection transformation for JWST gWCS was
  computed at wrong location in the tangent plane. [#118]


0.6.2 (07-April-2020)
=====================

- When WCS has valid bounding box, estimate scale at the center of the
  bounding box. [#117]

- Adjust the point at which tangent plane-to-tangent plane transformation
  is computed by 1/2 pixels for JWST corrections. This correction should
  have no measurable impact on computed corrections. [#115]


0.6.1 (09-March-2020)
=====================

- Fixed a bug in applying JWST correction for the case when alignment is
  performed twice on the same image. Due to this bug the inverse transformation
  was not updated. [#112]


0.6.0 (25-February-2020)
========================

- Fix a possible crash when aligning FITS WCS images due to an unusual way
  ``stwcs.wcsutil.all_world2pix`` handles (or not) scalar arguments. [#110]

- Modified the angle at which the reported rotation angles are reported.
  Now rotation angles have the range ``[-180, 180]`` degrees. [#109]

- Added support FITS WCS that use ``PC`` matrix instead of the ``CD`` matrix
  used in HSTs WCS. [#108]

- Bug fix for alignment of multi-chip FITS images: correction of how
  transformations from the reference tangent plane are converted to
  individual images' tangent planes. [#106]

- Significant re-organization of the ``fit_info`` dictionary. ``rot`` now
  becomes ``proper_rot`` and ``rotxy`` now becomes ``rot`` containing only
  ``rotx`` and ``roty``. Also, ``scale`` now is a tuple of only two scales
  ``sx`` and ``sy``. The geometric mean scale is now a separate field
  ``'<scale>'`` as well as the arithmetic mean of rotation angles
  (``'<rot>'``). Finally, ``'offset'`` in the fit functions from the
  ``linearfit`` module was renamed to ``'shift'`` in order to match the
  same field returned by functions from the ``imalign`` module. [#105]

- Linear fit functions now return the fit matrix ``F`` instead of its
  transpose. [#100]

- Linear fit functions (in the ``linearfit`` module) use ``longdouble``
  for internal computations. [#100]

- Re-designed the ``JWSTgWCS`` corrector class to rely exclusively on
  basic models available in ``astropy`` and ``gwcs`` instead of the ``TPCorr``
  class provided by the ``jwst`` pipeline. This eliminates the need to install
  the ``jwst`` pipeline in order to align ``JWST`` images. [#96, #98]


0.5.3 (15-November-2019)
========================

- Added logic to allow some input catalogs to be empty and to allow the
  alignment to proceed as long as there are at least two non-empty
  (image or group) input catalogs. [#94]


0.5.2 (26-July-2019)
====================

- Fixed a deprecation issue in logging and added logic to compute image group's
  catalog name using a common prefix (if exists) of the names of constituent
  images. [#92]

- Package version is now handled by ``setuptools_scm``.
  [#93]


0.5.1 (08-May-2019)
===================

- Fixed a bug in the "2dhist" algorithm resulting in a crash when 2D histogram
  has multiple maxima of the same value and no other value larger than
  one. [#90]


0.5.0 (22-April-2019)
=====================

- Fixed a bug due to which a warning log message "Failed to align catalog..."
  would be issued for successful alignments. [#84]

- Fixed a bug in creation of WCS image groups with empty catalogs. [#84]

- Fixed a bug in ``match2ref`` when it was run in a non-matching mode
  (``match=None``) dute to which it was impossible to detect the case
  when reference catalog has a different length from a supposedly matched
  WCS group catalog. [#84]

- Fixed a bug in computation of the bounding polygon of a reference catalog
  containing only two sources. [#84]

- Fixed a bug in ``convex_hull()`` resulting in incorrect type being returned
  in case of empty input coordinate lists or whne only one point
  is provided. [#84]

- Implemented a more robust estimate of the maximum type supported by
  ``numpy.linalg.inv``. [#82]

- Renamed ``wcsutils.planar_rot_3D`` to ``wcsutils.planar_rot_3d``. [#75]

- Renamed ``wcsutils.cartesian2spherical`` to
  ``wcsutils.cartesian_to_spherical`` and ``wcsutils.spherical2cartesian``
  to ``wcsutils.spherical_to_cartesian``. [#71]

- Improved "2dhist" algorithm that performs simple catalog pre-alignment used
  for source matching. [#69]

- Changed the default value of the ``searchrad`` parameter in
  ``matchutils.TPMatch`` to 3. [#69]


0.4.5 (14-March-2019)
=====================

- Fixed incorrect pointer type introduced in previous release [#67].


0.4.4 (13-March-2019)
=====================

- Fixed VS2017 compiler error, ``"void *": unknown size``. [#62, #63, #64]


0.4.3 (13-March-2019)
=====================

- Package maintenance release.


0.4.2 (21-February-2019)
========================

- Fixed a bug due to which the fitting code would crash is ``wuv`` were
  provided but ``wxy`` were set to ``None``. [#60]


0.4.1 (14-February-2019)
========================

- Code cleanup: removed debug print statements. [#59]


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
