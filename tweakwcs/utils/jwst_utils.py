# Licensed under a 3-clause BSD style license - see LICENSE.rst
r"""
A module that provides functions for automatically adding ``group_id``
tag to images based on information recorded in image's ``meta`` attribute.

This ``tweakwcs_group_id`` tag, if set, can be used by ``align_wcs``,
and other functions from the ``imalign`` module to
identify images that should be treated together as a group when "tweaking"
their ``WCS``. That is, all images within a group will have the same
correction applied to their ``WCS``\ es. This is often the case with images
that come from differenct chips of the same sensor chip assembly (SCA).

.. note::
    Grouping logic/algorithms are inherently telescope and instrument
    dependent.

:Authors: Mihai Cara (contact: help@stsci.edu)

:License: :doc:`../LICENSE`

"""
# STDLIB
import logging
import uuid

# LOCAL
from .. import __version__  # noqa: F401


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def assign_jwst_tweakwcs_groups(images):
    """ Assign group IDs to ``JWST`` images.

    Parameters
    ----------
    images: list of str or jwst.datamodels.DataModel
        A list of string file names to ``JWST`` data or
        `jwst.datamodels.DataModel` objects. On return, these data models
        or files will contain ``tweakwcs_group_id`` attribute in their
        ``meta`` dictionary identifying the tweakwcs group. Images within
        a group are aligned together.

    """
    try:
        from jwst.datamodels import open as open_data  # pylint: disable=W0611
    except ImportError:
        raise ImportError("'assign_jwst_tweakwcs_groups' requires that "
                          "'jwst' package be installed.")

    close = [isinstance(im, (str, bytes)) for im in images]

    group_ids = {}

    for im, cl in zip(images, close):
        model = open_data(im) if cl else im
        try:
            meta_ids = (
                model.meta.observation.program_number,
                model.meta.observation.observation_number,
                model.meta.observation.visit_number,
                model.meta.observation.visit_group,
                model.meta.observation.sequence_id,
                model.meta.observation.activity_id,
                model.meta.observation.exposure_number
            )

            if meta_ids in group_ids:
                gid = group_ids[meta_ids]
            else:
                gid = str(uuid.uuid4())
                # next loop will almost never execute as UUIDs should be
                # unique.
                while gid in group_ids.values():
                    gid = str(uuid.uuid4())  # pragma: no cover
                group_ids[meta_ids] = gid

        except Exception as e:
            gid = 'None'
            log.warning("Unable to assign a 'tweakwcs_group_id' to image "
                        "'{}' due to '{}'".format(model.meta['filename'], e))
            log.warning("'tweakwcs_group_id' for image '{}' will be set "
                        "to None".format(model.meta['filename']))

        finally:
            if "tweakwcs_group_id" not in model.meta:  # pragma: no branch
                twgid_schema = {
                    "type": "object",
                    "properties": {
                        "meta": {
                            "type": "object",
                            "properties": {
                                "tweakwcs_group_id": {
                                    "title": "tweakwcs group ID",
                                    "type": "string",
                                    "fits_keyword": "TWEAKGID"
                                }
                            }
                        }
                    }
                }
                model.extend_schema(twgid_schema)
            model.meta.tweakwcs_group_id = gid

            if cl:
                model.save(model.meta.filename)
