"""
A module containing unit tests for the `wcsutil` module.

Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
from itertools import groupby

import pytest

from tweakwcs.utils.jwst_utils import assign_jwst_tweakwcs_groups


@pytest.fixture(scope='function')
def data_model_list():
    pytest.importorskip("jwst")
    from jwst.datamodels import DataModel, ModelContainer

    models = []
    for k in range(6):
        m = DataModel()
        m.meta.observation.program_number = '0001'
        m.meta.observation.observation_number = '1'
        m.meta.observation.visit_number = '1'
        m.meta.observation.visit_group = '1'
        m.meta.observation.sequence_id = '01'
        m.meta.observation.activity_id = '1'
        m.meta.observation.exposure_number = '1'
        m.meta.instrument.name = 'NIRCAM'
        m.meta.instrument.channel = 'SHORT'
        m.meta.filename = 'file{:d}.fits'.format(k)
        models.append(m)

    models[-3].meta.observation.observation_number = '2'
    models[-2].meta.observation.observation_number = '3'
    models[-1].meta.observation.observation_number = '3'

    return ModelContainer(models)


@pytest.fixture
def defective_data_model():
    class DummyMeta(dict):
        def __init__(self, *args):
            super().__init__(args)
            dict.__setitem__(self, 'filename', 'dummy.file')

        @property
        def tweakwcs_group_id(self):
            return dict.__getitem__(self, 'tweakwcs_group_id')

        @tweakwcs_group_id.setter
        def tweakwcs_group_id(self, v):
            dict.__setitem__(self, 'tweakwcs_group_id', v)

    class BrokenModel():
        def __init__(self):
            self._meta = DummyMeta()

        @property
        def meta(self):
            return self._meta

        def extend_schema(self, *args, **kwargs):
            pass

    return BrokenModel()


def test_assign_jwst_tweakwcs_groups_fail(defective_data_model):
    pytest.importorskip("jwst")
    assign_jwst_tweakwcs_groups([defective_data_model])
    assert defective_data_model.meta.tweakwcs_group_id == 'None'


def test_assign_jwst_tweakwcs_groups(data_model_list, monkeypatch):
    """ Imitate string file names and file IO: """
    jwst = pytest.importorskip("jwst")
    models = {m.meta.filename: m for m in data_model_list}
    monkeypatch.setattr(jwst.datamodels.DataModel, 'save', lambda s, m: m)
    monkeypatch.setattr(jwst.datamodels, 'open', lambda fname: models[fname])

    assign_jwst_tweakwcs_groups(list(models.keys()))

    gids = sorted([model.meta.tweakwcs_group_id for model in data_model_list])
    assert sorted([len(list(g)) for k, g in groupby(gids)]) == [1, 2, 3]
