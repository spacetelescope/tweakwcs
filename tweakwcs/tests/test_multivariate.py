import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.modeling.models import Shift, Rotation2D
from astropy.modeling.fitting import LevMarLSQFitter
from ..multivariate import _multi_output_fit


def test_multivariate():
    inputs = [np.array([10., 10., 20., 20.]), np.array([10., 20., 20., 10.])]
    outputs = [np.array([8.06101731, 0.98994949, 8.06101731, 15.13208512]),
               np.array([12.16223664, 19.23330445, 26.30437226, 19.23330445])]
    model = (Shift() & Shift()) | Rotation2D()
    fit_model = _multi_output_fit(LevMarLSQFitter(), model, inputs, outputs)
    assert_allclose(fit_model.parameters, np.array(
        [4.3, -7.1, 45.]), rtol=1e-5, atol=1e-5)


def test_single_output():
    inputs = [[0, 0, 0, 0]]
    outputs = [[0, 0, 0, 0]]
    model = Shift()
    with pytest.raises(ValueError):
        fit_model = _multi_output_fit(LevMarLSQFitter, model, inputs, outputs)
