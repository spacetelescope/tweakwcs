"""
Initial implementation of a function to fit multiple outputs 
using astropy.modeling
"""

from astropy.modeling import FittableModel
from astropy.modeling.models import Mapping, Const1D
from astropy.modeling.models import math
import numpy as np


class _Index(FittableModel):

    def __init__(self):
        self._inputs = ()
        self._outputs = ()
        self._n_inputs = 1
        self._n_outputs = 1
        super().__init__()
        self.inputs = ('x',)
        self.outputs = ('x',)

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_outputs(self):
        return self._n_outputs

    def __repr__(self):
        return f'<Index()>'

    def evaluate(self, *args):
        if len(args) != 1:
            raise TypeError(f'Index model only takes 1 input, got {len(args)}')
        input = args[0]
        if np.isscalar(input):
            return 0
        inp_array = np.array(input)
        if len(inp_array.shape) != 1:
            raise ValueError('Index currently only handles 1D arrays')
        size = inp_array.shape[0]
        return np.arange(size)

    def inverse(self):
        raise NotImplementedError('Index model has no inverse by design')


def _multi_output_fit(fitter, model, inputs, outputs, **kwargs):
    """
    Function to enable fits to models with more than one output variable.

    Parameters
    ----------
    fitter: instance of an Astropy fitter class
    model: instance of an Astropy model with 2 or more output variables to be fit
    inputs: list
        A list of input numpy arrays or array compatible values.
    outputs: list
        A list of output values to be fit corresponding to the input value provided.
        Must have at least two output arrays.
    kwargs: dict
        Any optional arguments to pass to the fitter.

    Returns
    -------
    Returns the fit model

    Notes
    -----
    While the machinery to prepare the model and its corresponding values to fit 
    is general for any number of inputs and outputs, currently the astropy fitters
    only support two inputs.
    """
    # Determine number of inputs and ouputs of model
    n_inputs = model.n_inputs
    n_outputs = model.n_outputs
    if n_outputs < 2:
        raise ValueError(
            "Pointless to use this fitting function for less than 2 output values")
    # Assumes all inputs and ouputs are 1D arrays and aleady broadcasted
    # Generate input repetition (based on number of outputs)
    expanded_inputs = []
    expander = np.ones((n_outputs, 1))
    for input in inputs:
        newinput = input.copy()
        newinput.shape = (1, input.shape[0])
        expanded_inputs.append(np.ravel(expander * newinput))
    newoutputs = np.concatenate(outputs)
    # Handle the same with weights, if present
    if 'weights' in kwargs:
        newweights = np.ravel(expander * weights)
    else:
        newweights = None
    inputsize = inputs[0].shape[0]
    # Now build the relevant model from the one to be be fit with
    switch = (_Index() & Const1D(inputsize, fixed={
              'amplitude': True})) | math.Floor_divideUfunc()
    mod = (Mapping((0, 0, 0), n_inputs) |
           (switch & Const1D(0, fixed={'amplitude': True}))
           | math.EqualUfunc()) * (model | Mapping((0,), n_outputs))
    for i in range(1, n_outputs):
        mod += (Mapping((0, 0, 0), n_inputs) |
                (switch & Const1D(i, fixed={'amplitude': True}))
                | math.EqualUfunc()) * (model | Mapping((i,), n_outputs))
    mod._fittable = True
    # Do the fit
    # Not general due to limitations of astropy fits currently
    fitmod = fitter(mod, expanded_inputs[0],
                    expanded_inputs[1], newoutputs, weights=newweights, **kwargs)
    return fitmod.left.right.left  # Fragile?
