"""
Initial implementation of a function to fit multiple outputs 
using astropy.modeling

Notes
-----

This presumes that the multiple outputs are given equal weight in
computing the least squares value. A future version may allow 
specifying different weights when the quantities involved are
not directly comparable. The motivating case for developing this
was to compute the proper transformation of 2D coordinates, and as
such, the least square computation is equivalent to computing the
least squares radial differences in position. This could be generalized
to 3D, obviously.

For simplicity, the following description assumes the above motivating
example where there are only two outputs to be fit. The code below
actually can support any number of outputs so long as the fitters
can support more than two input variables.
The trick employed here is to combine the two outputs as though 
the model only produces one output for the astropy model fitting.
This is done by doubling the inputs to the model (i.e., repeating
all inputs twice, i.e., concatenating the input array with itself).
Likewise, the "measured" first output is concatenated with the
"measured" second output. Finally, a new model is constructed from
the one to be fit by wrapping it with machinery that effectively
selects the first output for the first half of inputs, and selects
the second output for the second half of the inputs, thus when
computing residuals will end up summing the squared differences
for the first output with the squared differences with the
second output. Given the doubling of the number of input, 
one must be careful to interpret the actual sigma for the 
differences (in this case probably a square root of two factor
should be dividec).

There are extensive comments to highlight how this is done 
with the wrapping model.

When the fit is complete, the original model is extracted and
returned.

Ultimately we expect to move this to astropy, but until then
these functions and classes are private
"""

from astropy.modeling import FittableModel
from astropy.modeling.models import Mapping, Const1D
from astropy.modeling.models import math
import numpy as np

# Define the EqualUFunc

_EqualUfunc = math.ufunc_model('equal')



class _Index(FittableModel):
    """
    This is a model that accepts an input value or array of values and
    generates an array of values that gives the index value of each 
    input value. E.g., in effect an arange function that matches the 
    size of the input. This is useful for selecting parts of the input
    based on position.

    For example, if the input value is an array with 10 values, the
    evaluation of this model will return an array [0, 1, 2,...,9]
    regardless of what the inputs values are.
    """

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
    # This replicates the inputs by how many outputs are produced by 
    # the supplied model.
    expanded_inputs = []
    expander = np.ones((n_outputs, 1))
    for input in inputs:
        newinput = input.copy()
        newinput.shape = (1, input.shape[0])
        expanded_inputs.append(np.ravel(expander * newinput))
    # Here the outputs are combined into one array.
    newoutputs = np.concatenate(outputs)
    # Handle the same with weights, if present
    if 'weights' in kwargs:
        newweights = np.ravel(expander * weights)
    else:
        newweights = None
    inputsize = inputs[0].shape[0]
    # Now build the relevant model from the one to be be fit with.
    # This submodel takes two inputs and generates two outputs.
    # The first input is used by _Index to generate an array
    # that is essentially an arange array for the total length
    # of the input. The second inputs is ignored since Const1D
    # ignores it. The "arange" ouput of _Index() and the
    # constant value (set to the original input size) is fed
    # to floor_divide, which will produce an output array
    # where the first input has value 0, the next replication
    # of the input has value 1, and so on for however many replications
    # exist. This array is used in the constructed model as
    # a means of selecting which replication is used for the
    # computed output value.
    switch = (_Index() & Const1D(inputsize, fixed={
              'amplitude': True})) | math.Floor_divideUfunc()
    # Because accummulating models should start with a mode
    # instance, there is duplicated code here (I suppose
    # we could make the inital value some expansion of a
    # Constant of 0, instead).
    # The for loop handles the subsequent outputs by summing
    # selected region with also selecting the desired model
    # output. The only difference the initial model is that
    # the initial model selects the first output, the rest
    # are indexed by i.
    #
    # Detailed description of the selection operation:
    # Note the initial definiton of mod has two major 
    # components joined by multiplication. The simpler
    # of the two is the second, which is an evaluation
    # of the supplied model with a selection of the 
    # first output to be returned (different outputs in
    # the for loop are selected).
    # The first component is designed to multiply by
    # 1 all the model values for the first half of the
    # input arrays (i.e., the first instance of the replicated
    # original input). This is done by using the first
    # input variable for 3 input dimensions (which input
    # dimension to be used is immaterial, all that is actually
    # used is the length of the array). The first two dimensions
    # are used for the switch model, described above, which 
    # will provide values that corresponds to which
    # input replication that array position corresponds to.
    # The output of switch is a single dimension. The output
    # of switch is combined with the third input dimension
    # (whose value is irrelevant since it is being fed to
    # a constant model). These two output dimensions are 
    # fed to a model that checks for equality between the
    # two inputs. In this case, where the switch output
    # has a value of 0, and because the constant provided
    # is 0, the equality will be True, and in an expression
    # will be treated as 1. For the second replication of
    # input values, the switch output will be 1, and the 
    # result of the equality will be False, and thus be
    # treated a 0 in the math expression.
    # The consequence is that this model produces values
    # for the first output for the first occurance of 
    # original input arrays, and 0's for all subsequent
    # occurances.
    mod = (Mapping((0, 0, 0), n_inputs) |
           (switch & Const1D(0, fixed={'amplitude': True}))
           | _EqualUfunc()) * (model | Mapping((0,), n_outputs))
    # This for loop then adds models that will sum values for
    # the second occurances of the input arrays for the second
    # output of the original model, and so forth for all 
    # subseuent outputs.
    for i in range(1, n_outputs):
        mod += (Mapping((0, 0, 0), n_inputs) |
                (switch & Const1D(i, fixed={'amplitude': True}))
                | _EqualUfunc()) * (model | Mapping((i,), n_outputs))
    mod._fittable = True
    # Do the fit
    # Not general due to limitations of astropy fits currently
    fitmod = fitter(mod, expanded_inputs[0],
                    expanded_inputs[1], newoutputs, weights=newweights, **kwargs)
    # Return the original model seleted out of the fit model
    # This is correct so long as the structure of the compund model
    # above is not changed.
    return fitmod.left.right.left  # Fragile?
