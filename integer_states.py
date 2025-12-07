import nengo
import nengo_spa as spa
import numpy as np

from nengo.exceptions import ValidationError
from nengo.networks.ensemblearray import EnsembleArray
from nengo.params import BoolParam, Default, IntParam, NumberParam

from nengo_spa.network import Network
from nengo_spa.networks import IdentityEnsembleArray
from nengo_spa.vocabulary import VocabularyOrDimParam

class BinaryState(Network):
    """
    This code is almost completely ripped from the nengo_spa source, with a minor
    edit to allow me to apply weird configuration things.
    Represents a single vector, with optional memory.

    This is a minimal SPA network, useful for passing data along (for example,
    visual input).

    Parameters
    ----------
    vocab : Vocabulary or int
        The vocabulary to use to interpret the vector. If an integer is given,
        the default vocabulary of that dimensionality will be used.
    subdimensions : int, optional (Default: 16)
        The dimension of the individual ensembles making up the vector.
        Must divide *dimensions* evenly. The number of sub-ensembles
        will be ``dimensions // subdimensions``.
    neurons_per_dimension : int, optional (Default: 50)
        Number of neurons per dimension. Each ensemble will have
        ``neurons_per_dimension * subdimensions`` neurons, for a total of
        ``neurons_per_dimension * dimensions`` neurons.
    feedback : float, optional (Default: 0.0)
        Gain of feedback connection. Set to 1.0 for perfect memory,
        or 0.0 for no memory. Values in between will create a decaying memory.
    represent_cc_identity : bool, optional
        Whether to use optimizations to better represent the circular
        convolution identity vector. If activated, the `.IdentityEnsembleArray`
        will be used internally, otherwise a normal
        `nengo.networks.EnsembleArray` split up regularly according to
        *subdimensions*.
    feedback_synapse : float, optional (Default: 0.1)
        The synapse on the feedback connection.
    **kwargs : dict
        Keyword arguments passed through to `nengo_spa.Network`.

    Attributes
    ----------
    input : nengo.Node
        Input.
    output : nengo.Node
        Output.
    """

    vocab = VocabularyOrDimParam("vocab", default=None, readonly=True)
    subdimensions = IntParam("subdimensions", default=1, low=1, readonly=True)
    neurons_per_dimension = IntParam(
        "neurons_per_dimension", default=1, low=1, readonly=True
    )
    feedback = NumberParam("feedback", default=0.0, readonly=True)
    feedback_synapse = NumberParam("feedback_synapse", default=0.1, readonly=True)
    
    def __init__(
        self,
        vocab=Default,
        subdimensions=Default,
        neurons_per_dimension=Default,
        feedback=Default,
        feedback_synapse=Default,
        **kwargs,
    ):
        super(BinaryState, self).__init__(**kwargs)

        self.vocab = vocab
        self.subdimensions = subdimensions
        self.neurons_per_dimension = neurons_per_dimension
        self.feedback = feedback
        self.feedback_synapse = feedback_synapse
        dimensions = self.vocab.dimensions

        if dimensions % self.subdimensions != 0:
            raise ValidationError(
                f"Dimensions ({dimensions}) must be divisible by "
                f"subdimensions ({self.subdimensions})",
                attr="dimensions",
                obj=self,
            )

        with self:
            with binary():
                self.state_ensembles = EnsembleArray(
                    self.neurons_per_dimension * self.subdimensions,
                    dimensions // self.subdimensions,
                    ens_dimensions=self.subdimensions,
                    label="state",
                )

            if self.feedback is not None and self.feedback != 0.0:
                nengo.Connection(
                    self.state_ensembles.output,
                    self.state_ensembles.input,
                    transform=self.feedback,
                    synapse=self.feedback_synapse,
                )

        self.input = self.state_ensembles.input
        self.output = self.state_ensembles.output
        self.declare_input(self.input, self.vocab)
        self.declare_output(self.output, self.vocab)

def binary():
    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].intercepts = nengo.dists.Choice([0.0])
    config[nengo.Ensemble].encoders = nengo.dists.Choice([[1.0]])
    config[nengo.Ensemble].eval_points = nengo.dists.Uniform(0, 1)
    config[nengo.Ensemble].max_rates = [400]
    return config

"""
model = spa.Network()
with model:
    n = 5
    inp = nengo.Node([1] * n)

    with binary():
        integer2 = BinaryState(n,
                             subdimensions=1,
                             neurons_per_dimension=1
                             )

    nengo.Connection(inp, integer2.input)
    dip = nengo.Node(size_in=n, output=lambda t, x: x @ np.array([1, -1, 1, 1, -1]))
    bias = nengo.Node([1])
    nengo.Connection(bias, dip, transform=-1/n * np.ones((n, 1)))
    nengo.Connection(integer2.output, dip, transform=2/(0.7 * n))
"""
