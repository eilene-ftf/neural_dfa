import nengo
import nengo_spa as spa
import numpy as np
from typing import Iterable
from dataclasses import dataclass

from integer_states import BinaryState

@dataclass
class InputVar:
    name: str
    sac: str

@dataclass
class StateVar:
    name: str
    sac: str

# Strings designate state variables, while semantic pointers are found in the vocabulary
# floats designate the values of 1D populations, while ints designate meaningless
# state-value identifiers
type StateType = int|float|spa.SemanticPointer
type STTState = tuple[StateType|None, ...]
type STTVar = tuple[StateType|StateVar|InputVar|None, ...]

class DFA(spa.Network):
    """Deterministic Finite-state Automaton.
    Implements a DFA from a provided state-transition table, neurally.

    This class is an absolute mess, but I'll try to explain it:

    Args:
        statevars:  A list of tuples (str, type) consisting of the names state variables are to take on,
                    and their (symbolic) types. Allowed types are `spa.SemanticPointer`, `float`, `int`.
                    A state-transition table consists of conditions that are sensitive to the values of
                    state variables, and actions that update the values of state variables.
        inputs:     A list of tuples (str, int) indicating the name and dimension of input nodes. The 
                    nodes can be accessed for a DFA named `dfa` via `dfa.input_{name}` or `dfa.inputs[name]`.
        outputs:    A list of tuples (str, str) indicating the name of an output node and a state variable
                    that should be forwarded to it.
        table:      A state-transition table provided as a dictionary of the form `{condition: action, ...}`
                    where `condition` is a tuple of state-values for each state variable (in the same order
                    as state vars) that will trigger the condition, and `action` is a tuple of state-values
                    or `InputVar`s or `StateVar`s to update the corresponding state variables with. If the 
                    action is an `InputVar` or a `StateVar`, the input or state varialbe with the name `name`
                    will be conditionally routed to the target state varialbe via the sacrificial state 
                    variable `sac`. The state transition table must have a condition for all accessible 
                    states, or else the DFA will break. To give it an accepting state, just associate a 
                    condition with itself. IMPORTANT: conditions should ALWAYS be mutually exclusive; you
                    WILL break this if they aren't.
        voc:        The vocabulary for the network. You have to provide exactly one SPA `Vocabulary`, no more
                    and no less, because I hate Hitler and you should too.
        start:      The initial state of the DFA (values in the same order as `statevars`).
        neurons:    Number of neurons per dimension anywhere that calls for it (except integer state variables)
        synapse:    The delay synapse between the determination that a condition is met and the selection 
                    of an action.
        thresh:     I'm gonna be real with you this one's a little arcane. It's used to clip action 
                    activations so they don't occur unless one condition is much more strongly met than
                    all others.

    Using the state-transition table, generates a DFA like so:
    1.  Creates input nodes, as specified.
    2.  Creates two subnetworks: `selector` and `transitions`; their behaviour is modeled on the SPA
        `ActionSelection` network, but they work slightly differently. Essentially, condition networks
        are plugged into the `selector`, which determines which are most strongly met, then this feeds 
        into `transitions` with two connections: first direct, then one laterally inhibiting its neighbours.
        The `selector` is a winner-take-all network whose inputs are also competitive, however, it is 
        upper-bounded rather than lower-bounded, so the winning input is associated with an exact value, while 
        there's no specific expected value for losing inputs. It feeds into the transitions network, which is
        lower-bounded, so the result is that there's an upper and lower bound on final activation, and that
        activations are pushed towards one of two thresholds (at 0 and 1), and the output is one-hot.
    3.  Creates a subnetwork of state variables. These are all just SPA `State` objects. For each passed type 
        in `statevars`, if type `SemanticPointer` was given, adds a `State` with vocabulary `voc`. 
        If type `float` was given, adds a one-dimensional `State`. If type `int` was given, I'm still figuring 
        this one out.
    4.  Creates a `Node` object that pushes the initial state to the state variables.
    5.  Creates output nodes, as specified.
    6.  Connects conditions to the selector.
    7.  Creates a subnetwork of conditions. This uses the BigWedge object below to compute an iterated 
        conjunction of input values, which should be 1d and equal to almost-exactly 0 or 1. Will yield
        either 0 if the condition is unmet, or 1 if it is met at the output. Conditions are set so as to be
        met when all the corresponding values of state variables approximately equal the values specified
        in the state-transition table condition.
    8.  Creates a subnetwork of gates to be used for conditional routing (if any). Otherwise vestigial.
    9.  Connects state variables to conditions in the condition subnetwork according to conditions specified
        in the state-transition table. i.e., when each state variable has the value specified in the 
        state-transition table condition field, the corresponding condition network will output a 1, and 
        otherwise, it will be 0.
    10. Connects the one-hot action activation in transitions to the relevant state variables or gates 
        (adds gate to the gate subnetwork). Actions can either project values to state variables, or
        conditionally route inputs or other state variables. If the former, the values just appear as 
        connection weights input to the state variables. If the latter, the source is connected to a 
        sacrificial state varaible, and a gate is inhibitorily connected to the neurons of the sacrificial 
        variable, which is then connected to the sink state variable. Gates are set high by default, so
        the sacrificial variable serves as a passthrough that is usually inactivated. The action activation 
        is inhibitorily connected to the gate, so when the action is triggered, the gate is inactivated,
        disinhibiting the sacrificial state variable, which passes through its value.
    """
    def __init__(self, 
                 statevars:list[tuple[str, type]],
                 inputs:list[tuple[str, int]],
                 outputs: list[tuple[str, str]], 
                 table:dict[STTState, STTVar],
                 voc:spa.Vocabulary,
                 start:STTState|None=None,
                 neurons:int=50, 
                 synapse:float=0.01,
                 thresh:float=0.2,
                 label:str="DFA", 
                 *args, 
                 **kwargs
                 ):
        super(spa.Network, self).__init__(label=label, *args, **kwargs)

        self.neur = neurons
        self.dim = voc.dimensions
        self.synapse = synapse
        self.voc = voc

        with self:
            # Create input nodes
            self.inputs = {}
            self.ordered_inputs = []
            for name, d in inputs:
                self.inputs[name] = nengo.Node(size_in=d, label=f"input_{name}")
                self.ordered_inputs.append(self.inputs[name])
                setattr(self, f"input_{name}", self.inputs[name])

            # Set up the action selection
            # The extreme parameter values to get action activations to be as close to binary as possible
            self.selector = spa.networks.selection.WTA(n_neurons=self.neur, 
                                                       n_ensembles=len(table), 
                                                       threshold=0
                                                       )
            self.transitions = nengo.networks.EnsembleArray(self.neur, 
                                                            len(table), 
                                                            intercepts=nengo.dists.Uniform(thresh*2, 1.0), 
                                                            encoders=nengo.dists.Choice([[1.0]]),
                                                            radius=66/96,
                                                            )
            with self.transitions:
                self.transitions.bias = nengo.Node([1])
            # Positive activation for each condition
            nengo.Connection(self.selector.output, 
                             self.transitions.input, 
                             synapse=self.synapse, 
                             transform=-16/(thresh)
                             )
            # Inhibit neighbours
            nengo.Connection(self.selector.output, 
                             self.transitions.input, 
                             synapse=self.synapse, 
                             transform=16/(thresh*(len(table)-1))*(1-np.eye(len(table)))
                             )
            nengo.Connection(self.transitions.bias, 
                             self.transitions.input, 
                             transform=(-64*thresh)*np.ones((len(table), 1))
                             )


            # There's also competitive activation on the selector input
            self.select_in = nengo.Node(size_in=len(table))
            nengo.Connection(self.select_in, self.selector.input, transform=(1/(len(table)-1)) * (1-np.eye(2)))
            nengo.Connection(self.select_in, self.selector.input, transform=-1)

            # Set up the state variables
            with spa.Network() as self.statevars:
                self.statevars.svs = {}
                self.statevars.ordered_svs = []
                int_vars = {}
                last_map = {}
                int_map = {}
                self.vocab_maps = {}

                for condition, action in table.items():
                    for i, n in enumerate(condition):
                        if isinstance(n, int):
                            if i in int_vars:
                                if n not in int_vars[i]:
                                    int_vars[i].add(n)
                                    last_map[i][-1] += 1
                                    if last_map[i][-1] == 2:
                                        last_map[i][-1] = 0
                                        last_map[i].append(0)
                                    int_map[(i, n)] = last_map[i]
                            else:
                                int_vars[i] = {n}
                                last_map[i] = [0]
                                int_map[(i, n)] = [0]
                
                for condition, action in table.items():
                    for i, n in enumerate(condition):
                        if isinstance(n, int):
                            size = int(np.ceil(np.log2(len(int_vars[i]))))
                            cursz = len(int_map[(i, n)])
                            if not isinstance(int_map[(i, n)], np.ndarray):
                                int_map[(i, n)] = np.array(int_map[(i, n)] + [0 for _ in range(size-cursz)])
                            if i not in self.vocab_maps:
                                self.vocab_maps[i] = spa.Vocabulary(size)
                            if f"Sym_{n}" not in self.vocab_maps[i]:
                                self.vocab_maps[i].add(f"Sym_{n}", int_map[(i, n)])

                for i, (name, stype) in enumerate(statevars):
                    if stype is spa.SemanticPointer:
                        self.statevars.svs[name] = spa.State(self.voc, label=name)
                    elif stype is float:
                        self.statevars.svs[name] = spa.State(1, subdimensions=1, label=name)
                    elif stype is int:
                        self.statevars.svs[name] = BinaryState(self.vocab_maps[i], label=name)
                    else:
                        raise NotImplementedError(stype)

                    self.statevars.ordered_svs.append(self.statevars.svs[name])
                    setattr(self.statevars, name, self.statevars.svs[name])
                self.statevars.int_state_bias = nengo.Node([1])

            if start:
                self.initial_state = nengo.Node(output=lambda t: 1 if t < 0.05 else 0)
                for i, value in enumerate(start):
                    if isinstance(value, spa.SemanticPointer):
                        nengo.Connection(self.initial_state,
                                         self.statevars.ordered_svs[i].input,
                                         transform=value.v.reshape((self.dim, 1))
                                         )
                    elif isinstance(value, float):
                        nengo.Connection(self.initial_state,
                                         self.statevars.ordered_svs[i].input,
                                         transform=value
                                         )
                    elif isinstance(value, int):
                        nengo.Connection(self.initial_state,
                                         self.statevars.ordered_svs[i].input,
                                         transform=int_map[(i, value)].reshape((len(int_map[(i, value)]), 1))
                                         )
                    elif not value:
                        pass
                    else:
                        raise NotImplementedError(type(value))


            self.output_nodes = {}
            self.ordered_outputs = []
            for node_name, sv_name in outputs:
                self.output_nodes[node_name] = nengo.Node(size_in=len(self.statevars.svs[sv_name].output), 
                                                     label=f"output_{node_name}"
                                                     )
                self.ordered_outputs.append(self.output_nodes[node_name])
                tf = 1
                if isinstance(self.statevars.svs[sv_name], BinaryState):
                    tf = 1/0.7
                nengo.Connection(self.statevars.svs[sv_name].output, self.output_nodes[node_name], transform=tf)



            # Set up the conditionals
            with spa.Network() as self.conditions:
                self.conditions.conds = []
                self.conditions.inputs = []
                self.conditions.output = nengo.Node(size_in=len(table))
                for i, (condition, action) in enumerate(table.items()):
                    # a condition is a tuple in the same order as statevars
                    self.conditions.conds.append(BigWedge(len(condition), 
                                                          label=f"condition {i}",
                                                          ignore=np.where([c is None for c in condition])[0]))
                    self.conditions.inputs.append(nengo.Node(size_in=len(condition), label=f"input {i}"))
                    nengo.Connection(self.conditions.inputs[i], self.conditions.conds[i].input)

                    nengo.Connection(self.conditions.conds[i].output, self.conditions.output[i])
                    setattr(self.conditions, f"condition_{i}", self.conditions.conds[i])
                    setattr(self.conditions, f"input_{i}", self.conditions.inputs[i])

            nengo.Connection(self.conditions.output, self.select_in)

            def scale(ptr:spa.SemanticPointer) -> np.float64:
                return ptr.v @ ptr.v

            def weightby(ptr:spa.SemanticPointer):
                return (1/(0.75*scale(ptr))) * ptr.v.reshape((1, self.dim))

            with spa.Network() as self.gates:
                self.gates.gatebias = nengo.Node([1])
                self.gates.gate_pops = []

            # Connect statevars to conditionals weighted by preferred value,
            # Connect selected actions to statevars
            for i, ((condition, action), ci, to) in enumerate(zip(table.items(), self.conditions.inputs, self.transitions.output)):
                # Connect statevars to conditionals
                for j, (sv, c, inp) in enumerate(zip(self.statevars.ordered_svs, condition, ci)):
                    if isinstance(c, spa.SemanticPointer):
                        nengo.Connection(sv.output,
                                         inp,
                                         transform=weightby(c)
                                         )
                    elif isinstance(c, float):
                        nengo.Connection(sv.output,
                                         inp
                                         )
                    elif isinstance(c, int): # this may be broken but who knows
                        weightvec = (2/(0.7 * len(int_map[(j, c)])) * (2*int_map[(j, c)]-1))
                        nengo.Connection(sv.output,
                                         inp,
                                         transform=(weightvec.reshape((1, len(weightvec)))
                                                    )
                                         )
                        nengo.Connection(self.statevars.int_state_bias,
                                         inp,
                                         transform=(-1 * np.sign(weightvec))
                                         )
                    elif not c:
                        pass
                    else:
                        raise NotImplementedError(type(c))

                # Connect actions to statevars
                for j, (sv, a) in enumerate(zip(self.statevars.ordered_svs, action)):
                    if isinstance(a, spa.SemanticPointer): # project a symbol
                        nengo.Connection(to,
                                         sv.input,
                                         transform=a.v.reshape((self.dim, 1))
                                         )
                    elif isinstance(a, InputVar) or isinstance(a, StateVar): # route input/statevar to statevar via sacrificial statevar
                        sac = self.statevars.svs[a.sac]
                        inp = None
                        if isinstance(a, InputVar):
                            inp = self.inputs[a.name]
                            nengo.Connection(inp, sac.input)
                        elif isinstance(a, StateVar):
                            inp = self.statevars.svs[a.name]
                            inp >> sac
                        sac >> self.statevars.ordered_svs[j]
                        with sac.state_ensembles:
                            sac.state_ensembles.add_neuron_input()
                        sac_neurons = len(sac.state_ensembles.neuron_input)
                        with self.statevars:
                            setattr(self.statevars, f"{a.sac}_neurons", nengo.Node(size_in=sac_neurons))
                        nengo.Connection(getattr(self.statevars, f"{a.sac}_neurons"),
                                         sac.state_ensembles.neuron_input
                                         )
                        with self.gates as gates:
                            gates.gate_pops.append(nengo.Ensemble(self.neur, dimensions=1))
                        nengo.Connection(self.gates.gatebias, self.gates.gate_pops[-1])
                        nengo.Connection(self.gates.gate_pops[-1], 
                                         getattr(self.statevars, f"{a.sac}_neurons"),
                                         transform=-1 * np.ones((sac_neurons, 1)))

                        nengo.Connection(to,
                                         self.gates.gate_pops[-1].neurons,
                                         transform=-2*np.ones((self.neur, 1)))
                    elif isinstance(a, float): # project a float value
                        nengo.Connection(to,
                                         sv.input,
                                         transform=a
                                         )
                    elif isinstance(a, int): # project the binary map of the integer
                        nengo.Connection(to,
                                         sv.input,
                                         transform=int_map[(j, a)].reshape((sv.vocab.dimensions, 1))
                                         )
                    elif not a: # If None, then we're skipping this one
                        pass
                    else:
                        raise NotImplementedError(type(a))


class BigWedge(spa.Network):
    """Computes an iterated conjunction (kind of).
    (Relatively) efficiently computes the conjunction of `dim` conditions. 
    The output will be 1 if all input dimensions are 1 (or very close).
    Expects as input dot products bounded on [0, 1]. 
    I have not clipped the inputs *at* 1, so you can technically go over, but don't. It will break.
    If your inputs are in expectation 0.8, multiply them by 1/0.8 on the way in.
    The tolerance for deviation here is exceedingly small, your inputs really all need to be
    upper bounded at 1.

    How it works:
    - There's a population `hot` that's biased to 1.
    - There's a popuation `suppress` that inhibits the neurons of `hot`.
        - `suppress` is saturated by another bias node with an input of `dim`.
        - `suppress` therefore completely inhibits `hot`.
    - The `dim` input dimensions are each individually inhibit the neurons of suppress.
        - When all dimensions are 1, suppress is inhibited.
        - `hot` is therefore disinhibited, and the value 1 can pass through to the output.

    It is basically a weird and gate.
    """
    def __init__(self,
                 dim:int,
                 neur:int=50,
                 thresh:float=0.2,
                 ignore:Iterable[int]=[],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.neur = neur
        self.dim = dim
        
        with self:
            self.input = nengo.Node(size_in=self.dim)
            self.output = nengo.Node(size_in=1)
            self.saturate = nengo.Node([self.dim-len(ignore)])
            self.hotbias = nengo.Node([1])
            self.suppress = nengo.Ensemble(n_neurons=self.neur, 
                                             dimensions=1, 
                                             intercepts=nengo.dists.Uniform(thresh, 1.0),
                                             encoders=nengo.dists.Choice([[1.0]])
                                             )
            self.hot = nengo.Ensemble(n_neurons=self.neur, 
                                        dimensions=1,
                                        intercepts=nengo.dists.Uniform(thresh, 1.0),
                                        encoders=nengo.dists.Choice([[1.0]])
                                        )
            inp_weights = -np.ones((self.neur, self.dim))
            inp_weights[*ignore] = 0
            nengo.Connection(self.saturate, self.suppress.neurons, transform=np.ones((self.neur, 1))) 
            nengo.Connection(self.input, self.suppress.neurons, transform=inp_weights)
            nengo.Connection(self.suppress, self.hot.neurons, transform=-2*np.ones((self.neur, 1)))
            nengo.Connection(self.hotbias, self.hot)
            nengo.Connection(self.hot, self.output)
