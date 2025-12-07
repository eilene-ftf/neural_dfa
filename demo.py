import nengo
import nengo_spa as spa
import numpy as np

from neural_dfa import DFA, InputVar, StateVar

d = 128
voc = spa.Vocabulary(d)
voc.populate("Apple;Banana;Cherry;Durian;Elderberry;Fig;Grape;Hawthorn")
statevars = [("statevar1", spa.SemanticPointer),
             ("statevar2", spa.SemanticPointer),
             ("statevar3", spa.SemanticPointer),
             ("statevar4", int),
             ("dummyin", spa.SemanticPointer),
             ("bananapass", spa.SemanticPointer)
            ]

table = {
        (voc["Apple"], voc["Banana"], None, 1): (voc["Banana"], voc["Apple"], StateVar("statevar1", "bananapass"), 0), 
        (voc["Banana"], voc["Apple"], None, 0): (voc["Apple"], voc["Banana"], InputVar("a", "dummyin"), 1)
        }

inputs = [
        ("a", d),
        ]

outputs = [("fruit", "statevar1"),
           ("otherfruit", "statevar2"),
           ("sometimesfruit", "statevar3"),
           ("strangefruit", "statevar4"),
           ]
with spa.Network() as model:
    dfa = DFA(statevars, inputs, outputs, table, voc, start=(voc["Apple"], voc["Banana"], None, 1)) 

    def direct_conf():
        conf = nengo.Config(nengo.Ensemble)
        conf[nengo.Ensemble].neuron_type = nengo.neurons.Direct()
        return conf

    with direct_conf(): 
        a = spa.State(voc)
        output_states = [spa.State(voc, label=outname) for outname, _ in outputs[:-1]]
        output_states.append(spa.State(len(dfa.output_nodes["strangefruit"]), subdimensions=1, label="strangefruit"))
    nengo.Connection(a.output, dfa.input_a) 

    
    for outnode, state in zip(dfa.ordered_outputs, output_states):
        nengo.Connection(outnode, state.input)
        

