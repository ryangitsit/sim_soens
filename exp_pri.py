import numpy as np
import matplotlib.pyplot as plt

from soen_plotting import raster_plot, activity_plot

from super_input import SuperInput
from params import net_args 

from super_library import NeuralZoo
from super_functions import *
from soen_sim import network, dendrite, HardwareInTheLoop
from soen_component_library import common_synapse


'''
Notes:
 - dend_load_arrays_thresholds_saturations
 - Fix array attachment default ib
 - smooth soen_sim before new pri
 - self.name = 'unnamed_dendrite__{}'.format(self.unique_label) -- mystery dend?
'''

def integration():
    times = np.arange(0,300,100)
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)

    weights = [[[0.5]]]
    loops_present = [[['pri']]]

    # no synapse on soma, synapse on first (only) dend
    syn_struct = [[[[0]],[[1]]]] 

    mono_dend = NeuralZoo(type="custom",weights=weights,synaptic_structure=syn_struct,s_th=1,types=loops_present) 
    print(mono_dend.dendrite_list)

    # adding signal only to dendrite at the 1rst layer (soma at 0th layer)
    mono_dend.synapses[0][1][0][0].add_input(input.signals[0])

    net = network(sim=True,dt=.1,tf=150,nodes=[mono_dend],new_way=True)
    print(mono_dend.dendrite_list)
    title = "Monosynaptic Neuron with Intermediate Dendrite"

    # weighting is turned off here, because for only 1 dendrite, phi_r = dend.signal*weighting
    mono_dend.plot_neuron_activity(net,phir=True,title=title,weighting=False)
integration()










