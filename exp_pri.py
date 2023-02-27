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

def dend_pri():
    # times = np.arange(0,1000,100)
    times = np.array([200,400,600])
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)
    # input = SuperInput(channels=1, type='constant',phi_app=0.3)

    weights = [[[0.3]]]
    loops_present = [[['pri']]]

    syn_struct = [[[[0]],[[.55]]]] 

    plt.figure(figsize=(6.4*1.2,4.8*1.2))
    PHIP = [-0.13157894736842107,0.02631578947368418,0.18421052631578938,0.3421052631578947]
    for phip in PHIP:
        mono_dend = NeuralZoo(type="custom",weights=weights,synaptic_structure=syn_struct,s_th=1,types=loops_present,
                                            beta_di=2*np.pi*1e4,tau_di=250,ib_di=2.4,pri=True, offset_flux=0) 
        mono_dend.dendrites[1][0][0].phi_p = phip
        
        mono_dend.synapses[0][1][0][0].add_input(input.signals[0])
        net = network(sim=True,dt=.1,tf=1400,nodes=[mono_dend],new_way=True)
        # mono_dend.plot_neuron_activity(net,phir=True,weighting=False,input=input)
        Ic = mono_dend.dendrites[1][0][0].Ic
        plt.plot(net.t,mono_dend.dendrites[1][0][0].s*Ic,label=f'$\phi_p=${phip}')
        plt.legend()
    plt.show()

# dend_pri()



def offset():
    # times = np.arange(0,1000,100)
    times = np.array([200,400,600])
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)

    weights = [[[0.3]]]
    syn_struct = [[[[0]],[[1]]]] 

    plt.figure(figsize=(6.4*1.2,4.8*1.2))
    PHIP = np.arange(-0.15,.16,.05)
    # PHIP = [1]
    for phip in PHIP:
        mono_dend = NeuralZoo(type="custom",weights=weights,synaptic_structure=syn_struct,s_th=1,
                                            beta_di=2*np.pi*1e4,tau_di=250, offset_flux=phip) 
        mono_dend.synapses[0][1][0][0].add_input(input.signals[0])
        net = network(sim=True,dt=.1,tf=1400,nodes=[mono_dend],new_way=True)
        # mono_dend.plot_neuron_activity(net,phir=True,weighting=False,input=input)
        Ic = mono_dend.dendrites[1][0][0].Ic
        plt.plot(net.t,mono_dend.dendrites[1][0][0].s,label=f'$\phi$ offset = {np.round(phip,2)}')
        plt.legend()
    plt.show()
    # mono_dend.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=True)
offset()










