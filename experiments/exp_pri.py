import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../sim_soens')
sys.path.append('../')

from sim_soens.soen_plotting import raster_plot, activity_plot
from sim_soens.super_input import SuperInput
from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *
from sim_soens.soen_components import network, dendrite, HardwareInTheLoop


'''
Notes:
 - dend_load_arrays_thresholds_saturations
 - Fix array attachment default ib
 - smooth soen_sim before new pri
 - self.name = 'unnamed_dendrite__{}'.format(self.unique_label) -- mystery dend?
'''

def main():
    def dend_pri(input):
        '''
        Deprecated
        '''
        weights = [[[1]]]
        loops_present = [[['pri']]]

        syn_struct = [[[[0]],[[1]]]]

        signals = []
        plt.figure(figsize=(6.4*1.2,4.8*1.2))
        # PHIP = [-0.13157894736842107,0.02631578947368418,0.18421052631578938,0.3421052631578947]
        # PHIP = np.arange(-.35,.35,.05)
        PHIP = np.arange(0,.35,.05)
        for phip in PHIP:
            mono_dend = SuperNode(weights=weights,synaptic_structure=syn_struct,s_th=1,types=loops_present,
                                                beta_di=2*np.pi*1e4,tau_di=250,ib_di=2.4,pri=True, offset_flux=0) 
            mono_dend.dendrites[1][0][0].phi_p = phip
            
            mono_dend.synapses[0][1][0][0].add_input(input.signals[0])
            net = network(sim=True,dt=.1,tf=1400,nodes=[mono_dend],new_way=True)
            # mono_dend.plot_neuron_activity(net,phir=True,weighting=False,input=input)
            Ic = mono_dend.dendrites[1][0][0].Ic
            signals.append(mono_dend.dendrites[1][0][0].s*Ic)
            plt.plot(net.t,mono_dend.dendrites[1][0][0].s*Ic,label=f'$\phi_p=${np.round(phip,2)}')
            plt.legend()
        plt.xlabel("Simulation Time (ns)")
        plt.ylabel("Signal (Ic)")
        plt.title("Plasticity via PRI Dendrite")
        plt.show()
        signals.append(net.t)
        signals = np.array(signals)
        # from super_functions import picklit
        # picklit(signals,"results","pri_test_arrays")


    def offset(input):
        '''
        Current primary plasticity mechanism
        '''
        weights = [[[0.3]]]
        syn_struct = [[[[0]],[[1]]]] 

        plt.figure(figsize=(6.4*1.2,4.8*1.2))
        # PHIP = np.arange(-0.15,.36,.05)
        PHIP = np.arange(-.15,.16,.05)
        # PHIP = [1]
        for phip in PHIP:
            mono_dend = SuperNode(weights=weights,synaptic_structure=syn_struct,s_th=1,
                                                beta_di=2*np.pi*1e4,tau_di=250, offset_flux=phip)#,ib_di=1.42) 
            mono_dend.synapses[0][1][0][0].add_input(input.signals[0])
            net = network(sim=True,dt=.1,tf=1400,nodes=[mono_dend],new_way=True)
            # mono_dend.plot_neuron_activity(net,phir=True,weighting=False,input=input)
            Ic = mono_dend.dendrites[1][0][0].Ic
            plt.plot(net.t,mono_dend.dendrites[1][0][0].s,label=f'$\phi$ offset = {np.round(phip,2)}')
            plt.legend()
        plt.xlabel("Simulation Time (ns)")
        plt.ylabel("Signal (Ic)")
        plt.title("Plasticity via Flux Offset")
        plt.show()
        # mono_dend.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=True)


    def biasing(input):
        '''
        Also works as a plasticity mechanism
        '''
        syn_struct = [[[[1]]]]
        # plt.figure(figsize=(12,4))
        plt.figure(figsize=(8,6))
        BIAS = np.arange(1.4,2,.09)
        for i in BIAS:
            mono_point = SuperNode(synaptic_structure=syn_struct,s_th=100,
                                beta_ni=2*np.pi*1e4,tau_ni=250,ib_n=i)
            mono_point.synapses[0][0][0][0].add_input(input.signals[0])
            net = network(sim=True,dt=.1,tf=1400,nodes=[mono_point])
            # signals.append(mono_point.s)
            Ic = mono_point.dendrites[0][0][0].Ic
            plt.plot(net.t,mono_point.neuron.dend_soma.s*Ic,label=f"bias current = {np.round(i,2)}")
        # print(mono_point.neuron.__dict__.keys()
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.subplots_adjust(right=.8)
        # plt.subplots_adjust(bottom=.15)
        plt.legend()
        plt.xlabel("Simulation Time (ns)")
        plt.ylabel("Signal (Ic)")
        plt.title("Plasticity via Bias Change")
        plt.show()


    times = np.arange(0,1000,51)
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)

    # dend_pri(input)
    offset(input)
    biasing(input)

if __name__=='__main__':
    main()