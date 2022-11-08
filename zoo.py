#%%
import numpy as np

from super_input import SuperInput

from _util import physical_constants, set_plot_params, index_finder
from _util__soen import dend_load_rate_array, dend_load_arrays_thresholds_saturations
from soen_sim import input_signal, synapse, neuron, network
from soen_sim_lib__common_components__simple_gates import common_dendrite, common_synapse, common_neuron

from super_input import SuperInput
from params import default_neuron_params
from _plotting__soen import raster_plot

'''
Here a class for calling from a 'zoo' of possible neurons is implemented.

Plan:
 - Syntax for custom neuron calls based on dendritic structures and parameters.
 - Library of predefined neurons, both bio-inspired and engineering-specific
 - Should include a testing/plotting paradigm inherent in the class
'''

#%%
input = SuperInput(channels=9, type='random', total_spikes=50, duration=100)

print(input.spike_rows)
raster_plot(input.spike_arrays)

print(default_neuron_params)

#%%
class CustomNeurons():

    def __init__(self,**entries):
        self.__dict__.update(entries)

        if self.type == '3fractal':
            self.fractal_three()

    def fractal_three(self):
        H = 3 # depth
        n = [3,3] # fanning at each layer, (length = H-1), from soma to synapses

        dendrites = [ [] for _ in range(H-1) ]
        synapses = []

        count=0
        last_layer = 1
        # returns dendrites[layer][dendrite] = dendrites[H-1][n_h]
        for h in range(H-1): 
            for d in range(n[h]*last_layer):
                dendrites[h].append(common_dendrite(count, 'ri', self.beta_di, 
                                    self.tau_di, self.ib))

                if h == H-2:
                    print("connecting synapses")
                    synapses.append(common_synapse(count))
                    dendrites[h][d].add_input(synapses[d], 
                                              connection_strength = self.w_sd)
            last_layer = n[h]
        
        for i,layer in enumerate(dendrites):
            print("layer:", i)
            for j,d in enumerate(layer):
                print("  dendrite", j)
                if i < H-2:
                    for g in range()
                    d.add_input(dendrites[i+1][j*i+j], connection_strength=self.w_dd)
                    print(j*i+j)
        # for i,D in enumerate(dendrites):
        #     print("layer: ", i)
        #     for ii, d in enumerate(D):
        #         print("  dendrite", ii)

        fractal_neuron = common_neuron(1, 'ri', self.beta_ni, self.tau_ni, 
                                       self.ib, self.s_th_factor_n*self.s_max_n, 
                                       self.beta_ref, self.tau_ref, self.ib_ref)
        

neo = CustomNeurons(type='3fractal',**default_neuron_params)
print(neo.type)
print("connecting dendrites")# %%

