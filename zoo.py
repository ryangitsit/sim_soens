#%%
import numpy as np

from super_input import SuperInput

from _util import (
    physical_constants, set_plot_params, index_finder)
from _util__soen import (
    dend_load_rate_array, dend_load_arrays_thresholds_saturations)
from soen_sim import input_signal, synapse, neuron, network
from soen_sim_lib__common_components__simple_gates import (
    common_dendrite, common_synapse, common_neuron)

from super_input import SuperInput
from params import default_neuron_params
from _plotting__soen import raster_plot

'''
Here a class for calling from a 'zoo' of possible neurons is implemented.

Plan:
 - Syntax for custom neuron calls based on dendritic structures and parameters.
 - Library of predefined neurons, both bio-inspired and engineering-specific
 - Should include a testing/plotting paradigm inherent in the class
 - Add more explicit connectivity defintions and corresponding plotting
'''

#%%
# input = SuperInput(channels=9, type='random', total_spikes=1000, duration=100)

# print(input.spike_rows)
# raster_plot(input.spike_arrays)

# print(default_neuron_params)

#%%
class CustomNeurons():

    def __init__(self,**entries):
        self.__dict__.update(entries)

        if self.type == '3fractal':
            self.fractal_three()


    def fractal_three(self):
        H = 3 # depth
        n = [3,3] # fanning at each layer, (length = H-1), from soma to synapses

        fractal_neuron = common_neuron(1, 'ri', self.beta_ni, self.tau_ni, 
                                       self.ib, self.s_th_factor_n*self.s_max_n, 
                                       self.beta_ref, self.tau_ref, self.ib_ref)
        fractal_neuron.name = 'name'
        dendrites = [ [] for _ in range(H-1) ]
        synapses = []

        count=0
        count_syn=0
        last_layer = 1
        # returns dendrites[layer][dendrite] = dendrites[H-1][n_h]
        for h in range(H-1): 
            for d in range(n[h]*last_layer):
                dendrites[h].append(common_dendrite(count, 'ri', self.beta_di, 
                                    self.tau_di, self.ib))

                if h == H-2:
                    synapses.append(common_synapse(d))
                    dendrites[h][d].add_input(synapses[d], 
                                              connection_strength = self.w_sd)
                count+=1
            last_layer = n[h]

        for i,layer in enumerate(dendrites):
            print("layer:", i)
            for j,d in enumerate(layer):
                print("  dendrite", j)
                if i < H-2:
                    for g in range(n[1]):
                        d.add_input(dendrites[i+1][j*n[1]+g], 
                                    connection_strength=self.w_dd)
                        # print(j,j*n[1]+g)
                    fractal_neuron.add_input(d, connection_strength=self.w_dn)
        self.dendrites = dendrites
        self.synapses = synapses
        self.fractal_neuron = fractal_neuron

    def plot_structure(self):
        import matplotlib.pyplot as plt
        Ns = len(self.dendrites[1])
        Nt = len(self.dendrites[0])
        plt.figure(figsize=(16, 10))
        #plt.subplot(121)
        plt.plot(np.zeros(Ns)+.5, np.arange(Ns), 'ok', ms=10)
        plt.plot(np.ones(Nt)+1.5, np.arange(Nt)+(.5*Ns-.5*Nt), 'ok', ms=10)
        # for i, j in zip(S.i, S.j):
        #     plt.plot([0, 1], [i, j*(Ns/Nt)+.5*(Ns/Nt)], '-k', linewidth=200*S.w[i,j])
        plt.xticks([.5, 2.5], ['Source', 'Target'])
        plt.ylabel('Neuron index')
        plt.xlim(-.1,3)
        plt.ylim(-1, max(Ns, Nt))
        # plt.subplot(122)
        # plt.plot(S.i, S.j, 'ok',ms=2)
        # plt.xlim(-1, Ns)
        # plt.ylim(-1, Nt)
        # plt.xlabel('Source neuron index')
        # plt.ylabel('Target neuron index')
        plt.show()
        
        

# default_neuron_params['w_dd'] = 1
# default_neuron_params['w_dn'] = 1
# default_neuron_params['tau_di'] = 100


# neo = CustomNeurons(type='3fractal',**default_neuron_params)

# # neo.plot_structure()

# # print(neo.fractal_neuron.__dict__)
# # print

# # for k,v in neo.fractal_neuron.__dict__.items():
# #     print(k,v)




# #%%
# for i in range(len(neo.synapses)):
#     in_ = input_signal(name = 'input_synaptic_drive', 
#                        input_temporal_form = 'arbitrary_spike_train', 
#                        spike_times = input.spike_rows[i])
#     neo.synapses[i].add_input(in_)

# net = network(name = 'network_under_test')
# net.add_neuron(neo.fractal_neuron)
# net.neurons['name'].name = 1
# print(net.neurons['name'].name)
# # network_object.neurons[neuron_key].dend__ref.synaptic_inputs['{}__syn_refraction'.format(network_object.neurons[neuron_key].name)].spike_times_converted = np.append(network_object.neurons[neuron_key].dend__ref.synaptic_inputs['{}__syn_refraction'.format(network_object.neurons[neuron_key].name)].spike_times_converted,tau_vec[ii+1])
# net.run_sim(dt = 10, tf = 1000)
# spikes = [ [] for _ in range(2) ]
# S = []
# Phi_r = []
# count = 0
# for neuron_key in net.neurons:
#     s = net.neurons[neuron_key].dend__nr_ni.s
#     S.append(s)
#     phi_r = net.neurons[neuron_key].dend__nr_ni.phi_r
#     Phi_r.append(phi_r)
#     spike_t = net.neurons[neuron_key].spike_times
#     spikes[0].append(np.ones(len(spike_t))*count)
#     spikes[1].append(spike_t)
#     count+=1
# spikes[0] =np.concatenate(spikes[0])
# spikes[1] = np.concatenate(spikes[1])/1000


# raster_plot(spikes,duration=1000)
# # %%
