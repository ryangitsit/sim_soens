#%%
import numpy as np

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


from super_input import SuperInput

'''
Here a class for calling from a 'zoo' of possible neurons is implemented.

Plan:
 - Syntax for custom neuron calls based on dendritic structures and parameters.
 - Library of predefined neurons, both bio-inspired and engineering-specific
 - Should include a testing/plotting paradigm inherent in the class
 - Add more explicit connectivity defintions and corresponding plotting
'''


class NeuralZoo():

    def __init__(self,**entries):
        self.__dict__.update(entries)

        if self.type == '3fractal':
            self.fractal_three()

        if self.type == 'single':
            self.single()

        if self.type == 'custom':
            self.custom()
    
    def single(self):

        self.synapse = common_synapse(1)

        self.dendrite = common_dendrite(1, 'ri', self.beta_di, 
                                          self.tau_di, self.ib)
                                    
        self.dendrite.add_input(self.synapse, connection_strength = self.w_sd)

        self.neuron = common_neuron(1, 'ri', self.beta_ni, self.tau_ni, 
                                      self.ib, self.s_th_factor_n*self.s_max_n, 
                                      self.beta_ref, self.tau_ref, self.ib_ref)

        self.neuron.add_input(self.dendrite, connection_strength = self.w_dn)


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
            # print("layer:", i)
            for j,d in enumerate(layer):
                # print("  dendrite", j)
                if i < H-2:
                    for g in range(n[1]):
                        d.add_input(dendrites[i+1][j*n[1]+g], 
                                    connection_strength=self.w_dd)
                        # print(j,j*n[1]+g)
                    fractal_neuron.add_input(d, connection_strength=self.w_dn)
        self.dendrites = dendrites
        self.synapses = synapses
        self.fractal_neuron = fractal_neuron


    def custom(self):
        custom_neuron = common_neuron(1, 'ri', self.beta_ni, self.tau_ni, 
                                       self.ib, self.s_th_factor_n*self.s_max_n, 
                                       self.beta_ref, self.tau_ref, self.ib_ref)
        custom_neuron.name = 'custom_neuron'

        if hasattr(self, 'structure'):
            print("structure")
            arbor = structure
        elif hasattr(self, 'weights'):
            print("weights")
            arbor = weights

        dendrites = [ [] for _ in range(len(arbor)) ]
        synapses = []

        count=0
        count_syn=0
        last_layer = 1
        den_count = 0
        for i,layer in enumerate(arbor):
            c=0
            for j,dens in enumerate(layer):
                sub = []
                for k,d in enumerate(dens):
                    sub.append(common_dendrite(f"lay{i}_branch{j}_den{k}", 'ri', 
                                        self.beta_di,self.tau_di, self.ib))
                    den_count+=1
                    c+=1
                dendrites[i].append(sub)
        # for d in dendrites:
        #     print(d)
        print("\n")
        for i,l in enumerate(dendrites):
            for j, subgroup in enumerate(l):
                for k,d in enumerate(subgroup):
                    if i==0:
                        # print(i,j,k, " --> soma")
                        custom_neuron.add_input(d, connection_strength=weights[i][j][k])
                    else:
                        # print(i,j,k, " --> ", i-1,0,j)
                        # print(np.concatenate(dendrites[i-1])[j])
                        # d.add_input(np.concatenate(dendrites[i-1])[j], connection_strength=weights[i][j][k])
                        np.concatenate(dendrites[i-1])[j].add_input(d, connection_strength=weights[i][j][k])

        self.dendrites = dendrites
        for i,l in enumerate(dendrites):
            for j, subgroup in enumerate(l):
                for k,d in enumerate(subgroup):
                    keys = list(d.dendritic_inputs.keys())
                    print(i,j,k," - >", d.dendritic_connection_strengths)
                    # for k in keys:
                    #     print(i,j,k," - >", d.dendritic_inputs[k].connection_strengths)

                    
    def plot_structure(self):
        # add connection strengths
        # print(self.dendrites[0][0].dendritic_connection_strengths)
        import matplotlib.pyplot as plt
        layers = [[] for i in range(len(self.dendrites))]
        for i in range(len(layers)):
            for j in range(len(self.dendrites[i])):
                layers[i].append(list(self.dendrites[i][j].dendritic_inputs.keys()))
        colors = ['r','b','g',]
        Ns = [len(layers[i]) for i in range(len(layers))]
        Ns.reverse()
        Ns.append(1)
        for i,l in enumerate(layers):
            for j,d in enumerate(l):
                if len(d) > 0:
                    for k in layers[i][j]:
                        plt.plot([i+.5, i+1.5], [k-3,j+3], '-k', color=colors[j], linewidth=1)
        for i in range(Ns[-2]):
            plt.plot([len(layers)-.5, len(layers)+.5], [i+len(Ns),len(Ns)+1], '-k', color=colors[i], linewidth=1)
        for i,n in enumerate(Ns):
            if n == np.max(Ns):
                plt.plot(np.ones(n)*i+.5, np.arange(n), 'ok', ms=10)
            else:
                plt.plot(np.ones(n)*i+.5, np.arange(n)+(.5*np.max(Ns)-.5*n), 'ok', ms=10)
        plt.xticks([.5, 1.5,2.5], ['Layer 1', 'layer 2', 'soma'])
        plt.yticks([],[])
        plt.xlim(0,len(layers)+1)
        plt.ylim(-1, max(Ns))
        plt.title('Dendritic Arbor')
        plt.show()



structure = [
             [2],
             [3,2],
             [3,2,0,2,2]
            ]

weights = [
           [[.2,.5]],
           [[.2,.5,.4],[.2,.2]],
           [[.1,.1,.1],[.7,.7],[0],[.5,.6],[.3,.2]]
          ]

for w in weights:
    print(w)
arb = NeuralZoo(type="custom",weights=weights,**default_neuron_params) 
arb.plot_structure()



# times = np.arange(0,500,50)
# indices = np.zeros(len(times)).astype(int)
# def_spikes = [indices,times]

# # input_ = SuperInput(channels=1, type='random', total_spikes=int(500/42), duration=500)
# input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes, duration=500)

# # print(input.spike_rows)
# # raster_plot(input.spike_arrays)


# default_neuron_params['w_dn'] = 0.42
# default_neuron_params['tau_di'] = 1000
# default_neuron_params['tau_ref'] = 50
# default_neuron_params["s_th_factor_n"] = 0.1

# neo = NeuralZoo(type='single',**default_neuron_params)

# neo.synapse.add_input(input.signals[0])

# net = network(name = 'network_under_test')
# net.add_neuron(neo.neuron)
# # net.neurons['name'].name = 1
# net.run_sim(dt = .1, tf = 500)
# tau_convert = 1/net.neurons[1].time_params['t_tau_conversion']
# net.get_recordings()
# spikes = [net.spikes[0],net.spikes[1]*1000]
# # print(spikes)

# raster_plot(spikes,duration=500)
# spd = neo.dendrite.synaptic_inputs[1].phi_spd
# dend_s = neo.dendrite.s
# signal = net.neurons[1].dend__nr_ni.s
# ref = net.neurons[1].dend__ref.s

# import matplotlib.pyplot as plt
# plt.figure(figsize=(12,4))
# plt.plot(spd[::10], label='phi_spd')
# plt.plot(dend_s[::10], label='dendtrite signal')
# plt.plot(signal[::10], label='soma signal')
# plt.plot(ref[::10], label='refractory signal')
# spike_height = [signal[::10][int(net.spikes[1][x]*1000)] for x in range(len(net.spikes[1]))]
# plt.plot(net.spikes[1]*1000,spike_height,'xk', label='neuron fires')
# plt.legend()



# default_neuron_params['w_dd'] = 1
# default_neuron_params['w_dn'] = 1
# default_neuron_params['tau_di'] = 100


# neo = NeuralZoo(type='3fractal',**default_neuron_params)

# neo.plot_structure()

# print(neo.fractal_neuron.__dict__)
# print

# for k,v in neo.fractal_neuron.__dict__.items():
#     print(k,v)



#%%
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



# # %%

# %%
