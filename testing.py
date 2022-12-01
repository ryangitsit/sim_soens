'''
THIS FILE IS FOR LOCAL DEVELOPMENT PURPOSES ONLY
'''


################################################################################
################################################################################
################################################################################
                        ###  CUSTOM NEURON TESTING  ###
                        ###  CUSTOM NEURON TESTING  ###
                        ###  CUSTOM NEURON TESTING  ###
################################################################################
################################################################################
################################################################################

from neural_zoo import NeuralZoo
from params import weights_3, default_neuron_params, nine_pixel_params
from super_input import SuperInput
from soen_sim import input_signal, synapse, neuron, network
from _plotting__soen import raster_plot
import numpy as np
import matplotlib.pyplot as plt

weights_3 = weights = [
                [[1,1,1]],
                [[1,1,1],[1,1,1],[1,1,1]],
            ]
default_neuron_params['tau_ref'] = 35
arb = NeuralZoo(type="custom",weights=weights,**default_neuron_params) 

print(arb.__dict__.keys())
# print(arb.synapses[0][0].spd_duration)
# input = SuperInput(channels=9, type='random', total_spikes=1000, duration=100)
# # raster_plot(input.spike_arrays)
# for i,g in enumerate(arb.synapses):
#     for s in g:
#         s.add_input(input_signal(name = 'input_synaptic_drive', 
#                                  input_temporal_form = 'arbitrary_spike_train', 
#                                  spike_times = input.spike_rows[i]))
# # input_signal(name = 'input_synaptic_drive', input_temporal_form = 'arbitrary_spike_train', spike_times = input.spike_rows[i])


# net = network(name = 'network_under_test')
# net.add_neuron(arb.neuron)
# # print(net.neurons.keys())
# # print(net.neurons['custom_neuron'].dend__ref.synaptic_inputs.keys())
# if 'custom_neuron__syn_refraction' not in net.neurons['custom_neuron'].dend__ref.synaptic_inputs.keys():
#     net.neurons[list(net.neurons.keys())[0]].name = 1
# # net.neurons[list(net.neurons.keys())[0]].name = 1
# net.run_sim(dt = .1, tf = 100)
# net.get_recordings()
# spikes = [net.spikes[0],net.spikes[1]*1000]
# # arb.arbor_activity_plot()
# print(spikes[1])
# raster_plot(spikes)
# signal = net.neurons['custom_neuron'].dend__nr_ni.s
# ref = net.neurons['custom_neuron'].dend__ref.s
# print(arb.dendrites)
# plt.figure(figsize=(12,6))
# for i,layer in enumerate(arb.dendrites):
#     for j,dens in enumerate(layer):
#         for k,d in enumerate(dens):
#             # spd = arb.dendrites[i][j][k].synaptic_inputs[i].phi_spd
#             dend_s = arb.dendrites[i][j][k].s
#             # plt.plot(spd,'--') #, label='phi_spd_'+str(i))
#             plt.plot(dend_s[::10],'--', linewidth=1) #, label='dendtrite signal '+str(i))
# plt.plot(signal[::10], linewidth=2, color='darkcyan', label='soma signal')
# plt.plot(ref[::10], linewidth=2, color='crimson', label='refractory signal')
# spike_height = [signal[::10][int(net.spikes[1][x]*1000)] for x in range(len(net.spikes[1]))]
# plt.plot(net.spikes[1]*1000,spike_height,'xk', label='neuron fires')
# plt.plot(input.spike_arrays[1],np.zeros(len(input.spike_arrays[1])),'xr', label='neuron fires')
# plt.legend()
# plt.show()




############ single neuron
# default_neuron_params['tau_ref'] = 1
# default_neuron_params["s_th_factor_n"] = 0.1
# default_neuron_params['w_dn'] = 1
# default_neuron_params['w_sd'] = 1
# default_neuron_params['w_dd'] = 1

# neo = NeuralZoo(type='single',**default_neuron_params)
# times = np.arange(0,500,1)
# indices = np.zeros(len(times)).astype(int)
# def_spikes = [indices,times]
# input_ = SuperInput(channels=1, type='defined', defined_spikes=def_spikes, duration=500)
# neo.synapse.add_input(input_.signals[0])
# net = network(name = 'network_under_test')
# net.add_neuron(neo.neuron)
# net.run_sim(dt = .1, tf = 500)
# net.get_recordings()
# spd = neo.dendrite.synaptic_inputs[1].phi_spd
# dend_s = neo.dendrite.s

# signal = net.neurons[1].dend__nr_ni.s
# ref = net.neurons[1].dend__ref.s
# spikes = [net.spikes[0],net.spikes[1]*1000]
# # arb.arbor_activity_plot()
# print(spikes[1])
# plt.figure(figsize=(12,4))
# plt.plot(spd[::10], label='phi_spd')
# plt.plot(dend_s[::10], label='dendtrite signal')
# plt.plot(signal[::10], label='soma signal')
# plt.plot(ref[::10], label='refractory signal')
# spike_height = [signal[::10][int(net.spikes[1][x]*1000)] for x in range(len(net.spikes[1]))]
# plt.plot(net.spikes[1]*1000,spike_height,'xk', label='neuron fires')
# # plt.plot(input_.spike_arrays[1],np.zeros(len(input_.spike_arrays[1])),'xr', label='input event')
# plt.legend()
# plt.show()


# neuron = NeuralZoo(type="custom",**nine_pixel_params) 
# neuron.plot_custom_structure()


# arb.neuron.name = 1
# # indices = np.array([0,1,4,7,8]) # z-pixel array
# indices = np.array([1,4,3,6,8])-1 # v
# # indices = np.array([2,4,6,7,9])-1 # n
# times = np.ones(len(indices))*20
# def_spikes = [indices,times]
# input = SuperInput(channels=9, type='defined', defined_spikes=def_spikes, duration=100)
# input_signal(name = 'input_synaptic_drive', input_temporal_form = 'arbitrary_spike_train', spike_times = input.spike_rows[i])

# count =0
# for g in arb.synapses:
#     for s in g:
#         for i,row in enumerate(input.spike_rows):
#             if i == int(s.name)-1:
#                 s.add_input(input_signal(name = 'input_synaptic_drive', 
#                 input_temporal_form = 'arbitrary_spike_train', spike_times = input.spike_rows[i]))
#                 count+=1
# print(count)

# net = network(name = 'network_under_test')
# net.add_neuron(arb.neuron)
# # print(net.neurons.keys())
# # print(net.neurons['custom_neuron'].dend__ref.synaptic_inputs.keys())
# if 'custom_neuron__syn_refraction' not in net.neurons['custom_neuron'].dend__ref.synaptic_inputs.keys():
#     net.neurons[list(net.neurons.keys())[0]].name = 1
# net.run_sim(dt = .1, tf = 100)
# net.get_recordings()

# spikes = [net.spikes[0],net.spikes[1]*1000]
# # arb.arbor_activity_plot()
# print(spikes[1])
# # raster_plot(spikes)

# #%%
# # print(arb.neuron.dend__nr_ni.dendritic_connection_strengths)
# import matplotlib.pyplot as plt
# plt.plot(arb.neuron.dend__nr_ni.s)






################################################################################
                        ###  CUSTOM NETWORK TESTING  ###
################################################################################



# input_single = SuperInput(channels=50, type='random', total_spikes=100, duration=100)
# input_MNIST = SuperInput(type='MNIST', index=0, slow_down=100, duration=1000)
# single_net= SuperNet(N=50,duration=100,**net_args)#,params=net_args) #dendrites,synapses

# single_net.connect_input(input_MNIST)

# single_net.run()
# single_net.record(['spikes'])

# #%%

# spikes = single_net.spikes
# spikes = [spikes[0],spikes[1]]
# from _plotting__soen import raster_plot
# raster_plot(spikes,duration=100,input=input_single.spike_arrays)

#%%


###########################################################################
###########################################################################
###########################################################################

# make single line
# super_input = SuperInput(**input_args)
# mnist_data, mnist_indices, mnist_spikes = super_input.MNIST()
# spikes = [mnist_indices[0],mnist_spikes[0]]
# input = super_input.array_to_rows(spikes)

# # make single line
# super_net = SuperNet(dend_load_arrays_thresholds_saturations('default_ri'),**net_args)
# super_net.param_setup()
# super_net.make_input_signal(input)
# super_net.make_neurons()
# super_net.make_net()

# # Add preemptory monitor statement
# spiked, S = super_net.run()
# print("spikes = ", len(spiked[0]),"\n\n")

# # Call this from a separate plotting file/class
# # super_net.raster_plot(spiked)
# input_spikes = super_input.rows_to_array(input)
# super_net.raster_input_plot(spiked,input_spikes)

# labels = ["zero","zero","zero","one","one","one","two","two","two"]
# for i,pattern in enumerate(labels):

#     print("pattern index: ",i)
#     replica = str(i%3)

#     spikes = [mnist_indices[i],mnist_spikes[i]]
#     input = super_input.array_to_rows(spikes)

#     super_net = SuperNet(dend_load_arrays_thresholds_saturations('default_ri'),**net_args)
#     super_net.param_setup()
#     super_net.make_input_signal(input)

#     super_net.make_neurons()
#     super_net.make_net()

#     spiked, S = super_net.run()
#     print("spikes = ", len(spiked[0]),"\n\n")

#     # super_net.raster_plot(spiked)
#     # input_spikes = super_input.rows_to_array(input)
#     # super_net.raster_input_plot(spiked,input_spikes)
#     dir = 'test_cases'
#     super_net.spks_to_txt(spiked,3,dir,f"spikes_{pattern}_{replica}")

#%%


### Previous sweep
# arg-parser?

# # input = super_input.gen_rand_input(10,25)
# labels = ["zero","zero","zero","one","one","one","two","two","two"]
# # labels = ["zero"] 
# count = 0
# for in_p in np.arange(0.2,1,.2):
#     print(in_p)
#     for res_p in np.arange(0.2,1,.2):
#         for sd in [-1,0]:
#             for dn in [-1,0]:
#                 for tau_spread in np.arange(50,350,50):
#                     net_args["input_p"] = in_p
#                     net_args["reservoir_p"] = res_p
#                     net_args["w_sd"][0] = sd
#                     net_args["w_dn"][0] = dn
#                     net_args["tau_di"] = [1000-int(tau_spread/2),1000+int(tau_spread/2)]

#                     dir = f"spks_in{in_p[:3]}_res{res_p[:3]}_sd{sd}_dn{dn}_taudis{tau_spread}"
#                     print(dir)
#                     for i,pattern in enumerate(labels):

#                         print("pattern index: ",i)
#                         replica = str(i%3)

#                         spikes = [mnist_indices[i],mnist_spikes[i]]
#                         input = super_input.array_to_rows(spikes)

#                         super_net = SuperNet(dend_load_arrays_thresholds_saturations('default_ri'),**net_args)
#                         super_net.param_setup()
#                         super_net.make_input_signal(input)

#                         super_net.make_neurons()
#                         super_net.make_net()

#                         spiked, S = super_net.run()
#                         print("spikes = ", len(spiked[0]),"\n\n")

#                         # super_net.raster_plot(spiked)
#                         # input_spikes = super_input.rows_to_array(input)
#                         # super_net.raster_input_plot(spiked,input_spikes)

#                         super_net.spks_to_txt(spiked,3,dir,f"spikes_{pattern}_{replica}")