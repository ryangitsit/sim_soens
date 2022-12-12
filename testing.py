'''
THIS FILE IS FOR LOCAL DEVELOPMENT PURPOSES ONLY

TODO
 - DENDRITE IN NEURON, ENTRIES METHOD
 - Add multi-synapse connections
 - Plotting for phi_nr
 - Flag for rollover
 - Dendrite after skip/intermediate
 - Experiment with auto refraction
 - Try self-connect
 - Fix defined_spikes
 - Finish recursive search and plotting
 - Finish activity plotting
 - Make callable plotting functions
 - Integrate jeff's new changes
 - Neural motifs
 - Networking
 - Preliminary sweep
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


from params import default_neuron_params
from super_input import SuperInput
from soen_sim import input_signal, synapse, neuron, network
# from soen_plotting import raster_plot
import numpy as np
import matplotlib.pyplot as plt

times = np.arange(0,150,25)
indices = np.zeros(len(times)).astype(int)
def_spikes = [indices,times]
input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes, duration=150)


default_neuron_params['s_th'] = 0.75
# default_neuron_params["normalize_input_connection_strengths"] = False
# default_neuron_params['tau_di'] = 10 



# mono_point = NeuralZoo(type='mono_point',**default_neuron_params)
# mono_point.synapses[0][0][0][0].add_input(input.signals[0])
# net = network(sim=True,dt=.01,tf=150,nodes=[mono_point])
# mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron")

# mono_dend = NeuralZoo(type='mono_dendrite',**default_neuron_params)
# mono_dend.synapses[0][1][0][0].add_input(input.signals[0])
# net = network(sim=True,dt=.01,tf=150,nodes=[mono_dend])
# title = "Monosynaptic Neuron with Intermediate Dendrite"
# mono_dend.plot_neuron_activity(net,title=title)

# mono_dend_soma = NeuralZoo(type='mono_dend_soma',**default_neuron_params)
# mono_dend_soma.synapses[0][1][0][0].add_input(input.signals[0])
# net = network(sim=True,dt=.01,tf=150,nodes=[mono_dend_soma])
# title = "Monosynaptic Neuron; Synapse Feeds Intermediate Dendrite and Soma"
# mono_dend_soma.plot_neuron_activity(net,title=title)


# self_feed = NeuralZoo(type='self_feed',**default_neuron_params)
# self_feed.synapses[0][1][0][0].add_input(input.signals[0])
# net = network(sim=True,dt=.01,tf=150,nodes=[self_feed])
# title = "Monosynaptic Neuron with Intermediate Self-Feeding Dendrite"
# self_feed.plot_neuron_activity(net,title=title)

# mono_plus_minus = NeuralZoo(type='mono_plus_minus',**default_neuron_params)
# mono_plus_minus.synapses[0][1][0][0].add_input(input.signals[0])
# mono_plus_minus.synapses[0][1][0][1].add_input(input.signals[0])
# mono_plus_minus.synapses[0][0][0][0].add_input(input.signals[0])
# net = network(sim=True,dt=.01,tf=150,nodes=[mono_plus_minus])
# title = "Monosynaptic Neuron; Synapse Feeds Exctiatory and Inhibitory Dendrites, and Soma"
# mono_plus_minus.plot_neuron_activity(net,title=title,input=input)
# # print(mono_plus_minus.neuron.dend__nr_ni.synaptic_connection_strengths)
# # print(mono_plus_minus.neuron.dend__nr_ni.dendritic_connection_strengths)

# default_neuron_params['s_th'] = 0.3
# double_ref = NeuralZoo(type='double_ref',**default_neuron_params)
# double_ref.synapses[0][1][0][0].add_input(input.signals[0])
# print(double_ref.neuron.dend__nr_ni.dendritic_connection_strengths)
# net = network(sim=True,dt=.01,tf=150,nodes=[double_ref])
# title = "double_ref"
# double_ref.plot_neuron_activity(net,title=title)

# point_3ex_1in = NeuralZoo(type='point_3ex_1in',**default_neuron_params)
# point_3ex_1in.synapses[0][0][0][0].add_input(input.signals[0])
# point_3ex_1in.synapses[1][0][0][0].add_input(input.signals[0])
# point_3ex_1in.synapses[2][0][0][0].add_input(input.signals[0])
# point_3ex_1in.synapses[3][0][0][0].add_input(input.signals[0])
# net = network(sim=True,dt=.01,tf=150,nodes=[point_3ex_1in])
# title = "3 Excitatory and 1 Inhibitory Synapse to Soma"
# point_3ex_1in.plot_neuron_activity(net,title=title,input=input)

# # print(point_3ex_1in.neuron.dend__nr_ni.synaptic_connection_strengths)
# # print(point_3ex_1in.neuron.dend__nr_ni.dendritic_connection_strengths)

# asym_plus_minus = NeuralZoo(type='asym_plus_minus',**default_neuron_params)
# asym_plus_minus.synapses[0][1][0][0].add_input(input.signals[0])
# asym_plus_minus.synapses[1][1][0][0].add_input(input.signals[0])
# asym_plus_minus.synapses[2][1][0][0].add_input(input.signals[0])
# asym_plus_minus.synapses[3][1][0][0].add_input(input.signals[0])
# asym_plus_minus.synapses[4][1][0][1].add_input(input.signals[0])
# net = network(sim=True,dt=.01,tf=150,nodes=[asym_plus_minus])
# title = "3 Excitatory and 1 Inhibitory Dendrite; Each with +3,-1 Synapses"
# asym_plus_minus.plot_neuron_activity(net,title=title,input=input)

# print(asym_plus_minus.dendrites[1][0][0].synaptic_connection_strengths)
# print(asym_plus_minus.dendrites[1][0][1].synaptic_connection_strengths)
# # print(asym_plus_minus.neuron.dend__nr_ni.dendritic_connection_strengths)


# denex3_denin1 = NeuralZoo(type='denex3_denin1',**default_neuron_params)

# denex3_denin1.synapses[0][1][0][0].add_input(input.signals[0])
# denex3_denin1.synapses[1][1][0][0].add_input(input.signals[0])
# denex3_denin1.synapses[2][1][0][0].add_input(input.signals[0])
# denex3_denin1.synapses[3][1][0][0].add_input(input.signals[0])

# denex3_denin1.synapses[4][1][0][1].add_input(input.signals[0])
# denex3_denin1.synapses[5][1][0][1].add_input(input.signals[0])
# denex3_denin1.synapses[6][1][0][1].add_input(input.signals[0])
# denex3_denin1.synapses[7][1][0][1].add_input(input.signals[0])

# denex3_denin1.synapses[8][1][0][2].add_input(input.signals[0])
# denex3_denin1.synapses[9][1][0][2].add_input(input.signals[0])
# denex3_denin1.synapses[10][1][0][2].add_input(input.signals[0])
# denex3_denin1.synapses[11][1][0][2].add_input(input.signals[0])

# denex3_denin1.synapses[12][1][0][3].add_input(input.signals[0])

# net = network(sim=True,dt=.01,tf=150,nodes=[denex3_denin1])
# title = "Monosynaptic Point Neuron with Intermediate Dendrite"
# denex3_denin1.plot_neuron_activity(net,title=title,input=input)


# proximal_basal = NeuralZoo(type='proximal_basal',**default_neuron_params)

# proximal_basal.synapses[0][1][0][0].add_input(input.signals[0])
# proximal_basal.synapses[1][1][0][0].add_input(input.signals[0])
# proximal_basal.synapses[2][1][0][0].add_input(input.signals[0])
# proximal_basal.synapses[3][1][0][0].add_input(input.signals[0])

# proximal_basal.synapses[4][1][0][1].add_input(input.signals[0])
# proximal_basal.synapses[5][1][0][1].add_input(input.signals[0])
# proximal_basal.synapses[6][1][0][1].add_input(input.signals[0])
# proximal_basal.synapses[7][1][0][1].add_input(input.signals[0])

# proximal_basal.synapses[8][1][0][2].add_input(input.signals[0])

# net = network(sim=True,dt=.01,tf=150,nodes=[proximal_basal])
# title = "Monosynaptic Point Neuron with Intermediate Dendrite"
# proximal_basal.plot_neuron_activity(net,title=title,input=input)



'''
todo:
factor dens by weighting?
'''



# synaptic_structure = [[[0]],[[.4,.5,.6]]]
# weights = [[[1,1,1]]]
# mono_dend = NeuralZoo(type="custom",weights=weights, synaptic_structure=synaptic_structure,**default_neuron_params)
# mono_dend.plot_custom_structure()
# mono_dend.synapses[1][0][0].add_input(input.signals[0])
# mono_dend.synapses[1][0][1].add_input(input.signals[0])
# mono_dend.synapses[1][0][2].add_input(input.signals[0])

# net = network(name = 'network_under_test')
# net.add_neuron(mono_dend.neuron)
# net.run_sim(dt = .01, tf = 150)
# net.get_recordings()


# print(mono_dend.dendrites[0][0][0].dendritic_connection_strengths)




# plt.figure(figsize=(12,4))
# plt.plot(net.t,spd, label='phi_spd')
# plt.plot(net.t,signal,  label='soma signal', linewidth=4)
# for d in mono_dend.dendrites[1][0]:
#     plt.plot(net.t,d.s, label='dendrite signal')
# plt.plot(net.t,ref, ':',color = 'r', label='refractory signal')
# plt.axhline(y = mono_dend.s_th, color = 'purple', linestyle = '--',label='Threshold')
# plt.legend()
# plt.show()

# default_neuron_params['beta_ni'] = 2*np.pi*1e2
# default_neuron_params['ib_n'] = default_ib
# default_neuron_params['s_th'] = 0.1
# synaptic_structure = [[[0]],[[1]]]
# weights = [[[.5]]]
# mono_dend = NeuralZoo(type="custom",weights=weights, synaptic_structure=synaptic_structure,**default_neuron_params)

# mono_dend.synapses[1][0][0].add_input(input.signals[0])
# net = network(name = 'network_under_test')
# net.add_neuron(mono_dend.neuron)
# net.run_sim(dt = .01, tf = 150)
# net.get_recordings()

# spd = mono_dend.synapses[1][0][0].phi_spd
# signal = mono_dend.dendrites[0][0][0].s
# dend_s = mono_dend.dendrites[1][0][0].s
# ref = mono_dend.neuron.dend__ref.s
# print(mono_dend.s_th)
# print(mono_dend.dendrites[0][0][0].integrated_current_threshold)
# plt.figure(figsize=(12,4))
# plt.plot(net.t,spd, label='phi_spd')
# plt.plot(net.t,signal, label='soma signal')
# plt.plot(net.t,ref, label='refractory signal')
# plt.axhline(y = mono_dend.s_th, color = 'purple', linestyle = '--',label='Threshold')
# plt.legend()
# plt.show()

# import time
# startTime = time.time()
# synaptic_structure = [[[0]],[[1]]]
# weights = [[[.5]]]
# for i in range(1000):
#     mono_dend = NeuralZoo(type="custom",weights=weights, synaptic_structure=synaptic_structure,**default_neuron_params)
# executionTime = (time.time() - startTime)
# print('Execution time in seconds: ' + str(executionTime))

# default_neuron_params['beta_ni'] = 2*np.pi*1e2
# default_neuron_params['ib_n'] = default_ib
# default_neuron_params['s_th'] = 0.75
# synaptic_structure = [[[0]],[[1]]]

# W = np.arange(0,1.6,.2)

# plt.figure(figsize=(24,8))
# for w in W:

#     weights = [[[w]]]
#     mono_dend = NeuralZoo(type="custom",weights=weights, synaptic_structure=synaptic_structure,**default_neuron_params)

#     mono_dend.synapses[1][0][0].add_input(input.signals[0])
#     # mono_dend.synapses[0][0][0].add_input(input.signals[0])


#     net = network(name = 'network_under_test')
#     net.add_neuron(mono_dend.neuron)
#     net.run_sim(dt = .01, tf = 150)
#     net.get_recordings()

#     spd = mono_dend.synapses[1][0][0].phi_spd
#     signal = mono_dend.dendrites[0][0][0].s
#     dend_s = mono_dend.dendrites[1][0][0].s
#     ref = mono_dend.neuron.dend__ref.s

#     # plt.plot(net.t,spd, label='phi_spd')
#     plt.plot(net.t,signal, label=f'soma signal, w = {np.round(w,1)}')
#     # plt.plot(net.t,dend_s, label='dendrite signal')
#     # plt.plot(net.t,ref, label='refractory signal')
# plt.axhline(y = mono_dend.s_th, color = 'purple', linestyle = '--',label='Threshold')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()


## control
# from soen_component_library import common_neuron, common_dendrite, common_synapse
# default_neuron_params['beta_ni'] = 2*np.pi*1e2
# default_neuron_params['ib_n'] = default_ib
# default_neuron_params['s_th'] = 0.15
# synapse = common_synapse(1)
# synapse.add_input(input.signals[0])

# ib_ref = default_neuron_params['ib_ref']
# tau_ref = default_neuron_params['tau_ref']
# beta_ref = default_neuron_params['beta_ref']
# s_th= 0.15# = default_neuron_params['s_th']
# ib_n = default_neuron_params['ib_n']
# tau_ni = default_neuron_params['tau_ni']
# beta_ni = default_neuron_params['beta_ni']
# ib = default_neuron_params['ib']
# tau_di = default_neuron_params['tau_di']
# beta_di = 2*np.pi*1e2 # default_neuron_params['beta_di']

# dendrite = common_dendrite(1, 'ri', beta_di, 
#                                     tau_di, ib)
                            
# dendrite.add_input(synapse, connection_strength = 1)

# neuron = common_neuron(1, 'ri', beta_ni, tau_ni, 
#                                 ib_n, s_th, 
#                                 beta_ref, tau_ref, ib_ref)

# neuron.add_input(dendrite, connection_strength = 1)
# neuron.add_input(synapse, connection_strength = 1)
# print(neuron.dend__nr_ni.dendritic_connection_strengths)
# net = network(name = 'network_under_test')
# net.add_neuron(neuron)
# net.run_sim(dt = .01, tf = 150)
# net.get_recordings()

# spd = synapse.phi_spd
# signal = neuron.dend__nr_ni.s
# ref = neuron.dend__ref.s
# dend = dendrite.s
# plt.figure(figsize=(12,4))

# plt.plot(net.t,spd, label='phi_spd')
# plt.plot(net.t,signal, label='soma signal')
# plt.plot(net.t,ref, label='refractory signal')
# plt.plot(net.t,dend, label='dendrite signal')
# plt.axhline(y = neuron.s_th, color = 'purple', linestyle = '--',label='Threshold')
# plt.legend()


# synaptic_structure = [[[1]]]

# mono = NeuralZoo(type="custom",synaptic_structure=synaptic_structure,**default_neuron_params)
# mono.synapses[0][0][0].add_input(input.signals[0])

# net = network(name = 'network_under_test')
# net.add_neuron(mono.neuron)
# net.run_sim(dt = .1, tf = 150)
# net.get_recordings()

# spd = mono.synapses[0][0][0].phi_spd
# signal = mono.dendrites[0][0][0].s
# ref = mono.neuron.dend__ref.s

# plt.figure(figsize=(12,4))

# plt.plot(net.t,spd, label='phi_spd')
# plt.plot(net.t,signal, label='soma signal')
# plt.plot(net.t,ref, label='refractory signal')

# plt.axhline(y = mono.s_th, color = 'purple', linestyle = '--',label='Threshold')
# plt.legend()
# plt.show()

# default_neuron_params['beta_ni'] = 2*np.pi*1e2
# default_neuron_params['ib_n'] = default_ib
# default_neuron_params['s_th'] = 0.75
# synaptic_structure = [[[0]],[[1]]]
# weights = [[[1]]]
# mono_dend = NeuralZoo(type="custom",weights=weights, synaptic_structure=synaptic_structure,**default_neuron_params)

# mono_dend.synapses[1][0][0].add_input(input.signals[0])
# net = network(name = 'network_under_test')
# net.add_neuron(mono_dend.neuron)
# net.run_sim(dt = .01, tf = 150)
# net.get_recordings()

# spd = mono_dend.synapses[1][0][0].phi_spd
# signal = mono_dend.dendrites[0][0][0].s
# dend_s = mono_dend.dendrites[1][0][0].s
# ref = mono_dend.neuron.dend__ref.s

# plt.figure(figsize=(12,4))
# plt.plot(net.t,spd, label='phi_spd')
# plt.plot(net.t,signal, label='soma signal')
# plt.plot(net.t,dend_s, label='dendrite signal')
# plt.plot(net.t,ref, label='refractory signal')
# plt.axhline(y = mono_dend.s_th, color = 'purple', linestyle = '--',label='Threshold')
# plt.legend()
# plt.show()

# default_neuron_params['beta_ni'] = 2*np.pi*1e2
# default_neuron_params['s_th'] = 0.75
# synaptic_structure = [[[0]],[[1]]]
# weights = [[[1]]]
# mono_dend = NeuralZoo(type="custom",weights=weights, synaptic_structure=synaptic_structure,**default_neuron_params)

# mono_dend.synapses[1][0][0].add_input(input.signals[0])
# net = network(name = 'network_under_test')
# net.add_neuron(mono_dend.neuron)
# net.run_sim(dt = .01, tf = 150)
# net.get_recordings()

# print(mono_dend.neuron.dend__nr_ni.dendritic_inputs)
# print(mono_dend.neuron.dend__nr_ni.dendritic_connection_strengths)
# print(mono_dend.neuron.dend__nr_ni.synaptic_inputs)
# spd = mono_dend.synapses[1][0][0].phi_spd
# signal = mono_dend.dendrites[0][0][0].s
# dend_s = mono_dend.dendrites[1][0][0].s
# ref = mono_dend.neuron.dend__ref.s

# plt.figure(figsize=(24,8))
# plt.plot(net.t,spd, label='phi_spd')
# plt.plot(net.t,signal, label='soma signal')
# plt.plot(net.t,dend_s, label='dendrite signal')
# plt.plot(net.t,ref, label='refractory signal')
# plt.axhline(y = mono_dend.s_th, color = 'purple', linestyle = '--',label='threshold')
# plt.legend()
# plt.show()





































# # custom parameters
# default_neuron_params['w_dn'] = 0.9
# default_neuron_params['w_dd'] = 0.9
# default_neuron_params['tau_di'] = 10
# default_neuron_params['tau_ref'] = 35
# default_neuron_params["s_th"] = 0.2

# # single 3-fractal neurons (9 synapses feed into 9 dendrites, feed into 3 dendrites, feed into 1 soma)
# neo = NeuralZoo(type='3fractal',**default_neuron_params)
# # neo.plot_structure()

# input = SuperInput(channels=9, type='random', total_spikes=30, duration=500)
# # raster_plot(input.spike_arrays,notebook=True)

# # Attach one input channel per synapse
# for i in range(len(neo.synapses)):
#     neo.synapses[i].add_input(input.signals[i])

# # create and run network
# net = network(name = 'network_under_test')
# net.add_neuron(neo.fractal_neuron)
# # net.neurons[list(net.neurons.keys())[0]].name = 1

    
# net.run_sim(dt = .1, tf = 500)
# net.get_recordings()
# # print(net.spikes)
# signal = net.neurons[1].dend__nr_ni.s
# ref = net.neurons[1].dend__ref.s
# plt.figure(figsize=(12,4))

# for i in range(len(neo.dendrites)):
#     for j in range(len(neo.dendrites[i])):
#         if i > 0:
#             spd = neo.dendrites[i][j].synaptic_inputs[j].phi_spd
#             plt.plot(spd[::10],'--') #, label='phi_spd_'+str(i))
# plt.plot(input.spike_arrays[1],np.zeros(len(input.spike_arrays[1])),'xr', label='input event')
# plt.show()
# plt.close()

# plt.figure(figsize=(12,4))
# for i in range(len(neo.dendrites)):
#     for j in range(len(neo.dendrites[i])):
#         dend_s = neo.dendrites[i][j].s
#         if i == 0:
#             plt.plot(dend_s[::10],'-', linewidth=1, label='dend layer '+str(i))
#         if i == 1:
#             plt.plot(dend_s[::10],'--', linewidth=1, label='dend layer '+str(i))

# plt.show()
# plt.close()
# plt.figure(figsize=(12,4))
# plt.plot(ref[::10], linewidth=2, color='crimson', label='refractory signal')
# plt.plot(net.t,net.signal[0], label='soma signal')
# plt.plot(net.spikes[1],net.spike_signals[0],'xk', label='neuron fires')
# plt.axhline(y = neo.s_th, color = 'purple', linestyle = '--')
# plt.legend()
# plt.show()
# plt.close()

# times = np.arange(0,150,25)
# indices = np.zeros(len(times)).astype(int)
# def_spikes = [indices,times]
# input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes, duration=150)
# input_ = SuperInput(channels=1, type='defined', defined_spikes=def_spikes, duration=150)

# default_neuron_params['beta_ni'] = 2*np.pi*1e2
# # default_neuron_params['ib_n'] = default_ib
# default_neuron_params['s_th'] = 0.3
# synaptic_structure = [[[1]],[[1]]]
# weights = [[[1]]]

# mono_dend_ss = NeuralZoo(type="custom",weights=weights, synaptic_structure=synaptic_structure,**default_neuron_params)
# # mono_dend_ss.plot_custom_structure()

# print(mono_dend_ss.synapses)
# print(mono_dend_ss.synapses[0][0][0])
# print(mono_dend_ss.synapses[1][0][0])
# mono_dend_ss.synapses[0][0][0].add_input(input.signals[0])
# mono_dend_ss.synapses[1][0][0].add_input(input_.signals[0])

# print(mono_dend_ss.synapses[0][0][0].__dict__.keys())
# print(mono_dend_ss.synapses[1][0][0].__dict__.keys())

# print(mono_dend_ss.synapses[0][0][0].input_signal.__dict__)
# print(mono_dend_ss.synapses[1][0][0].input_signal.spike_times)


# net = network(name = 'network_under_test')
# net.add_neuron(mono_dend_ss.neuron)
# net.run_sim(dt = .01, tf = 150)
# net.get_recordings()

# print(mono_dend_ss.synapses[0][0][0].__dict__.keys())
# print(mono_dend_ss.synapses[1][0][0].__dict__.keys())


# print(mono_dend_ss.neuron.dend__nr_ni.synaptic_connection_strengths)
# print(mono_dend_ss.neuron.dend__nr_ni.dendritic_connection_strengths)
# spd = mono_dend_ss.synapses[1][0][0].phi_spd
# signal = mono_dend_ss.dendrites[0][0][0].s
# dend_s = mono_dend_ss.dendrites[1][0][0].s
# ref = mono_dend_ss.neuron.dend__ref.s
# print(mono_dend_ss.dendrites[1][0][0].name)
# plt.figure(figsize=(12,4))
# plt.plot(net.t,spd, label='phi_spd')
# plt.plot(net.t,signal, label='soma signal')
# plt.plot(net.t,dend_s, label='dendrite signal')
# plt.plot(net.t,ref, label='refractory signal')
# plt.axhline(y = mono_dend_ss.s_th, color = 'purple', linestyle = '--',label='Threshold')
# plt.legend()
# plt.show()

# synaptic_structure = [[[0]],[[1]]]
# weights = [[[1]]]
# mono_dend = NeuralZoo(type="custom",weights=weights, synaptic_structure=synaptic_structure,**default_neuron_params)

# mono_dend.synapses[1][0][0].add_input(input.signals[0])
# print(mono_dend.dendrites[1][0][0].synaptic_inputs[1].input_signal.__dict__)
# net = network(name = 'network_under_test')
# net.add_neuron(mono_dend.neuron)
# net.run_sim(dt = .01, tf = 150)
# net.get_recordings()

# spd = mono_dend.synapses[1][0][0].phi_spd
# signal = mono_dend.dendrites[0][0][0].s
# ref = mono_dend.neuron.dend__ref.s

# plt.figure(figsize=(12,4))
# plt.plot(net.t,spd, label='phi_spd')
# plt.plot(net.t,signal, label='soma signal')
# plt.plot(net.t,ref, label='refractory signal')
# plt.axhline(y = mono_dend.s_th, color = 'purple', linestyle = '--',label='Threshold')
# plt.legend()
# plt.show()



# synaptic_structure = [[[1]]]
# mono = NeuralZoo(type="custom",synaptic_structure=synaptic_structure,**default_neuron_params)

# # def_spikes = [np.zeros(3).astype(int),np.array([.5,10,75])]

# times = np.arange(0,150,25)
# indices = np.zeros(len(times)).astype(int)
# def_spikes = [indices,times]
# input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes, duration=150)
# mono.synapses[0][0][0].add_input(input.signals[0])



# synaptic_structure = [[[0]],[[1]]]
# weights = [[[1]]]
# mono_dend = NeuralZoo(type="custom",weights=weights, synaptic_structure=synaptic_structure,**default_neuron_params)

# mono_dend.dendrites[0][0][0].add_input(input.signals[0])



# net = network(name = 'network_under_test')
# net.add_neuron(mono_dend.neuron)
# net.run_sim(dt = .1, tf = 150)
# net.get_recordings()
# print(net.neurons[1].spike_times)

# spd = mono_dend.dendrites[0][0][0].synaptic_inputs[1].phi_spd

# dend_s = mono_dend.dendrites[0][0][0].s
# signal = net.neurons[1].dend__nr_ni.s
# ref = net.neurons[1].dend__ref.s

# plt.figure(figsize=(12,4))
# plt.plot(net.t,net.signal[0], label='soma signal')
# # plt.plot(net.spikes[1],net.spike_signals[0],'xk', label='neuron fires')
# plt.axhline(y = mono.s_th, color = 'purple', linestyle = '--')
# plt.plot(net.t,spd, label='phi_spd')
# # plt.plot(net.t,dend_s, label='dendrite signal')
# plt.plot(ref[::10], label='refractory signal')
# plt.legend()
# plt.show()




################################################################################
                        ###  Nine Pixel Example  ###
################################################################################


# nine_pixel_params['weights']= [
#         [[1.5,.9678933,.3]],
#         [[0.5,0.5],[0.5,0.5],[0.5,0.5]],
#         [[0.35,-0.65],[0.35,-0.65],[0.35,-0.65],[0.35,-0.65],[0.35,-0.65],[0.35,-0.65]]
#     ]
# nine_pixel_params["s_th"] = 0.05
# # nine_pixel_params["tau_ref"] = 50
# nine_neuron = NeuralZoo(type="custom",**nine_pixel_params) 
# # nine_neuron.plot_custom_structure()

# z = np.array([0,1,4,7,8]) # z-pixel array
# v = np.array([1,4,3,6,8])-1 # v
# n = np.array([2,4,6,7,9])-1 # n
# letters = [z,v,n]

# for let in letters:
    
#     indices = let
#     times = np.ones(len(indices))*20
#     def_spikes = [indices,times]
#     input = SuperInput(channels=9, type='defined', defined_spikes=def_spikes, duration=100)

#     count = 0
#     for g in nine_neuron.synapses:
#         for s in g:
#             for i,row in enumerate(input.spike_rows):
#                 if i == int(s.name)-1:
#                     s.add_input(input_signal(name = 'input_synaptic_drive', 
#                     input_temporal_form = 'arbitrary_spike_train', spike_times = input.spike_rows[i]))
#                     count+=1
#     # print(count)

#     net = network(name = 'network_under_test')
#     net.add_neuron(nine_neuron.neuron)

#     if 'custom_neuron__syn_refraction' not in net.neurons[list(net.neurons.keys())[0]].dend__ref.synaptic_inputs.keys():
#         net.neurons[list(net.neurons.keys())[0]].name = 1
#     print(net.neurons[1].dend__nr_ni.dendritic_connection_strengths)
#     net.run_sim(dt = .1, tf = 100)
#     net.get_recordings()

#     spikes = [net.spikes[0],net.spikes[1]*1000]
#     # nine_neuron.arbor_activity_plot()
#     print(len(spikes[1]))
#     import matplotlib.pyplot as plt
#     # spd = nine_neuron.dendrites[0][0][0].synaptic_inputs[1].phi_spd

#     dend_s = nine_neuron.dendrites[0][0][0].s
#     signal = net.neurons[1].dend__nr_ni.s
#     ref = net.neurons[1].dend__ref.s

#     plt.figure(figsize=(12,4))
#     plt.plot(net.t,net.signal[0], label='soma signal')
#     # plt.plot(net.spikes[1],net.spike_signals[0],'xk', label='neuron fires')
#     plt.axhline(y = nine_neuron.s_th, color = 'purple', linestyle = '--')
#     # plt.plot(net.t,spd, label='phi_spd')
#     # plt.plot(net.t,dend_s, label='dendrite signal')
#     # plt.plot(ref[::10], label='refractory signal')
#     plt.legend()
#     plt.show()
#     # raster_plot(spikes)

# # #%%
# # # print(arb.neuron.dend__nr_ni.dendritic_connection_strengths)






















# # default_neuron_params["s_th_factor_n"] = 0.3
# default_neuron_params['w_dn'] = 1
# default_neuron_params['tau_di'] = 1000
# default_neuron_params['tau_ref'] = 35
# # default_neuron_params["beta_ni"] = 2*np.pi*1e3
# def_spikes = [[0],[.5]]
# input_ = SuperInput(channels=1, type='defined', defined_spikes=def_spikes, duration=100)

# single = NeuralZoo(type='single',**default_neuron_params)
# single.synapse.add_input(input_.signals[0])

# net = network(name = 'network_under_test')
# net.add_neuron(single.neuron)
# net.run_sim(dt = .001, tf = 100)
# net.get_recordings()
# print(net.neurons)
# spd = single.dendrite.synaptic_inputs[1].phi_spd
# phi_ref = net.neurons[1].dend__ref.phi_r
# phi_nr = net.neurons[1].dend__nr_ni.phi_r
# dend_s = single.dendrite.s
# signal = net.neurons[1].dend__nr_ni.s
# sig = single.neuron.dend__nr_ni.s
# ref = net.neurons[1].dend__ref.s
# _time_vec = single.neuron.time_params['time_vec']
# _tau_vec = single.neuron.time_params['tau_vec']
# spike_times = single.neuron.spike_times
# print(spike_times)
# print(len(signal))
# spike_signal = []
# spike_times = spike_times/single.neuron.time_params['t_tau_conversion']
# print(spike_times)
# for spike in spike_times:
#     spike_signal.append(signal[int(spike*10)])

# print(len(spike_signal))
# print(len(spike_times))

# print(net.neurons[1].__dict__)
# print(single.neuron.dend__nr_ni.s)
# print(single.s_th_factor_n)

# print(net.neurons[1].dend__nr_ni.dendritic_connection_strengths)
# plt.figure(figsize=(12,4))
# plt.plot(spd[::10], label='phi_spd')
# plt.plot(dend_s[::10], label='dendtrite signal')
# plt.plot(phi_ref[::10], label='phi_ref')
# plt.plot(phi_nr[::10], label='phi_nr')
# plt.plot(net.t,net.signal[0], label='soma signal')
# plt.plot(net.spikes[1],net.spike_signals[0],'xk', label='neuron fires')
# plt.axhline(y = single.s_th, color = 'r', linestyle = '--')
# plt.plot(ref[::10], label='refractory signal')
# plt.plot(net.spikes[1]*1000,net.spikes[0],'xk', label='neuron fires')
# plt.plot(net.spikes[1]*1000,signal[::10][int(net.spikes[1]*1000)],'xk', label='neuron fires')
# plt.legend()
# plt.show()
# single.neuron.plot_simple = True
# single.neuron.plot()
# plt.show()

# weights_3 = weights = [
#                 [[1,1,1]],
#                 [[1,1,1],[1,1,1],[1,1,1]],
#             ]


# default_neuron_params['tau_ref'] = 35
# arb = NeuralZoo(type="custom",weights=weights,**default_neuron_params) 

# # print(arb.__dict__.keys())
# # print(arb.synapses[0][0].spd_duration)
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