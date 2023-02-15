import numpy as np
import matplotlib.pyplot as plt

from soen_plotting import raster_plot, activity_plot

from super_input import SuperInput
from params import net_args 

from super_library import NeuralZoo
from super_functions import *
from soen_sim import network, dendrite, HardwareInTheLoop
from soen_component_library import common_synapse



times = np.concatenate([np.arange(0,500,75),np.arange(500,1000,75)])
indices = np.zeros(len(times)).astype(int)
def_spikes = [indices,times]
input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes, duration=500)


WA = [[[.6,.5]]]

nA = NeuralZoo(type='custom',weights=WA)
# print("NAME: ", nA.name)
nA.synaptic_layer()
nA.uniform_input(input)

exin = ["plus","minus"]
nA.trace_dendrites = []
for lay in nA.dendrites[1:]:
    for group in lay:
        for i,d in enumerate(group):
            cs = WA[0][0][i]
            # print(cs)
            for ei in exin:
                trace_dend = dendrite(name=f'd{i}_{ei}')
                trace_dend.add_input(d,connection_strength=cs)#2*np.random.rand())
                syn = common_synapse(f'{d.name}_tracesyn_{trace_dend.name}_{np.random.rand()}')
                trace_dend.add_input(syn,connection_strength=1)
                nA.trace_dendrites.append(trace_dend)
                nA.dendrite_list.append(trace_dend)
                nA.synapse_list.append(syn)


WB = [[[.4,.5]]]

nB = NeuralZoo(type='custom',weights=WB)
nB.synaptic_layer()
nB.uniform_input(input)
exin = ["plus","minus"]
nB.trace_dendrites = []
for lay in nB.dendrites[1:]:
    for group in lay:
        for i,d in enumerate(group):
            cs = WB[0][0][i]
            # print(cs)
            for ei in exin:
                trace_dend = dendrite(name=f'd{i}_{ei}')
                trace_dend.add_input(d,connection_strength=cs)#2*np.random.rand())
                syn = common_synapse(f'{d.name}_tracesyn_{trace_dend.name}_{np.random.rand()}')
                trace_dend.add_input(syn,connection_strength=1)
                nB.trace_dendrites.append(trace_dend)
                nB.dendrite_list.append(trace_dend)
                nB.synapse_list.append(syn)


# print(HW.__dict__)
nodes=[nA,nB]

# plasticity=True
plasticity=False

if plasticity==True:
    title="Error Module Engaging Plasticity at t=500ns (Neuron 2 Correct Output)"
    HW = HardwareInTheLoop()
else:
    title="No Plasiticity (Neuron 2 Correct Output)"
    HW = None


net = network(sim=True,dt=.01,tf=1000,nodes=nodes,null_synapses=True,hardware=HW)

# # print(nA.trace_dendrites[0].__dict__.keys(),"\n\n")

# nA.plot_neuron_activity(net,phir=True,input=input)


subtitles= ["Neuron 1","Neuron 2"]
activity_plot(nodes,net,title=title,subtitles=subtitles, size=(16,6))


# plt.figure(figsize=(16,8))
# for i,trace in enumerate(nA.trace_dendrites):
#     plt.plot(net.t,trace.phi_r,'--',label="phi "+str(i))
#     # plt.plot(net.t,trace.s, label = "signal "+str(i))
# plt.legend()
# plt.show()
# plt.figure(figsize=(16,8))
# for i,trace in enumerate(nB.trace_dendrites):
#     plt.plot(net.t,trace.phi_r,'--',label="phi "+str(i))
#     # plt.plot(net.t,trace.s, label = "signal "+str(i))
# plt.legend()
# plt.show()

'''
Backend notes:
 - neuronA.source_type
    - qd, ec, delay_delta
    - line 383 -> argmin in forloop?
 - tau_vec
    - necessary?
 - just couple 
 - hardware in the loop module in soen_functions?
    - perhaps object that gets passed around with internal logic functions
    - refers to neurons by name and compares to inform update input
 - if plasiticity, check condition of plastic modules and change r_fq accordingly
    - pass extra info into dendrite updater?
'''
