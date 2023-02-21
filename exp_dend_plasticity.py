import numpy as np
import matplotlib.pyplot as plt

from soen_plotting import raster_plot, activity_plot

from super_input import SuperInput
from params import net_args 

from super_library import NeuralZoo
from super_functions import *
from soen_sim import network, dendrite, HardwareInTheLoop
from soen_component_library import common_synapse

"""
ADD NEURON OUTOUT TO TRACE DENDRITES
"""


times_1 = np.concatenate([np.arange(0,500,75),
                          np.arange(500,1000,75),
                          np.arange(1000,1500,75)])

times_2 = np.concatenate([np.arange(0,500,75),
                          np.arange(500,1000,75),   
                          np.arange(1000,1500,75)])

indices = np.concatenate([np.zeros(len(times_1)).astype(int),np.ones(len(times_2)).astype(int)])
times = np.concatenate([times_1,times_2])
def_spikes = [indices,times]
input = SuperInput(channels=2, type='defined', defined_spikes=def_spikes, duration=1500)

# raster_plot(input.spike_arrays)


trace_factor=.25
threshold=0.5
trace_syn_factor=1

WA = [[[.6,.5]]]

nA = NeuralZoo(type='custom',weights=WA,s_th=threshold)
# print("NAME: ", nA.name)
nA.synaptic_layer()
# nA.uniform_input(input)
nA.synapse_list[0].add_input(input.signals[0])
nA.synapse_list[1].add_input(input.signals[1])
print(nA.synapse_list)

exin = ["plus","minus"]
nA.trace_dendrites = []
for lay in nA.dendrites[1:]:
    for group in lay:
        for i,d in enumerate(group):
            cs = WA[0][0][i]*trace_factor
            # print(cs)
            for ei in exin:
                trace_dend = dendrite(name=f'n1_d{i}_{ei}')
                trace_dend.add_input(d,connection_strength=cs)#2*np.random.rand())
                syn = common_synapse(f'{d.name}_tracesyn_{trace_dend.name}_{int(np.random.rand()*100000)}')
                trace_dend.add_input(syn,connection_strength=trace_syn_factor)
                nA.trace_dendrites.append(trace_dend)
                nA.dendrite_list.append(trace_dend)
                nA.synapse_list.append(syn)


WB = [[[.4,.5]]]

nB = NeuralZoo(type='custom',weights=WB,s_th=threshold)
nB.synaptic_layer()
# nB.uniform_input(input)
nB.synapse_list[0].add_input(input.signals[0])
nB.synapse_list[1].add_input(input.signals[1])

exin = ["plus","minus"]
nB.trace_dendrites = []
for lay in nB.dendrites[1:]:
    for group in lay:
        for i,d in enumerate(group):
            cs = WB[0][0][i]*trace_factor
            # print(cs)
            for ei in exin:
                trace_dend = dendrite(name=f'n2_d{i}_{ei}')
                trace_dend.add_input(d,connection_strength=cs)#2*np.random.rand())
                syn = common_synapse(f'{d.name}_tracesyn_{trace_dend.name}_{int(np.random.rand()*100000)}')
                trace_dend.add_input(syn,connection_strength=trace_syn_factor)
                nB.trace_dendrites.append(trace_dend)
                nB.dendrite_list.append(trace_dend)
                nB.synapse_list.append(syn)


# print(HW.__dict__)
nodes=[nA,nB]

plasticity=True
# plasticity=False

if plasticity==True:
    title="Error Module Engaging Plasticity at t=500ns (Neuron 2 Correct Output)"
    HW = HardwareInTheLoop(freq_factor=50)
else:
    title="No Plasiticity (Neuron 2 Correct Output)"
    HW = None


net = network(sim=True,dt=.1,tf=1500,nodes=nodes,null_synapses=True,new_way=True,hardware=HW)

# # print(nA.trace_dendrites[0].__dict__.keys(),"\n\n")

# nA.plot_neuron_activity(net,phir=True,input=input)


subtitles= ["Neuron 1","Neuron 2"]
activity_plot(nodes,net,title=title,subtitles=subtitles,input=input, size=(16,6),phir=True)


fig, axs = plt.subplots(2, 1,figsize=(12,6))
fig.suptitle("Change in Biases Associated with Traces over Time",fontsize=18)
for k,v in HW.trace_biases.items():
    if "n1" in k:
        axs[0].plot(v,label=k)
    else:
        axs[1].plot(v,label=k)
axs[0].set_title("Neuron 1")
axs[1].set_title("Neuron 2")
axs[0].legend()
plt.show(block=False)




fig, axs = plt.subplots(2, 1,figsize=(12,6))
for i,trace in enumerate(nA.dendrite_list):
    if "plus" not in trace.name and "minus" not in trace.name:
        if "ref" not in trace.name and "nr_ni" not in trace.name:
            # axs[0].plot(net.t,trace.phi_r,'--',label="phi "+trace.name)
            axs[0].plot(net.t,trace.bias_dynamics, label = trace.name)
axs[0].set_title("Neuron 1")
for i,trace in enumerate(nB.dendrite_list):
    if "plus" not in trace.name and "minus" not in trace.name:
        if "ref" not in trace.name and "nr_ni" not in trace.name:
            # axs[1].plot(net.t,trace.phi_r,'--',label="phi "+trace.name)
            axs[1].plot(net.t,trace.bias_dynamics, label = trace.name)
axs[0].set_title("Neuron 1")
axs[1].set_title("Neuron 2")
fig.suptitle("Change in Biases of Forward Dendrites",fontsize=18)
plt.legend()
plt.show(block=False)

fig, axs = plt.subplots(2, 1,figsize=(12,6))
for i,trace in enumerate(nA.trace_dendrites):
    # axs[0].plot(net.t,trace.phi_r,'--',label="phi "+trace.name)
    axs[0].plot(net.t,trace.s, label = trace.name)
axs[0].set_title("Neuron 1")

for i,trace in enumerate(nB.trace_dendrites):
    # axs[1].plot(net.t,trace.phi_r,'--',label="phi "+trace.name)
    axs[1].plot(net.t,trace.s, label = trace.name)
axs[1].set_title("Neuron 2")
fig.suptitle("Trace Dendrite Signals",fontsize=18)
plt.legend()
plt.show()

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

 - Consequent of inactive trace dendrites on feed-forward dendrites?
'''
