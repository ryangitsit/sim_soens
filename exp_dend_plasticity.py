import numpy as np
import matplotlib.pyplot as plt

from soen_plotting import raster_plot

from super_input import SuperInput
from params import net_args 

from super_library import NeuralZoo
from super_functions import *
from soen_sim import network, dendrite



times = np.concatenate([np.arange(0,500,250),np.arange(500,1000,51)])
indices = np.zeros(len(times)).astype(int)
def_spikes = [indices,times]
input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes, duration=500)


W = [[[.6,.8]]]

n = NeuralZoo(type='custom',weights=W)
n.synaptic_layer()
n.uniform_input(input)

exin = ["plus","minus"]
n.trace_dendrites = []
for lay in n.dendrites[1:]:
    for group in lay:
        for i,d in enumerate(group):
            cs = W[0][0][i]
            print(cs)
            for ei in exin:
                trace_dend = dendrite(name=f'd{i}_{ei}')
                trace_dend.add_input(d,connection_strength=cs)#2*np.random.rand())
                n.trace_dendrites.append(trace_dend)
                n.dendrite_list.append(trace_dend)

net = network(sim=True,dt=.01,tf=1000,nodes=[n])

# print(n.trace_dendrites[0].__dict__.keys(),"\n\n")

n.plot_neuron_activity(net,phir=True,input=input)


plt.figure(figsize=(16,8))
for i,trace in enumerate(n.trace_dendrites):
    plt.plot(net.t,trace.phi_r,'--',label="phi "+str(i))
    plt.plot(net.t,trace.s, label = "signal "+str(i))
plt.legend()
plt.show()



