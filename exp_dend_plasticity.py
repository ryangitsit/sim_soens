import numpy as np
import matplotlib.pyplot as plt

from soen_plotting import raster_plot

from super_input import SuperInput
from params import net_args 

from super_library import NeuralZoo
from super_functions import *
from soen_sim import network, dendrite



times = np.arange(0,10,2) #np.concatenate([np.arange(0,500,250),np.arange(500,1000,51)])
indices = np.zeros(len(times)).astype(int)
def_spikes = [indices,times]
input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes, duration=5)


W = [[[.4,.6]]]

n = NeuralZoo(type='custom',weights=W)
n.synaptic_layer()
n.uniform_input(input)
# print(n.neuron.name)

exin = ["plus","minus"]
n.trace_dendrites = []
for lay in n.dendrites[1:]:
    for group in lay:
        for i,d in enumerate(group):
            for ei in exin:
                trace_dend = dendrite(name=f'd{i}_{ei}')
                trace_dend.add_input(d,connection_strength=1.5)
                # d.add_input(trace_dend,connection_strength=0.001)
                n.trace_dendrites.append(trace_dend)
                n.dend_list.append(trace_dend)
# print(n.dend_list)
# print(n.neuron.__dict__)
# print(n.neuron.dend__nr_ni.dendritic_inputs)
net = network(sim=True,dt=.1,tf=5,nodes=[n])
print(n.trace_dendrites[0].phi_r,"\n\n")
# print(n.dendrites[1][0][0].__dict__)
for dend in n.trace_dendrites:
    plt.plot(net.t,dend.s)
plt.show()
n.plot_neuron_activity(net,phir=True)

