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


W = [[[.4,.6]]]

n = NeuralZoo(type='custom',weights=W)
n.synaptic_layer()
n.uniform_input(input)

exin = ["plus","minus"]
n.trace_dendrites = []
# for lay in n.dendrites[1:]:
#     for group in lay:
#         for i,d in enumerate(group):
#             for ei in exin:
#                 trace_dend = dendrite(name=f'd{i}_{ei}')
#                 trace_dend.add_input(d,connection_strength=0.2)
#                 d.add_input(trace_dend,connection_strength=0.001)
#                 n.trace_dendrites.append(trace_dend)

net = network(sim=True,dt=.1,tf=1000,nodes=[n],new_way=True)
# print(n.trace_dendrites[0].__dict__,"\n\n")
# print(n.dendrites[1][0][0].__dict__)
# plt.plot(n.trace_dendrites)
n.plot_neuron_activity(net,phir=True)

