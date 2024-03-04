#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../sim_soens')

# from super_library import NeuralZoo
from sim_soens.super_input import SuperInput
from sim_soens.soen_components import network
from sim_soens.soen_plotting import activity_plot, arbor_activity, structure,raster_plot
from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *

from sim_soens.super_functions import *

from sim_soens.neuron_library import *
from sim_soens.network_library import *



#%%

def point_neuron():
    """
    Creates and runs a simple 'point' neuron with periodic spiketrain
    """

    # input
    np.random.seed(10)
    def_spikes = np.arange(0,500,100)
    inpt = SuperInput(channels=1, type='defined', defined_spikes = def_spikes)

    params = {

        "name"    : "point_neuron",

        "ib"      : 1.8,
        "ib_n"    : 1.8,

        "beta"    : 2*np.pi*1e3,
        "beta_ni" : 2*np.pi*1e3,
        "beta_di" : 2*np.pi*1e3,

        "tau"     : 150,
        "tau_ni"  : 150,
        "tau_di"  : 150,

        "s_th"    : 0.5,

        }


    # node
    node = SuperNode(**params)
    print(node.neuron.dend__ref.__dict__)
    node.parameter_print()
    node.uniform_input(inpt)
    net = network(sim=True,dt=1.0,tf=500,nodes=[node],backend='julia')
    node.plot_neuron_activity(net=net,spikes=True,phir=True,input=inpt,dend=False,ref=True)

point_neuron()

#%%
def test_max_signals():
    offsets = np.arange(.1,2.01,.1)
    # offsets = [0.7]
    nodes = []
    for i,off in enumerate(offsets):
        off = np.round(off,2)
        node = SuperNode(
            name=f'node_{off}',
            beta_di=2*np.pi*1e3,
            beta_ni=2*np.pi*1e3,beta=2*np.pi*1e3,s_th=1)
        node.neuron.dend_soma.offset_flux = off
        nodes.append(node)

    net = network(sim=True,nodes=nodes,tf=500,dt=1.0,backend='python')
    maxes = []
    plt.figure(figsize=(8,6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i,node in enumerate(nodes):
        dend = node.neuron.dend_soma
        maxes.append(np.max(dend.s))
        plt.plot(dend.s,label=node.name,color=colors[i%len(colors)])
        plt.plot(dend.phi_r,'--',color=colors[i%len(colors)])
    plt.legend()
    plt.show()

    plt.plot(offsets,maxes)
    plt.show()