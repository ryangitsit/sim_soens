#%%
import numpy as nps
import matplotlib.pyplot as plt

import sys
sys.path.append('../sim_soens')
sys.path.append('../')

from sim_soens.super_input import SuperInput
from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *
from sim_soens.soen_components import network
from sim_soens.soen_plotting import raster_plot

from sim_soens.super_algorithms import *
from sim_soens.input_library import *
#%%
syn_struct = [ [[[np.random.rand()]]] for _ in range(18)]

test_node = SuperNode(name='test_node',synaptic_structure=syn_struct)
test_node.parameter_print()
print(test_node.synapse_list[0].name)
print(test_node.dendrite_list[0].name)
# print(test_node.neuron.dend_soma.__dict__.keys())



#%%
def make_sigprop_net():

    # W = [
    #     [[.5,.5,.5]],
    #     [[.3,.3,.3],[.3,.3,.3],[.3,.3,.3]]
    # ]

    pixels  = 9
    targets = 3
    depth  = 3


    # make input layers 
    hidden_in  = []
    target_in = []

    for i in range(pixels):
        hidden_in.append(SuperNode(
            name    = f"hidden_in_{i}",
            # weights = W
        ))
        
    for i in range(targets):
        target_in.append(SuperNode(name = f"target_in_{i}"))


    # make internal layers
    layers = [[] for _ in range(depth)]
    for layer in range(depth):

        hidden = []
        target = []

        for i in range(pixels):

            hd = SuperNode(
                name    = f"hidden_{layer}{i}",
                synaptic_structure = [ [[[np.random.rand()]]] for _ in range(18)]
            )

            tg = SuperNode(
                name = f"target_{layer}{i}",
                synaptic_structure = [ [[[np.random.rand()]]] for _ in range(18)]
                )
            
            hd.neuron.dend_soma.offset_flux = np.random.rand()
            tg.neuron.dend_soma.offset_flux = np.random.rand()
            hidden.append(hd)
            target.append(tg)

        layers[layer].append(hidden)
        layers[layer].append(target)

    # layers = [
    #     [[[hidden],[target]]],
    #     [[[hidden],[target]]]]

    # print(len(layers))
    # for layer in layers:
    #     print(len(layer))
    #     for column in layer:
    #         print(len(column))
    #         for nodes in column:
    #             names = []
    #             for n in nodes:
    #                 names.append(n.name)
    #             print(names)


    # connect input layers to first internal layer
    for node_i in hidden_in:
        # print(node_i.name)
        for node_j in layers[0][0]:
            # print("  ",node_j.name)
            for syn in node_j.synapse_list:
                node_i.neuron.add_output(syn)

    for node_i in target_in:
        # print(node_i.name)
        for node_j in layers[0][1]:
            # print("  ",node_j.name)
            for syn in node_j.synapse_list:
                node_i.neuron.add_output(syn)


    # connect internal layers sequentially
    for i,layer in enumerate(layers[:-1]):

        # hidden to next hidden and target
        for idx_i,hidden_node in enumerate(layer[0]):
            # print(hidden_node.name)

            for idx_j,hidden_next in enumerate(layers[i+1][0]):
                # print("  ",hidden_next.name)
                syn = hidden_next.synapse_list[idx_i]
                hidden_node.neuron.add_output(syn)

            for idx_j,target_next in enumerate(layers[i+1][1]):
                # print("  ",target_next.name)
                syn = target_next.synapse_list[idx_i]
                hidden_node.neuron.add_output(syn)

        # target to next hidden and target
        for target_node in layer[1]:
            # print(target_node.name)

            for hidden_next in layers[i+1][0]:
                # print("  ",hidden_next.name)
                for syn in hidden_next.synapse_list:
                    target_node.neuron.add_output(syn)

            for target_next in layers[i+1][1]:
                # print("  ",target_next.name)
                for syn in target_next.synapse_list:
                    target_node.neuron.add_output(syn)

    return hidden_in, target_in, layers


hidden_in, target_in, layers = make_sigprop_net()

#%%
def add_sigprop_input(hidden_in,channels,duration,interval):

    spike_times = np.arange(0,duration,interval) 

    sigprop_input = SuperInput(type='defined', defined_spikes=spike_times)

    for i,node in enumerate(hidden_in):
        syn = node.synapse_list[0]
        # print(syn.name)
        syn.add_input(sigprop_input.signals[0])

    for i,node in enumerate(target_in):
        if i == 1:
            syn = node.synapse_list[0]
            # print(syn.name)
            syn.add_input(sigprop_input.signals[0])



channels = 9
duration = 500
interval = 100
add_sigprop_input(hidden_in,channels,duration,interval)
# %%

def run_forward_pass(hidden_in, target_in, layers, duration):

    all_nodes = hidden_in + target_in
    for layer in layers:
        col = layer[0]+layer[1]
        all_nodes += col
    
    print(len(all_nodes))

    for node in all_nodes:
        print(node.neuron.name)

    sig_net = network(
    sim     = True,      # run simulation
    tf      = duration,  # total duration (ns)
    nodes   = all_nodes, # nodes in network to simulate
    backend = 'julia'
    )
    return sig_net, all_nodes

sig_net, all_nodes = run_forward_pass(hidden_in, target_in, layers, duration)

#%%

from sim_soens.soen_plotting import *
activity_plot(all_nodes[:12],phir=True,size=(8,12))
#%%

def sig_raster(spikes,all_nodes):
    plt.style.use('seaborn-v0_8-dark-palette')
    plt.figure(figsize=(8,6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i,idx in enumerate(spikes[0]):
        node = all_nodes[int(idx)]
        if 'hidden' in node.name:
            plt.plot(spikes[1][i], idx, '.',ms=10,color=colors[0])
        elif 'target' in node.name:
            plt.plot(spikes[1][i], idx, '.',ms=10,color=colors[2])
    # plt.legend(['hidden','target'])

sig_raster(sig_net.spikes,all_nodes)


#%%
net = sig_net
raster_plot(net.spikes)

# move to within class
plt.figure(figsize=(12,4))
plt.plot(net.t,np.mean(net.signal,axis=0),label="signal")
plt.plot(net.t,np.mean(net.phi_r,axis=0),label="phi_r")
plt.legend()
plt.xlabel("Time(ns)",fontsize=16)
plt.ylabel("Signal",fontsize=16)
plt.title("Average Network Node Dynamics",fontsize=18)
plt.show()

# for node in all_nodes:
#    node.plot_neuron_activity(net=sig_net,spikes=False,phir=True)

# %%
