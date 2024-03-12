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

import time



#%%


def backend_timer_duration(backends,durations):
    """
    Creates and runs a simple 'point' neuron with periodic spiketrain
    """

    np.random.seed(10)

    # def_spikes = np.arange(0,500,100)
    # inpt = SuperInput(channels=1, type='defined', defined_spikes = def_spikes)

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

        "s_th"    : 10,

        }

    # durations = np.arange(1000,100001,1000)
    # durations = np.arange(100,501,100)
    # print(len(durations))

    
    run_times_per_duration = [[],[]]
    variance_per_duration = [[],[]]
    for b,backend in enumerate(backends):
        for d,duration in enumerate(durations):
            run_times = []
            for i in range(11):

                node = SuperNode(**params)
                node.dendrite_list[0].offset_flux = 0.5

                net = network(sim=True,dt=.1,tf=duration,nodes=[node],backend=backend,print_times=False)
                run_time = net.run_time

                if i != 0: run_times.append(run_time)

                print(
                    f"{backend} backend - duration {duration} - iteration {i} - run time {net.run_time}   ",
                    end="\r"
                    )

                del(node)
                del(net)
            run_times_per_duration[b].append(np.mean(run_times))
            variance_per_duration[b].append(np.std(run_times))
    runtime_data = [run_times_per_duration,variance_per_duration]
    return runtime_data

# durations = np.arange(1000,50001,1000)
# # durations = np.arange(100,501,100)
# backends = ['julia']
# runtime_data = backend_timer_duration(backends,durations)
# picklit(runtime_data,"results/profiling/","runtimes_pointneuron_16threads")

#%%


# plt.style.use("seaborn-v0_8-darkgrid")

# plt.figure(figsize=(8,4))
# colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# for b,backend in enumerate(backends):
#     plt.plot(durations,runtime_data[0][b],linewidth=2,label=backend,color=colors[b])
    
#     lower_bound = runtime_data[0][b]-0.5*np.array(runtime_data[1][b])
#     upper_bound = runtime_data[0][b]+0.5*np.array(runtime_data[1][b])

#     plt.fill_between(durations, lower_bound, upper_bound, 
#                      facecolor=colors[b], alpha=0.2)
    
# # ratio = np.array(runtime_data[0][0])/np.array(runtime_data[0][1])
# # plt.plot(durations,ratio,'--',linewidth=1,label="python/julia",color=colors[2])
# plt.title("Time Stepper Run Time for Single Dendrite",fontsize=16)
# plt.xlabel("Duration (ns)",fontsize=14)
# plt.ylabel("Run Time (s)",fontsize=14)
# plt.legend()
# plt.savefig("results/profiling/runtimes_point_neuron_16threads")
# plt.show()


#%%

def binary_fanin(layers):
    weights = [[np.random.rand(2) for _ in range(2**(l-1))] for l in range(1,layers+1)]
    # for w in weights:
    #     print(len(w))
    return weights

bf = binary_fanin(3)


#%%
def backend_timer_size(backends,layers):
    """
    Creates and runs a simple 'point' neuron with periodic spiketrain
    """

    np.random.seed(10)

    # def_spikes = np.arange(0,500,100)
    # inpt = SuperInput(channels=1, type='defined', defined_spikes = def_spikes)

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

        "s_th"    : 10,

        }

    run_times_per_layer = [[],[]]
    variance_per_layer= [[],[]]
    
    for b,backend in enumerate(backends):
        for l,layer in enumerate(layers):
            run_times = []
            for i in range(11):
                
                weights = binary_fanin(layer)

                node = SuperNode(weights=weights,**params)
                node.dendrite_list[0].offset_flux = 0.5

                net = network(sim=True,dt=.1,tf=1000,nodes=[node],backend=backend,print_times=False)
                run_time = net.run_time

                if i != 0: 
                    run_times.append(run_time)
                # else:
                #     node.plot_structure()

                print(
                    f"{backend} backend - layers {layer} - iteration {i} - run time {net.run_time}   ",
                    end="\r"
                    )

                del(node)
                del(net)
            run_times_per_layer[b].append(np.mean(run_times))
            variance_per_layer[b].append(np.std(run_times))

    runtime_data = [run_times_per_layer,variance_per_layer]
    picklit(runtime_data,"results/profiling/","runtimes_layers_16threads")
    return runtime_data


layers = np.arange(2,10,1).astype('int32')
backends = ['julia']
runtime_data = backend_timer_size(backends,layers)


plt.style.use("seaborn-v0_8-darkgrid")
plt.figure(figsize=(8,4))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for b,backend in enumerate(backends):
    plt.plot(layers,runtime_data[0][b],linewidth=2,label=backend,color=colors[b])
    
    lower_bound = runtime_data[0][b]-0.5*np.array(runtime_data[1][b])
    upper_bound = runtime_data[0][b]+0.5*np.array(runtime_data[1][b])

    plt.fill_between(layers, lower_bound, upper_bound, 
                     facecolor=colors[b], alpha=0.2)
    
# ratio = np.array(runtime_data[0][0])/np.array(runtime_data[0][1])
# plt.plot(durations,ratio,'--',linewidth=1,label="python/julia",color=colors[2])
plt.title("Time Stepper Run Time for Increasing Arbor size16 threads",fontsize=16)
plt.xlabel(r"Layers of Binary Fanin $N=2^{layers}$",fontsize=14)
plt.ylabel("Run Time (s)",fontsize=14)
plt.legend()
plt.savefig("results/profiling/runtimes_arbor_16threads")
plt.show()


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

        # "name"    : "point_neuron",

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

    weights = [[[]]]

    # node
    node = SuperNode(name='Neuron_0',**params)
    node2 = SuperNode(name='Neuron_1',**params)
    node.neuron.add_output(node2.synapse_list[0])
    nodes = [node,node2]

    plt.style.use("seaborn-v0_8-darkgrid")
    # node.parameter_print()
    node.uniform_input(inpt)
    net = network(sim=True,dt=1.0,tf=500,nodes=nodes,backend='python')
    print("Run time =",net.run_time)
    # print(node.neuron.dend_soma.phi_th)
    # node.plot_neuron_activity(net=net,spikes=True,phir=True,input=inpt,dend=False,ref=True)
    activity_plot(nodes,phir=True,ref=True,size=(12,8),subtitles=[node.name,node2.name],title="Networking in sim_soens")

# point_neuron()

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