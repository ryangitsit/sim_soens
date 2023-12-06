#%%
import numpy as nps
import matplotlib.pyplot as plt
import random

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

# syn_struct = [ [[[np.random.rand()]]] for _ in range(18)]

# test_node = SuperNode(name='test_node',synaptic_structure=syn_struct)
# test_node.parameter_print()
# print(test_node.synapse_list[0].name)
# print(test_node.dendrite_list[0].name)
# print(test_node.neuron.dend_soma.__dict__.keys())



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
        hi = SuperNode(
            name    = f"hidden_in_{i}",
            # synaptic_structure = [ [[[np.random.rand()]]] for _ in range(9)]
        )
        # hi.neuron.dend_soma.offset_flux = random.uniform(-1, 1)
        hidden_in.append(hi)
        
    for i in range(targets):
        ti = SuperNode(
            name = f"target_in_{i}",
            # synaptic_structure = [ [[[np.random.rand()]]] for _ in range(3)]
            )
        # ti.neuron.dend_soma.offset_flux = random.uniform(-1, 1)
        target_in.append(ti)


    # make internal layers
    layers = [[] for _ in range(depth)]
    for layer in range(depth):

        hidden = []
        target = []

        for i in range(pixels):
            
            if layer == 0:
                hd = SuperNode(
                    name    = f"hidden_{layer}{i}",
                    synaptic_structure = [ [[[np.random.rand()]]] for _ in range(9)]
                )

                tg = SuperNode(
                    name = f"target_{layer}{i}",
                    weights = [[np.random.rand(3)]]
                    )
                tg.synaptic_layer()
                hd.neuron.dend_soma.offset_flux = random.uniform(-1, 1)
                tg.neuron.dend_soma.offset_flux = random.uniform(-1, 1)
                hidden.append(hd)
                target.append(tg)

            else:
                hd = SuperNode(
                    name    = f"hidden_{layer}{i}",
                    synaptic_structure = [ [[[np.random.rand()]]] for _ in range(9)]
                )

                tg = SuperNode(
                    name = f"target_{layer}{i}",
                    synaptic_structure = [ [[[np.random.rand()]]] for _ in range(9)]
                    )
                
                hd.neuron.dend_soma.offset_flux = random.uniform(-1, 1)
                tg.neuron.dend_soma.offset_flux = random.uniform(-1, 1)
                hidden.append(hd)
                target.append(tg)

        layers[layer].append(hidden)
        layers[layer].append(target)


    # connect input layers to first internal layer
    for idx,node_i in enumerate(hidden_in):
        # print(node_i.name)
        for node_j in layers[0][0]:
            syn = node_j.synapse_list[idx]
            node_i.neuron.add_output(syn)
            # print("  ",node_j.name)
            # for syn in node_j.synapse_list:
            #     node_i.neuron.add_output(syn)

    for idx,node_i in enumerate(target_in):
        # print(node_i.name)
        for node_j in layers[0][1]:
            # print("  ",node_j.name)
            syn = node_j.synapse_list[idx]
            node_i.neuron.add_output(syn)
            # for syn in node_j.synapse_list:
                # node_i.neuron.add_output(syn)


    # connect internal layers sequentially
    for i,layer in enumerate(layers[:-1]):

        # hidden to next hidden and target
        for idx_i,hidden_node in enumerate(layer[0]):
            # print(hidden_node.name)

            for idx_j,hidden_next in enumerate(layers[i+1][0]):
                # print("  ",hidden_next.name)
                syn = hidden_next.synapse_list[idx_i]
                hidden_node.neuron.add_output(syn)

            # for idx_j,target_next in enumerate(layers[i+1][1]):
            #     # print("  ",target_next.name)
            #     syn = target_next.synapse_list[idx_i]
            #     hidden_node.neuron.add_output(syn)

        # target to next hidden and target
        for target_node in layer[1]:
            # print(target_node.name)

            # for hidden_next in layers[i+1][0]:
            #     # print("  ",hidden_next.name)
            #     for syn in hidden_next.synapse_list:
            #         target_node.neuron.add_output(syn)

            for target_next in layers[i+1][1]:
                # print("  ",target_next.name)
                for syn in target_next.synapse_list:
                    target_node.neuron.add_output(syn)

    return hidden_in, target_in, layers


# hidden_in, target_in, layers = make_sigprop_net()


def run_forward_pass(hidden_in, target_in, layers, duration):

    all_nodes = hidden_in + target_in
    for layer in layers:
        col = layer[0]+layer[1]
        all_nodes += col
    
    # print(len(all_nodes))

    # for node in all_nodes:
    #     print(node.neuron.name)

    sig_net = network(
    sim     = True,      # run simulation
    tf      = duration,  # total duration (ns)
    nodes   = all_nodes, # nodes in network to simulate
    backend = 'julia'
    )
    return sig_net, all_nodes

# sig_net, all_nodes = run_forward_pass(hidden_in, target_in, layers, duration)


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

# sig_raster(sig_net.spikes,all_nodes)



def plot_average_activity(net):
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

import numpy as np
def make_data(time):
    '''
    returns list of letters in pixel-array form
    '''
    # non-noisy nine-pixel letters
    letters = {
        'z': [1,1,0,
              0,1,0,
              0,1,1],

        'v': [1,0,1,
              1,0,1,
              0,1,0],

        'n': [0,1,0,
              1,0,1,
              1,0,1]
    }

    in_spikes = {}

    for name,letter in letters.items():
        indices = np.where(np.array(letter)==1)[0]
        in_spikes[name] = [indices,np.ones(len(indices))*time]

    return in_spikes

# data = make_data(25)



# def run_sig_prop(duration,interval):

hidden_in, target_in, layers = make_sigprop_net()
input_data = make_data(25)

#%%


def add_sigprop_input(hidden_in,target_in,input_data,letter_idx,duration,interval,targ=True):

    spike_times = input_data

    sigprop_input = SuperInput(type='defined', defined_spikes=spike_times)

    for i,node in enumerate(hidden_in):
        # for syn_idx, syn in enumerate(node.synapse_list):
            # print(syn.name)
        syn = node.synapse_list[0]
        syn.add_input(sigprop_input.signals[i])

    if targ==True:
        target_input = SuperInput(type='defined', defined_spikes=np.arange(0,duration,interval))
        for i,node in enumerate(target_in):
            if i == letter_idx:
                syn = node.synapse_list[0]
                # print(syn.name)
                syn.add_input(target_input.signals[0])


        # np.random.seed(letter_idx)
        # target_input = SuperInput(channels=3, type='random', total_spikes=30, duration=500)
        # # target_input.plot()
        # for i,node in enumerate(target_in):
        #     syn = node.synapse_list[0]
        #     # print(syn.name)
        #     syn.add_input(target_input.signals[i])


    return hidden_in, target_in

def cos_sim(a,b):
    cos_sims = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    # print(f"Cosine Simularities: {cos_sims}")
    return cos_sims


#%%
duration = 500
interval = 50
letters = list(input_data.keys())
all_spikes = [[] for _ in range(3)]
eta = .1

cos_sims = [[] for _ in range(3)]

for epoch in range(5000):
    print(f"\n\n====================== EPOCH {epoch} ======================")

    plt.style.use('seaborn-v0_8-dark-palette')
    fig, axs = plt.subplots(1, 3,figsize=(16,6))
    fig.subplots_adjust(hspace=0)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    outputs = [[] for _ in range(len(letters))]
    targ_outputs = [[] for _ in range(len(letters))]
    for run,letter in enumerate(letters):
        letter_idx = [i for i,let in enumerate(letters) if letter==let][0]

        if epoch>0 or run>0:
            # print(all_nodes[0].synapse_list[0].input_signal.__dict__.keys())
            for node in all_nodes:
                node.synapse_list[0].input_signal.spike_times=[]
                node.synapse_list[0].phi_spd = []
                node.neuron.dend_soma.phi_r  = []
                node.neuron.dend_soma.s      = []
        # else:
        #     print(hidden_in[0].synapse_list[0].__dict__.keys())

        hidden_in, target_in = add_sigprop_input(
            hidden_in, target_in,input_data[letter],letter_idx,duration,interval,targ=True
            )
        
        net, all_nodes = run_forward_pass(hidden_in, target_in, layers, duration)

        for l,layer in enumerate(layers):
            updates = np.zeros(len(layer[0]))
            errors = np.zeros(len(layer[0]))
            for i in range(len(layer[0])):
                hid_node = layer[0][i]
                targ_node = layer[1][i]
                hid_spks =  len(hid_node.neuron.spike_times)
                targ_spks = len(targ_node.neuron.spike_times)
                error = targ_spks - hid_spks
                errors[i] = error
                # print(hid_spks, ' -- ', targ_spks, ' --> ', error) 
                update_dend = hid_node.neuron.dend_soma
                step = np.min([error*eta*np.mean(update_dend.phi_r),.5])
                update_dend.offset_flux += step
                updates[i] = step

                if l == len(layers) - 1: 
                    outputs[run].append(hid_spks)
                    targ_outputs[run].append(targ_spks)


            # print(f"Error layer {l}: {errors}")
            # print(f"Update layer {l}: {updates}")

        for node in all_nodes:
            all_spikes[run].append(len(node.neuron.spike_times))

        

        # sig_raster(net.spikes,all_nodes)
        # plt.show()

        # if epoch%10 == 0:
        spikes = net.spikes
        for i,idx in enumerate(spikes[0]):
            node = all_nodes[int(idx)]
            if 'hidden' in node.name:
                axs[run].plot(spikes[1][i], idx, '.',ms=10,color=colors[0])
            elif 'target' in node.name:
                axs[run].plot(spikes[1][i], idx, '.',ms=10,color=colors[2])

        if run==2: plt.show()
    # plot_average_activity(net)

    # return net, all_nodes
    print(f"Outputs:\n{outputs[0]}\n{outputs[1]}\n{outputs[2]}\n")

    print(f"Target Outputs:\n{targ_outputs[0]}\n{targ_outputs[1]}\n{targ_outputs[2]}\n")

    a = np.array(outputs[0])
    b = np.array(outputs[1])
    c = np.array(outputs[2])

    for i,output in enumerate(outputs):
        sims = []
        for j,targput in enumerate(targ_outputs):
            sims.append(cos_sim(output,targput))
        print(letters[i], '  -->  ',letters[np.argmax(np.array(sims))])
        cos_sims[i].append(sims[i])

    s1 = np.linalg.norm(a - b)
    s2 = np.linalg.norm(a - c)
    s3 = np.linalg.norm(b - c)
    S  = s1 + s2 + s3
    A  = np.sqrt(S*(S-s1)*(S-s2)*(S-s3))
    print(f"Area: {A}")
    if epoch%10: print(cos_sims)


# for letter in ['z','v','n']:
# net, all_nodes = run_sig_prop(500,200)

# # %%
# for i in range(len(all_nodes)):
#     print(all_spikes[0][i], ' -- ',all_spikes[1][i], ' -- ',all_spikes[2][i])

# #%%
# for layer in layers:
#     for i in range(len(layer[0])):
#        hid_spks =  len(layer[0][i].neuron.spike_times)
#        targ_spks = len(layer[1][i].neuron.spike_times)
#        print(hid_spks, ' -- ', targ_spks, ' --> ', targ_spks - hid_spks) 

#%%
print(cos_sims)
for cos in cos_sims:
    plt.plot(cos)
plt.legend(labels=['z','v','n'])
plt.show()
#%%

a = np.array([1,5,3,9,10,6])
b = np.array([1,2,12,5,4,3])

from numpy import dot
from numpy.linalg import norm

cos_sim = dot(a, b)/(norm(a)*norm(b))

euc_dist = (norm(a-b))

print(cos_sim)
# print(euc_dist)
# print(np.random.rand(3))