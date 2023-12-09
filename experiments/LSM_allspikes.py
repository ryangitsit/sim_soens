#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../sim_soens')

from sim_soens.super_input import SuperInput
from sim_soens.soen_components import network
from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *
from sim_soens.neuron_library import *
from sim_soens.network_library import *
from sim_soens.super_net import SuperNet
from sim_soens.soen_plotting import raster_plot




'''
Plan
 - Reservoir of N neurons
 - Readout layer of C=num_classes neurons
 - Input dimensionality of C nuerons = N
 - Input dimensionality of N neurons = some grid size (convolutions?)
 - N neurons connected to eachothers' somas directly, arbor reserved for input
 - Arbor update rule only on C neurons
'''

def make_neurons(N,C):
    
    nodes = []
    for n in range(N):
        node = SuperNode(
            name = f'res_neuron_{n}',
            weights = [
                [np.random.rand(3)],
                [np.random.rand(3) for _ in range(3)]
            ]
        )
        node.normalize_fanin(1.5)
        nodes.append(node)


    codes = []
    for c in range(C):
        if N == 100:
            code = SuperNode(
                name = f'code_neuron_{c}',
                weights = [
                    [np.random.rand(3)],
                    [np.random.rand(3) for _ in range(3)],
                    [np.random.rand(3) for _ in range(9)],
                    [np.random.rand(4) for _ in range(27)],
                ]
            )
            code.normalize_fanin(1.5)
            codes.append(code)
        elif N == 10:
            code = SuperNode(
                name = f'code_neuron_{c}',
                weights = [
                    [np.random.rand(3)],
                    [np.random.rand(4) for _ in range(3)],
                ]
            )
            code.normalize_fanin(1.5)
            codes.append(code)


    return nodes, codes


def connect_nodes(nodes,p_connect):
    for i,N1 in enumerate(nodes):
        for j,N2 in enumerate(nodes):
            if np.random.rand() < p_connect and i!=j:
                syn = synapse(
                    name=f'{N2.name}_synsoma_from_n{i}'
                    )
                N2.synapse_list.append(syn)
                N2.neuron.dend_soma.add_input(syn,connection_strength=np.random.rand())
                N1.neuron.add_output(N2.synapse_list[-1])
    return nodes


def nodes_to_codes(nodes,codes):

    for n,node in enumerate(nodes):
        for c,code in enumerate(codes):
            node.neuron.add_output(code.synapse_list[n])
    
    return nodes, codes


def add_input(inp,nodes):
    #** check ref syn position
    input_dims = len(inp.signals)
    for n,node in enumerate(nodes):
        start = np.random.randint(input_dims-1)
        for s,syn in enumerate(node.synapse_list):
            syn.add_input(inp.signals[start%(input_dims-1)])
    
    return nodes

def run_network(nodes,codes):
    net = network(
        sim     =True,
        tf      = 500,
        nodes   = nodes+codes,
        backend = 'julia'
    )

    return net

def check_success(codes,digit):
    outputs = []
    for c,code in enumerate(codes):
        spikes = code.neuron.spike_times
        # print(f" {code.name} -> {len(spikes)}")
        outputs.append(len(spikes))
    
    prediction = np.argmax(np.array(outputs))
    print(f"   {digit} --> {prediction} :: {outputs}")
    if prediction == digit:
        # print("   success!")
        return 1
    else:
        # print("   keep learning!")
        return 0

def make_updates(codes,targets,eta):

    for c,code in enumerate(codes):
        spikes = code.neuron.spike_times
        error = targets[c] - len(spikes)

        for dend in code.dendrite_list:
            if 'ref' not in dend.name:
                step = error*np.mean(dend.s)*eta

                if dend.offset_flux < 0:
                    dend.offset_flux = np.max([dend.offset_flux+step,dend.phi_th])
                else:
                    dend.offset_flux = np.min([dend.offset_flux+step,dend.phi_th])
    return codes

def cleanup(net,nodes,codes):
    for n in nodes+codes:
        n.neuron.spikes_times = []
        n.neuron.spike_indices                       = []
        n.neuron.electroluminescence_cumulative_vec  = []
        n.neuron.time_params                         = []
        for dend in n.dendrite_list:
            dend.s = []
            dend.phi_r = []
        for syn in n.synapse_list:
            syn.phi_spd = []
    del(net)
    return nodes,codes
#%%

# classes = 3
# eta = 0.1
# for idx in range(classes):
#     np.random.seed(idx)
#     inp = SuperInput(channels=10, type='random', total_spikes=75, duration=500)
#     # raster_plot(inp.spike_arrays)

#     nodes = add_input(inp,nodes)

#     net = run_network(nodes,codes)

#     targets = np.zeros(classes)
#     targets[idx] = 10

#     make_updates(codes,targets,eta)
#     nodes,codes = cleanup(net,nodes,codes)


# %%

seed = np.random.randint(100)
np.random.seed(seed)
exp_name = f"res_{seed}"
print(exp_name)

nodes, codes = make_neurons(100,3)
nodes = connect_nodes(nodes,.3)
nodes, codes = nodes_to_codes(nodes,codes)

all_nodes = nodes + codes
# picklit(
#     all_nodes,
#     f"results/res_MNIST/{exp_name}/",
#     f"init_nodes"
#     )

digits = 3
samples = 10
eta = 0.1

def run_MNIST(exp_name,nodes,codes,digits,samples,eta):
    dataset = picklin("datasets/MNIST/","duration=5000_slowdown=100")
    exp_name = f"res_{seed}"
    runs = 100
    for run in range(runs):
        print(f"RUN: {run}")
        successes = 0
        seen = 0
        for sample in range(samples):
            print(f" sample: {sample}")
            for digit in range(digits):
                # print(f"Digit {digit} -- Sample {sample}")
    
                inp= SuperInput(
                    type="defined",
                    channels=784,
                    defined_spikes=dataset[digit][sample]
                    )
                # raster_plot(inp.spike_arrays)
                nodes = add_input(inp,nodes)
                
                net = run_network(nodes,codes)
                
                seen      += 1
                successes += check_success(codes,digit)
                if seen == 30:
                    print(f"  Epoch accuracy = {np.round(successes*100/seen,2)}%\n")
                    if successes == 30:
                        print("Converged!")
                        all_nodes = nodes + codes
                        # picklit(
                        #     all_nodes,
                        #     f"results/res_MNIST/{exp_name}/",
                        #     f"converged_nodes"
                        #     )
                        return nodes,codes

                targets = np.zeros(digits)
                targets[digit] = 10

                make_updates(codes,targets,eta)
                nodes,codes = cleanup(net,nodes,codes)
    return nodes, codes

nodes,codes = run_MNIST(exp_name,nodes,codes,digits,samples,eta)
