#%%
import numpy as np
import matplotlib.pyplot as plt
import os
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
                [np.random.rand(2)],
                [np.random.rand(2) for _ in range(2)],
                [np.random.rand(2) for _ in range(4)],
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

        elif N == 98:
            code = SuperNode(
                name = f'code_neuron_{c}',
                weights = [
                    [np.random.rand(7)],
                    [np.random.rand(7) for _ in range(7)],
                    [np.random.rand(2) for _ in range(49)],
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


def add_input(inp,nodes,method='random_windows'):
    if method == 'random_windows':
        #** check ref syn position
        input_dims = len(inp.signals)
        for n,node in enumerate(nodes):
            start = np.random.randint(input_dims-1)
            for s,syn in enumerate(node.synapse_list):
                syn.add_input(inp.signals[start%(input_dims-1)])

    elif method == 'synapse_sequential':
        input_dims = len(inp.signals)
        count=0
        for n,node in enumerate(nodes):
            for s,syn in enumerate(node.synapse_list):
                if 'ref' not in syn.name and 'soma' not in syn.name:
                    syn.add_input(inp.signals[count])
                    count+=1
        # print("Input dimensions met: ", count)
    
    return nodes

def run_network(all_nodes):
    net = network(
        sim     =True,
        tf      = 500,
        nodes   = all_nodes,
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
    # print(f"   {digit} --> {prediction} :: {outputs}")
    
    if prediction == digit:
        # print("   success!")
        success =  1
    else:
        # print("   keep learning!")
        success =  0
    return success,outputs,prediction

def make_updates(codes,targets,eta):
    update_sums = np.zeros(len(codes))
    for c,code in enumerate(codes):
        spikes = code.neuron.spike_times
        error = targets[c] - len(spikes)

        for dend in code.dendrite_list:
            if 'ref' not in dend.name:
                step = error*np.mean(dend.s)*eta
                old_offset = dend.offset_flux
                if dend.offset_flux < 0:
                    dend.offset_flux = np.max([dend.offset_flux+step,dend.phi_th])
                else:
                    dend.offset_flux = np.min([dend.offset_flux+step,dend.phi_th])
                update_sums[c]+=dend.offset_flux-old_offset
    return codes,update_sums

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


N = 98
C = 3

seed = np.random.randint(1000)
np.random.seed(seed)

exp_name = f"res_N={N}_seed={seed}"
print(exp_name)

nodes, codes = make_neurons(N,C)
nodes = connect_nodes(nodes,.1)
# nodes, codes = nodes_to_codes(nodes,codes)

#%%
# all_nodes = nodes + codes
# for n,node in enumerate(nodes):
#     picklit(
#         all_nodes,
#         f"results/res_MNIST/{exp_name}/nodes/",
#         f"init_node{n}"
#         )

digits = 3
samples = 10
eta = 0.01
#%%
def run_MNIST(exp_name,nodes,codes,digits,samples,eta):
    dataset = picklin("datasets/MNIST/","duration=5000_slowdown=100")
    exp_name = f"res_{seed}"
    runs = 100
    res_spikes = {
        str(0):[],
        str(1):[],
        str(2):[],
    }
    for run in range(runs):

        if run == 0:
            path = f"results/res_MNIST/{exp_name}"
            try:
                os.makedirs(path)    
            except FileExistsError:
                pass
            fig, axs = plt.subplots(digits,samples,figsize=(36,12))
            # plt.title(f"MNIST Reservoir + Readout Activity",fontsize=18)
            # plt.xlabel("Time (ns)")
            # plt.ylabel("Neuron Index (Readouts Neurons Above Line)")


        print(f"RUN: {run}")
        successes = 0
        seen = 0
        
        for sample in range(samples):
            print(f" sample: {sample}")
            for digit in range(digits):
                # print(f"Digit {digit} -- Sample {sample}")
    
                if run == 0:
                    inp = SuperInput(
                        type="defined",
                        channels=784,
                        defined_spikes=dataset[digit][sample]
                        )

                    # raster_plot(inp.spike_arrays)
                    nodes = add_input(inp,nodes,method='synapse_sequential')
                    
                    net = run_network(nodes)

                    spikes = net.spikes
                    axs[digit][sample].plot(spikes[1], spikes[0], '.k')
                    axs[digit][sample].axhline(y = N-.5, color = 'b', linestyle = '-') 
                    axs[digit][sample].set_xticks([])
                    axs[digit][sample].set_yticks([])


                    res_spikes[str(digit)].append([
                        net.spikes[0].astype(int).tolist(),
                        net.spikes[1].tolist()]
                        )
                    del(net)
                
                if run == 1:
                    picklit(
                        res_spikes,
                        f"results/res_MNIST/{exp_name}/",
                        f"res_spikes"
                        )

                inpt = SuperInput(
                    type="defined",
                    channels=N,
                    defined_spikes=res_spikes[str(digit)][sample]
                    )

                for c,code in enumerate(codes):
                    for s,signal in enumerate(inpt.signals):
                        code.synapse_list[s].add_input(signal)

                c_net = network(
                    sim     =True,
                    tf      = np.max(res_spikes[str(digit)][sample][1])+50,
                    nodes   = codes,
                    backend = 'julia'
                )

                success,outputs,prediction = check_success(codes,digit)

                seen      += 1
                successes += success
                if seen == 30:
                    print(f"  Epoch accuracy = {np.round(successes*100/seen,2)}%\n")
                    if successes == 30:
                        print("Converged!")
                        all_nodes = nodes + codes
                        picklit(
                            codes,
                            f"results/res_MNIST/{exp_name}/",
                            f"converged_nodes"
                            )
                        return nodes,codes

                targets = np.zeros(digits)
                targets[digit] = 10

                codes, update_sums = make_updates(codes,targets,eta)

                print(
                    f"   {digit} --> {prediction} :: {outputs} :: {np.round(update_sums,2)} :: {np.round(c_net.run_time,2)}"
                    )

                nodes,codes = cleanup(c_net,nodes,codes)


        if run == 0:
            plt.savefig(f"{path}/raster_all.png")
            plt.close()
    return nodes, codes

nodes,codes = run_MNIST(exp_name,nodes,codes,digits,samples,eta)

# %%
print(res_spikes)