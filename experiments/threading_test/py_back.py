from julia import Main as jl
import sys
sys.path.append('../../sim_soens')
sys.path.append('../../')

from sim_soens.super_input import SuperInput
from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *
from sim_soens.soen_sim import network, synapse

import time

import os
import pickle
file = "test_net.pickle"
file_to_read = open(file, "rb")
net = pickle.load(file_to_read)
file_to_read.close()

jl.include("py_to_threads.jl")
jl.include("thread_stepper.jl")


def jul_to_py(jul_net,net):
    nodes = []
    for node in net.nodes:
        for i,dend in enumerate(node.dendrite_list):
            jul_dend = jul_net["nodes"][node.name]["dendrites"][dend.name]
            dend.s     = jul_dend.s #[:-1]
            dend.phi_r = jul_dend.phir #[:-1]

            dend.ind_phi = jul_dend.ind_phi #[:-1]
            dend.ind_s = jul_dend.ind_s #[:-1]
            dend.phi_vec = jul_dend.phi_vec #[:-1]

            if "soma" in dend.name:
                spike_times = (jul_dend.out_spikes-1)* net.dt * net.time_params['t_tau_conversion']
                print("Spike times: ", spike_times)
                dend.spike_times        = spike_times
                node.neuron.spike_times = spike_times
            # # if net.print_times: print(sum(jul_net[node.name][i].s))/net.dt
        for i,syn in enumerate(node.synapse_list):
            jul_syn = jul_net["nodes"][node.name]["synapses"][syn.name]
            syn.phi_spd = jul_syn.phi_spd
        nodes.append(node)
    return nodes
    
def plot_nodes(nodes):
    print(net.nodes[0].neuron.spike_times)
    import matplotlib.pyplot as plt
    import numpy as np

    # plt.plot(net.nodes[0].neuron.dend_soma.s)
    # plt.show()

    for n,node in enumerate(nodes):
        lays = [[] for _ in range(len(node.dendrites))]
        phays = [[] for _ in range(len(node.dendrites))]
        for l,layer in enumerate(node.dendrites):
            for g,group in enumerate(layer):
                for d,dend in enumerate(group):
                    lays[l].append(dend.s)
                    phays[l].append(dend.phi_r)
        plt.style.use('seaborn-muted')
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        plt.figure(figsize=(8,4))
        for l,lay in enumerate(lays):
            if l == 0:
                lw = 4
            else:
                lw = 2
            plt.plot(
                np.mean(lay,axis=0),
                linewidth=lw,
                color=colors[l],
                label=f'Layer {l} Mean Signal'
                )
            plt.plot(
                np.mean(phays[l],axis=0),
                '--',
                linewidth=.5,
                color=colors[l],
                # label=f'Layer {l} Mean Flux'
                )
        plt.legend(loc='upper right')
        plt.show()

jul_net = jl.load("aftr_update.jld2")["data"]
nodes = jul_to_py(jul_net,net)
plot_nodes(nodes)

# def make_updates(net,nodes,config):
#     # check spiking output
#     spikes = array_to_rows(net.spikes,config.digits)

#     # define error by difference of desired with actual spiking for each node
#     errors = []
#     for nd in range(config.digits):
#         errors.append(desired[nd][digit] - len(spikes[nd]))

#     # output spike totals from each class
#     output = []
#     for nd in range(config.digits):
#         output.append(len(spikes[nd]))

#     # track outputs associated with each class
#     outputs[digit].append(output)

#     # clear data
#     for node in nodes:
#         node.neuron.spike_times                         = []
#         node.neuron.spike_indices                       = []
#         node.neuron.electroluminescence_cumulative_vec  = []
#         node.neuron.time_params                         = []

#     s = time.perf_counter()
    
#     offset_sums = [0 for _ in range(config.digits)]

#     # on all but every tenth run, make updates according to algorithm 1 with elasticity
#     if config.run%10 != 0 or config.run == 0:

#         if config.probabilistic == 1:
#             # print("Determined update")
#             if config.elasticity=="elastic":
#                 if sample == 0 and config.run == 0: print("elastic")
#                 for n,node in enumerate(nodes):
#                     for l,layer in enumerate(node.dendrites):
#                         for g,group in enumerate(layer):
#                             for d,dend in enumerate(group):
#                                 if 'ref' not in dend.name and 'soma' not in dend.name:
#                                     step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
#                                     flux = np.mean(dend.phi_r) + step #dend.offset_flux
#                                     if flux > 0.5 or flux < config.low_bound:
#                                         step = -step
#                                     dend.offset_flux += step
#                                     offset_sums[n] += dend.offset_flux
#                                 dend.s = []

#             if config.elasticity=="inelastic":
#                 if sample == 0 and config.run == 0: print("inealstic")
#                 for n,node in enumerate(nodes):
#                     for l,layer in enumerate(node.dendrites):
#                         for g,group in enumerate(layer):
#                             for d,dend in enumerate(group):
#                                 if 'ref' not in dend.name and 'soma' not in dend.name:
#                                     step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
#                                     flux = np.mean(dend.phi_r) + step #dend.offset_flux
#                                     if flux > 0.5 or flux < config.low_bound:
#                                         step = 0
#                                     dend.offset_flux += step
#                                     offset_sums[n] += dend.offset_flux
#                                 dend.s = []
#                                 dend.phi_r = []

#             if config.elasticity=="unbounded":
#                 if sample == 0 and config.run == 0: print("unbounded")
#                 for n,node in enumerate(nodes):
#                     for l,layer in enumerate(node.dendrites):
#                         for g,group in enumerate(layer):
#                             for d,dend in enumerate(group):
#                                 if 'ref' not in dend.name and 'soma' not in dend.name:
#                                     step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
#                                     dend.offset_flux += step
#                                     offset_sums[n] += dend.offset_flux
#                                 dend.s = []
#                                 dend.phi_r = []


#     # on the tenth run test, but don't update -- save full nodes with data
#     else:
#         # print("Skipping Update")
#         if sample == 0 and config.run%50 == 0:
#             # save the nodes!
#             picklit(
#                 nodes,
#                 f"{path}{name}/full_nodes/",
#                 f"full_0_{digit}_nodes_at_{config.run}"
#                 )
#             for node in nodes:
#                 for dend in node.dendrite_list:
#                     dend.s = []
#                     dend.phi_r = []

#     f = time.perf_counter()
#     # print("Update time: ", f-s)
#     # print("Total runtime", f-start)
#     for o,offset in enumerate(offset_sums):
#         offset_sums[o] = np.round(offset,2)

#     print(f"  {sample}  -  [{digit} -> {np.argmax(output)}]  -  {np.round(f-start,1)}  -  {output} - {offset_sums} ")

#     # CSV data
#     List = [sample,digit,output,errors,np.argmax(output),f-start,net.init_time,net.run_time,offset_sums]
#     with open(f'{path}{name}/learning_logger.csv', 'a') as f_object:
#         writer_object = writer(f_object)
#         writer_object.writerow(List)
#         f_object.close()

#     # delete old objects
#     del(net)
#     del(input_)





