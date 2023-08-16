from julia import Main as jl
import sys
sys.path.append('../../sim_soens')
sys.path.append('../../')

import os
import pickle
file = "test_net.pickle"
file_to_read = open(file, "rb")
net = pickle.load(file_to_read)
file_to_read.close()

jl.include("py_to_jul.jl")
jl.include("julia_stepper.jl")
jul_net = jl.load("out_dict.jld2")["data"]



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
            dend.spike_times        = spike_times
            node.neuron.spike_times = spike_times
        # # if net.print_times: print(sum(jul_net[node.name][i].s))/net.dt
    for i,syn in enumerate(node.synapse_list):
        jul_syn = jul_net["nodes"][node.name]["synapses"][syn.name]
        syn.phi_spd = jul_syn.phi_spd

print(net.nodes[0].neuron.spike_times)
import matplotlib.pyplot as plt

plt.plot(net.nodes[0].neuron.dend_soma.s)
plt.show()