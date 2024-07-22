#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 10 16:12:57 2024

@author: sadler

function with the SNN architecture to solve the graph partitioning probelm for a given negative Q matrix
"""

from inspect import GEN_SUSPENDED
import sys

import numpy as np

from constants_equations import *
from plot_functions import *

sys.path.append(r'C:\\Users\\sra1\\github\\sim_soens')
from sim_soens.soen_sim_data import *
from sim_soens.super_input import SuperInput
from sim_soens.super_node import SuperNode
from sim_soens.soen_components import network, synapse
from sim_soens.super_functions import *
from sim_soens.soen_plotting import raster_plot, activity_plot

def get_simulation_time(N):
    simulation_time = 200*N
    return simulation_time

def simulate_HNN(W, b, simulation_time, path, r=np.random.randint(0, 2**23-1)):
    
    # W = (2 * max_flux / (np.max(np.sum(np.abs(C), axis=0)))) * C
    
    N = W.shape[0] # number of nodes/neurons in each neuron group
    

    nodes = []
    for n in range(N):
        neuron = SuperNode(name=f"neuron_{n}",s_th=spiking_threshold, ib_n=b[n], offset_flux=flux_offset)
        nodes.append(neuron)
        
    inpts = []
    for i in np.arange(5):
        inpt = SuperInput(channels=N, type='random', total_spikes=math.ceil(N*3/4), duration=simulation_time*((i+1)/5)*(2/3))
        inpts.append(inpt)

    for j, inpt in enumerate(inpts):
        for i,channel in enumerate(inpt.signals):        
            nodes[i].synapse_list[0].add_input(channel)
        
    for i in range(N):
        for j in range(N):
            if W[i][j]!=0:
    
                syn = synapse(name=f'{nodes[j].neuron.name}_syn{i}-{j}')
                nodes[j].synapse_list.append(syn)
                nodes[j].neuron.dend_soma.add_input(syn,connection_strength=W[i][j])
                
                nodes[i].neuron.add_output(syn)
    
    
    net = network(
        sim     = True,
        tf      = simulation_time,
        nodes   = nodes,
        backend = 'python',
        dt=time_step
    )
    
    if N < 101:
        activity_plot(nodes, net=net, legend=False ,phir=True, title="SNN Neuron Activity", path=path, size=(simulation_time/50, 2*N))

    data = np.zeros(N, dtype=float)
    for i, n in enumerate(nodes):
        phi_r = n.dendrites[0][0][0].phi_r
        l = len(phi_r)
        phi_r_ave = np.average(phi_r[math.floor(0.95*l):])
        data[i] = phi_r_ave
    data_path = f'{path}_data'
    compress_pickle(data, data_path)
    
    return data_path