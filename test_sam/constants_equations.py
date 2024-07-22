#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 20 16:11:33 2024

@author: sadler

JJ neuron and SNN constants and equations
"""

from inspect import GEN_SUSPENDED
import numpy as np
import bz2
import _pickle as cPickle

# Pickle a file and then compress it into a file with extension 
def compress_pickle(data, filepath):
    with bz2.BZ2File(filepath + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)
 
# Load any compressed pickle file
def decompress_pickle(filepath):
    data = bz2.BZ2File(filepath + '.pbz2', 'rb')
    data = cPickle.load(data)
    return data

save_to = 'C:\\Users\\sra1\\github\\sim_soens\\test_sam\\simulations'

gpp_graph_types = ['trivial', 'WS', 'BA', 'PL', 'ER']
tsp_graph_types = ['random']

three_node = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
four_node = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
five_node = np.array([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0]])
six_node = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]])
seven_node = np.array([[0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0]])
trivial_graphs = [three_node, four_node, five_node, six_node, seven_node]

# constants related to finding the ideal partition
C_alpha = 1
B_gamma = 1

#
spiking_threshold = 0.1375
flux_offset = 0.3685
gpp_max_neuron_input = 0.78*2*flux_offset # max = 1.6*flux_offset min = 0.005*flux_offset
tsp_max_synaptic_weight = 0.10*2*0.5*flux_offset
ib_final = 1.8
ib_initial = ib_final
time_step = 0.05