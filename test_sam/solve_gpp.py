#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 20 14:48:37 2024

@author: sadler

solving all graph types

run using:
cd github/sim_soens/test_sam
python solve_gpp.py WS 69 6 0.13 runs=3
"""

import sys
import os
import time

import numpy as np
import networkx as nx
# import metis

from constants_equations import *
from graph_functions import *
from plot_functions import *
from hnn import *

r = np.random.randint(0, 2**23-1)

if len(sys.argv) < 3:
    print("Insufficient parameters")
    sys.exit()

graph_type = sys.argv[1] # graph type (trivial, WS, WS_SF, ER, SF)
N = int(sys.argv[2]) # number of nodes
time_str = time.strftime("%Y%m%d-%H%M%S")

# check for valid graph type
if graph_type not in gpp_graph_types:
    print("Invalid graph type has been selected")
    sys.exit()

graph_name = f'gpp_{N}_{graph_type}_{time_str}'
parent_path = os.path.join(save_to, graph_name)
os.makedirs(parent_path, exist_ok=True)

# optional arguments
num_runs = 1
extra_args = 0
for a in sys.argv:
    if a[:5] == 'runs=':
        if a[5:].isnumeric:
            num_runs = int(a[5:])
        else:
            with open(f'{parent_path}/#information.txt', 'a') as f:
                f.write("typo in runs \n")
                
            sys.exit()
        if num_runs < 1:
            with open(f'{parent_path}/#information.txt', 'a') as f:
                f.write("need atleast 1 run1 \n")
                
            sys.exit()
        extra_args += 1

# general exception
if N < 2:
    with open(f'{parent_path}/#information.txt', 'a') as f:
        f.write("the number of nodes is less than 4. This graph is meaningless \n")
        
    sys.exit()



# scale free graphs
if graph_type == 'BA':
    m = int(sys.argv[3]) # roughly N/20
    G = nx.barabasi_albert_graph(N, m, seed=r)
    A = nx.to_numpy_array(G)

elif graph_type == 'PL':
    m = int(sys.argv[3]) # roughly N/20
    p = float(sys.argv[4]) # roughly 0.1
    G = nx.powerlaw_cluster_graph(N, m, p)
    A = nx.to_numpy_array(G)

# Small world graphs
elif graph_type == 'WS':
    k = int(sys.argv[3]) # roughly N/10
    p = float(sys.argv[4]) # roughly 0.05
    G = nx.watts_strogatz_graph(N, k, p, seed=r)
    A = nx.to_numpy_array(G)

# Random graphs
elif graph_type == 'ER':
    p = float(sys.argv[3])
    G = nx.erdos_renyi_graph(N, p, seed=r)
    A = nx.to_numpy_array(G)

# Trivial graphs
elif graph_type == 'trivial':
    if len(sys.argv) != 3  + extra_args:
        with open(f'{parent_path}/#information.txt', 'a') as f:
            f.write("Insufficient parameters \n")
        sys.exit()
    
    # exceptions
    if (N < 3) or (N > 7):
        with open(f'{parent_path}/#information.txt', 'a') as f:
            f.write("the number of nodes is outside what I have trivial graphs for \n")
        sys.exit()

    A = trivial_graphs[N - 3]
    G = nx.from_numpy_array(A)


simulation_time = get_simulation_time(N)

# Save information about this graph and the simulation
with open(f'{parent_path}/#information.txt', 'a') as f:
    f.write(f'''Graph information for {graph_name}
Simulation started on {time_str}''')
    f.write('\n\nArguments:\npython ')
    for ar in sys.argv:
        f.write(f'{ar} ')
    f.write(f'''\n\nConstants:
    seed: {r}
    simulation_time: {simulation_time}
    time_step: {time_step}
    C_alpha: {C_alpha}
    B_gamma: {B_gamma}
    spiking_threshold: {spiking_threshold}
    flux_offset: {flux_offset}
    gpp_max_neuron_input: {gpp_max_neuron_input}
    ib_final: {ib_final}
    ib_initial: {ib_initial}''')

# create csv files about this graph
os.makedirs(f'{parent_path}/csv', exist_ok=True)
np.savetxt(f'{parent_path}/csv/A.csv', A, delimiter=',')


def solve_benchmark_partitions():
    """
    Finds the partition of a graph using traditional solvers
    """
    kernighan_lin_coms = nx.algorithms.community.kernighan_lin.kernighan_lin_bisection(G)
    kernighan_lin_x = get_x_from_coms(kernighan_lin_coms)
    kernighan_lin_s = get_s_from_coms(kernighan_lin_coms)

    comp = nx.algorithms.community.girvan_newman(G)
    girvan_newman_coms = tuple(sorted(c) for c in next(comp))
    if len(girvan_newman_coms) > 2:
        with open(f'{parent_path}/#information.txt', 'a') as f:
            f.write(f'Girvan_Newman found more than 2 partitions!!!\nIgnore these results!!!\n')
    girvan_newman_x = get_x_from_coms(girvan_newman_coms)
    girvan_newman_s = get_s_from_coms(girvan_newman_coms)

    # metis_lyapunov_M, metis_lyapunov_Q = get_partition_quality(metis_x, A)
    kernighan_lin_q = nx.algorithms.community.quality.modularity(G, kernighan_lin_coms)
    kernighan_lin_co = get_co(A, kernighan_lin_s)

    girvan_newman_q = nx.algorithms.community.quality.modularity(G, girvan_newman_coms)
    girvan_newman_co = get_co(A, girvan_newman_s)

    # plot_partition(G, metis_x, "metis_partition", parent_path)
    plot_partition(G, kernighan_lin_x, "kernighan_lin_partition", parent_path)
    plot_partition(G, girvan_newman_x, "girvan_newman_partition", parent_path)

    with open(f'{parent_path}/#information.txt', 'a') as f:
        f.write(f'''\n\nBenchmark Partitions:
        Solver \t\t Modularity \t CO
        kernighan-lin \t {kernighan_lin_q:.{4}g} \t {kernighan_lin_co:.{4}g}
        girvan-newman \t {girvan_newman_q:.{4}g} \t {girvan_newman_co:.{4}g}''')
    return

def solve_ideal_partitions():
    """
    Finds the Ideal solution to the GPP
    """
    ideal_q_x, ideal_co_x = get_ideal_partitions(A)

    plot_partition(G, ideal_q_x, "ideal_q", parent_path)
    plot_partition(G, ideal_co_x, "ideal_co", parent_path)

    ideal_q_coms = get_coms_from_x(ideal_q_x)
    ideal_q_s = get_s_from_coms(ideal_q_coms)

    ideal_co_coms = get_coms_from_x(ideal_co_x)
    ideal_co_s = get_s_from_coms(ideal_co_coms)

    ideal_q_q = nx.algorithms.community.quality.modularity(G, ideal_q_coms)
    ideal_q_co = get_co(A, ideal_q_s)

    ideal_co_q = nx.algorithms.community.quality.modularity(G, ideal_co_coms)
    ideal_co_co = get_co(A, ideal_co_s)

    with open(f'{parent_path}/#information.txt', 'a') as f:
        f.write(f'''\n\nIdeal Partitions:
        Solver \t\t Modularity \t CO
        ideal_q \t {ideal_q_q:.{4}g} \t {ideal_q_co:.{4}g}
        ideal_co \t {ideal_co_q:.{4}g} \t {ideal_co_co:.{4}g}''')
    return



# SNN Partitions
def solve_snn_partitions():
    """
    Simulates a neural network to solve the GPP
    """
      # how long the simulation needs to run for

    C = get_C(A)
    B = get_B(A)

    np.savetxt(f'{parent_path}/csv/B.csv', B, delimiter=',')
    np.savetxt(f'{parent_path}/csv/C.csv', C, delimiter=',')

    b = np.full(N, ib_initial)

    with open(f'{parent_path}/#information.txt', 'a') as f:
        f.write(f'''\n\nSNN Partitions:
        Solver \t\t Modularity \t CO \n''')

    for run in np.arange(num_runs):
        for M_type, HNN_M in {'Mod': B, 'CO': C, }.items():
            W = (2 * gpp_max_neuron_input / (np.max(np.sum(np.abs(HNN_M), axis=0)))) * HNN_M

            r_snn = np.random.randint(0, 2**23 - 1)
            filepath_activity = f'{parent_path}/activity_plot_{M_type}_{run}'

            data_path = simulate_HNN(W, b, simulation_time, filepath_activity, r=r_snn)

            data = decompress_pickle(data_path)

            snn_x = get_x(data)
            plot_partition(G, snn_x, f"snn_partition_{M_type}_{run}", parent_path)

            snn_coms = get_coms_from_x(snn_x)
            snn_s = get_s_from_coms(snn_coms)

            snn_q = nx.algorithms.community.quality.modularity(G, snn_coms)
            snn_co = get_co(A, snn_s)
            
            with open(f'{parent_path}/#information.txt', 'a') as f:
                f.write(f'\t#{run} {M_type} {r_snn}\t {snn_q:.{4}g} \t {snn_co:.{4}g}\n')
    return


solve_benchmark_partitions()
solve_snn_partitions()
print('\n\nNow calculating ideal partition. This may take a while...\n\n')
solve_ideal_partitions()