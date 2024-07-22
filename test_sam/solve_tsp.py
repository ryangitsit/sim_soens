#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 21 14:48:37 2024

@author: sadler

solving the traveling salesman problem

run using:
cd github/sim_soens/test_sam
python solve_tsp.py random 7 runs=3
"""

import sys
import os
import time

import numpy as np
import networkx as nx

from constants_equations import *
from hnn import *
from graph_functions import *

r = np.random.randint(0, 2**23-1)

if len(sys.argv) < 3:
    print("Insufficient parameters")
    sys.exit()

graph_type = sys.argv[1] # graph type (random)
nodes = int(sys.argv[2]) # number of nodes
time_str = time.strftime("%Y%m%d-%H%M%S")

# check for valid graph type
if graph_type not in tsp_graph_types:
    print("Invalid graph type has been selected")
    sys.exit()

graph_name = f'tsp_{nodes}_{graph_type}_{time_str}'
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
if nodes < 2:
    with open(f'{parent_path}/#information.txt', 'a') as f:
        f.write("the number of nodes is less than 2. This graph is meaningless \n")
        
    sys.exit()

if graph_type == 'random':
    A = get_random_A(nodes)
    G = nx.from_numpy_array(A)


simulation_time = get_simulation_time(nodes**2)

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
    spiking_threshold: {spiking_threshold}
    flux_offset: {flux_offset}
    tsp_max_synaptic_weight: {tsp_max_synaptic_weight}
    ib_final: {ib_final}
    ib_initial: {ib_initial}''')

# create csv files about this graph
os.makedirs(f'{parent_path}/csv', exist_ok=True)
np.savetxt(f'{parent_path}/csv/A.csv', A, delimiter=',')





# Benchmark Routes
def solve_benchmark_routes():
    """
    Finds solutions to the TSP using traditional methods
    """
    greedy_cycle = nx.algorithms.approximation.traveling_salesman.greedy_tsp(G, weight="weight", source=None)
    christofides_cycle = nx.algorithms.approximation.traveling_salesman.christofides(G, weight="weight", tree=None)
    annealing_cycle = nx.algorithms.approximation.traveling_salesman.simulated_annealing_tsp(G, "greedy", weight="weight", source=None, temp=100, move="1-1", max_iterations=10, N_inner=100, alpha=0.01, seed=None)

    greedy_distance = np.sum(np.fromiter((G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(greedy_cycle)), dtype=float))
    christofides_distance = np.sum(np.fromiter((G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(christofides_cycle)), dtype=float))
    annealing_distance = np.sum(np.fromiter((G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(annealing_cycle)), dtype=float))
    plot_cycle(G, list(nx.utils.pairwise(greedy_cycle)), f"greedy_route", path=parent_path)
    plot_cycle(G, list(nx.utils.pairwise(christofides_cycle)), f"christofide_route", path=parent_path)
    plot_cycle(G, list(nx.utils.pairwise(annealing_cycle)), f"annealing_route", path=parent_path)

    with open(f'{parent_path}/#information.txt', 'a') as f:
        f.write(f'''\n\nBenchmark Routes:
        Solver \t\t Distance
        Greedy \t\t {greedy_distance:.{3}g}
        Christofides \t {christofides_distance:.{3}g}
        Annealing \t {annealing_distance:.{3}g}''')
        
    return




# SNN Partitions
def solve_snn_routes():
    """
    Solves the TSP using an SNN
    """

    Q = get_Q(A)
    np.savetxt(f'{parent_path}/csv/Q.csv', Q, delimiter=',')

    # positive_mask = Q > 0
    # positive_values = Q * positive_mask

    # negative_mask = Q < 0
    # negative_values = Q * negative_mask

    # max_neuron_input = max(np.max(np.sum(np.abs(positive_values), axis=0)), np.max(np.sum(np.abs(negative_values), axis=0)))

    # W = (max_flux / max_neuron_input) * Q

    W = tsp_max_synaptic_weight * Q
    np.savetxt(f'{parent_path}/csv/W.csv', W, delimiter=',')

    b = np.full(nodes**2, ib_initial)


    with open(f'{parent_path}/#information.txt', 'a') as f:
        f.write(f'''\n\nSNN Routes:
        Solver \t\t Distance \t Validity\n''')

    for run in np.arange(num_runs):
        r_snn = np.random.randint(0, 2**23-1)
        filepath_activity = f'{parent_path}/activity_plot_{run}'
        
        data_path = simulate_HNN(W, b, simulation_time, filepath_activity, r=r_snn)
        
        data = decompress_pickle(data_path)


        snn_x = get_x(data)

        snn_cycle, is_valid = get_pairwise_cycle(snn_x)
        snn_distance = np.sum(np.fromiter((G[n][nbr]["weight"] for n, nbr in snn_cycle if G.has_edge(n, nbr)), dtype=float))
        plot_cycle(G, snn_cycle, f"salesman_route_run_{run}", parent_path)
        
        with open(f'{parent_path}/#information.txt', 'a') as f:
            f.write(f'\t#{run} {r_snn} \t {snn_distance:.{3}g} \t\t {is_valid}\n')
    return




# Ideal Routes
def solve_ideal_routes():
    """
    Finds the ideal solution to the TSP using brute force
    """
    ideal_cycle = get_ideal_cycles(A)

    ideal_distance = np.sum(np.fromiter((G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(ideal_cycle)), dtype=float))
    
    plot_cycle(G, list(nx.utils.pairwise(ideal_cycle)), f"ideal_route", path=parent_path)
    
    with open(f'{parent_path}/#information.txt', 'a') as f:
        f.write(f'''\n\nIdeal Routes:
        Solver \t\t Distance
        Ideal \t\t {ideal_distance:.{3}g}''')
    return


solve_benchmark_routes()
solve_snn_routes()
print('\n\nNow calculating ideal route. This may take a while...\n\n')
solve_ideal_routes()
