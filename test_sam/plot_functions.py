#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 20 15:40:29 2024

@author: sadler

Plotting functions
"""

import seaborn as sns
from matplotlib import pyplot as plt
import os
import networkx as nx
from constants_equations import *
import math
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("cyan_red", ["cyan", "red"])

def plot_heatmaps_gpp(filepath):
    data_dict = decompress_pickle(filepath)
    spikes_arrays = data_dict['spikes_arrays']
    fluxons_arrays = data_dict['fluxons_arrays']

    def plot_heatmap(arrays, filepath, title):
        for i in np.arange(arrays.shape[0]):
            plt.figure(figsize=[20, 10])
            if i > 0:
                array = arrays[i] - arrays[i-1]
            else:
                array = arrays[0]
            array_2d = array.reshape(1, -1)
            sns.heatmap(array_2d, cmap=cmap, vmin=0, vmax=max(1, np.max(array_2d)))
            plt.xlabel("Neuron Index/Corresponding Node")
            plt.title(f"Interval {i+1} {title}")
            plt.savefig(f'{filepath}_{title}_interval_{i+1}.png')
            plt.close('all')

    plot_heatmap(spikes_arrays, filepath, 'Spikes')
    plot_heatmap(fluxons_arrays, filepath, 'Fluxons')

    return

def plot_heatmaps_tsp(filepath):
    data_dict = decompress_pickle(filepath)
    spikes_arrays = data_dict['spikes_arrays']
    fluxons_arrays = data_dict['fluxons_arrays']

    def plot_heatmap(arrays, filepath, title):
        for i in np.arange(arrays.shape[0]):
            plt.figure(figsize = [20, 10])
            if i > 0:
                array = arrays[i] - arrays[i-1]
            else:
                array = arrays[0]
            num_nodes = int(math.sqrt(array.size))
            array_2d = array.reshape(num_nodes, num_nodes)
            sns.heatmap(array_2d, cmap=cmap, vmin=0, vmax=max(1, np.max(array_2d)))
            plt.xlabel(f"Position")
            plt.ylabel(f"City")
            plt.title(f"Interval {i+1} {title}")
            plt.savefig(f'{filepath}_{title}_interval_{i+1}.png')
            plt.close('all')

        return

    plot_heatmap(spikes_arrays, filepath, 'Spikes')
    plot_heatmap(fluxons_arrays, filepath, 'Fluxons')

    return

def plot_partition(G, x, title, path=''):
    N = x.size

    color_map = np.empty(N, dtype=str)
    for i in np.arange(N):
        if x[i] == 1:
            color_map[i] = 'red'
        else:
            color_map[i] = 'cyan'
    
    plt.figure(figsize=(20, 10))
    plt.title(title)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=color_map, with_labels=True)
    
    # Create a legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Spiking/partition 1'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, label='Not spiking/partition 0')]
    plt.legend(handles=legend_elements, loc='best')

    

    os.makedirs(f'{path}', exist_ok=True)
    plt.savefig(f'{path}/{title}.png')
    plt.close('all')
    return

def plot_cycle(G, cycle_pairs, title, path=''):
    # Plot the graph with weights
    plt.figure(figsize=(20, 10))
    plt.title(title)
    pos = nx.spring_layout(G)

    # Draw all edges with thickness proportional to weight and labels rounded to 3 significant figures
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, width=weights, edge_cmap=plt.cm.Blues)
    edge_labels = {(u, v): f"{d['weight']:.3g}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Highlight the route edges
    nx.draw_networkx_edges(G, pos, edgelist=cycle_pairs, edge_color='r', width=2)
    
    os.makedirs(path, exist_ok=True)
    plt.savefig(f'{path}/{title}.png')
    plt.close('all')
    
    return