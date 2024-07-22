#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 20 16:12:12 2024

@author: sadler

Partition & Graph functions
"""

import numpy as np
import math
from itertools import permutations
from sys import maxsize

import networkx as nx

from constants_equations import *

def get_random_A(N): 
    A = np.zeros((N, N), dtype=float) 
    for i in np.arange(N):  
        for j in np.arange(N): 
            if i == j:
                A[i][j] = 0 
            elif i > j:
                A[i][j] = A[j][i]
            else:
                A[i][j] = np.random.uniform()
    return A


def get_Q(A): # get the Q matrix
    N = A.shape[0]
    N2 = A.size
    non_zero_elements = A[A != 0]
    min_distance = np.min(non_zero_elements)
    max_distance = np.max(A)
    Q = np.zeros((N2, N2), dtype=float)
    for i in np.arange(N2):
        for j in np.arange(N2):
            if i > j: # graph is symmetric
                Q[i][j] = Q[j][i]
            elif i == j:
                Q[i][j] = 0
            elif ((i//N == j//N) or (i%N == j%N)): # inhibit neurons in same position or representing same city
                Q[i][j] = -1
            elif (i%N == (j-1)%N) or ((i-1)%N == j%N): # excite by inverse of distance so minimum distance is preferred
                Q[i][j] = 0.75 - (A[i//N][j//N] - min_distance)*(0.5/(max_distance-min_distance))
    return Q

def get_C(A): # get the Q matrix
    N = A.shape[0]
    C = A - C_alpha
    for i in np.arange(N):
        C[i][i] -= np.sum(C[i])
    return C

def get_B(A): # Get the modulatity matrix
    m2 = np.sum(A) # 2 times number of edges
    g = np.sum(A, axis=1) # degree matrix
    B = A - (B_gamma/m2)*np.outer(g, g)
    return B

def get_x(phi_r):
    return (phi_r > 0.19).astype(int)

def get_pairwise_cycle(x):
    num_nodes = int(math.sqrt(x.size))

    route_matrix = x.reshape(num_nodes, num_nodes)

    pairwise_cycle = []
    for j in range(num_nodes):
        for i in range(num_nodes):
            if route_matrix[i][j] == 1:
                for k in range(num_nodes):
                    if route_matrix[k][(j+1) % num_nodes] == 1:
                        pairwise_cycle.append((i, k))
    
    nodes = [node for pair in pairwise_cycle for node in pair]
    is_valid = ((len(pairwise_cycle) == num_nodes) and (set(nodes) == set(range(num_nodes))))

    return pairwise_cycle, is_valid

def get_coms_from_x(x):
    coms = [[], []]
    for i, val in enumerate(x):
        coms[val].append(i)
    return coms

def get_x_from_coms(coms):
    length = 0
    for c in coms:
        length += len(c)
    x = np.zeros(length, dtype=int)
    for i in np.arange(length):
        if i in coms[0]:
            x[i] = 1
    return x

def get_s_from_coms(coms):
    length = 0
    for c in coms:
        length += len(c)
    s = -np.ones(length, dtype=int)
    for i in np.arange(length):
        if i in coms[0]:
            s[i] = 1
    return s


def get_lyapunov(x, W, b=0):
    '''
    This is the lyapunov energy of our SNN
    Should be minimized
    '''
    if b == 0:
        b = np.zeros((x.size))
    V = (-1/2)*np.matmul(np.matmul(np.transpose(x), W), x) - np.matmul(np.transpose(b), x)
    return V

def get_co(A, s):
    co = np.matmul(np.matmul(np.transpose(s), (A - C_alpha*np.ones(A.shape, dtype=float))), s)/A.size
    return co

def get_ideal_partitions(A):
    G = nx.from_numpy_array(A)
    N = A.shape[0]
    x = np.zeros(N, dtype = int)

    coms = get_coms_from_x(x)
    s = get_s_from_coms(coms)

    q = nx.algorithms.community.quality.modularity(G, coms)
    co = get_co(A, s)

    q_x = x.copy()
    co_x = x.copy()

    while not np.all(x == 1):
        x[-1] += 1
        i = N
        while i>0:
            i -= 1
            if x[i] == 2:
                x[i] = 0
                x[i-1] += 1

        coms = get_coms_from_x(x)
        s = get_s_from_coms(coms)

        q_temp = nx.algorithms.community.quality.modularity(G, coms)
        co_temp = get_co(A, s)

        if q_temp > q:
            q = q_temp
            q_x = x.copy()
        if co_temp > co:
            co = co_temp
            co_x = x.copy()
        
    return q_x, co_x

def get_permutation_distance(A, permutation):
    path_length = 0
    k = 0
    for j in permutation: 
        path_length += A[k][j] 
        k = j 
    path_length += A[k][0] 
    return path_length

def get_ideal_cycles(A):
    N = A.shape[0]
    min_path_length = maxsize

    for permutation in permutations(range(1, N)):
        if permutation[0] < permutation[-1]:
            path_length = get_permutation_distance(A, permutation)

            if path_length < min_path_length:
                min_path_length = path_length
                ideal_permutation = permutation

    ideal_cycle = [0] + list(ideal_permutation) + [0]

    return ideal_cycle
