import numpy as np
import time

from sim_soens.soen_initialize import (
    dendrite_drive_construct,
    rate_array_attachment,
    synapse_initialization,
    output_synapse_initialization,
    transmitter_initialization,
    dendrite_data_attachment
)
from .soen_py_stepper import *
from .soen_numba_stepper import *


def run_soen_sim(net):
    '''
    Runs SOEN simulation (either via the python or julia backend)
    '''
                
    # network
    if type(net).__name__ == 'network':

        # convert to dimensionless time
        time_vec = np.arange(0,net.tf+net.dt,net.dt)  
        ttc = 1e-9/net.jj_params['tau_0']

        net.time_params = {
            'dt'               : net.dt, 
            'tf'               : net.tf, 
            'time_vec'         : time_vec, 
            't_tau_conversion' : ttc,
            'tau_vec'          : time_vec*ttc,
            'd_tau'            : net.dt*ttc,
            }

        # Julia backend
        if net.backend == 'julia':
            net = run_julia_backend(net)

        # Python backend
        else:
            net = run_python_backend(net)

    # formerly, there were unique sim methods for each element
    else:
        print('''
        Error: Simulations no longer supported for individual components.
                --> Instead run component in the context of a network
        ''')

    return net

def run_python_backend(net):
    '''
    The originial method of simulation (needs updating)
        - Initializes component parameters with soen_intialize
        - Runs the python backend from soen_py_stepper
    '''
    # print('Running Python Backend')
    start = time.perf_counter()
    # interate through all network nodes and initialize all related elements
    for node in net.nodes:
        # if net.print_times: print("Initializing neuron: ", neuron.name)
        node.neuron.time_params = net.time_params
        node.neuron.dend_soma.threshold_flag = False

        for dend in node.dendrite_list:
            # if net.print_times: print(" Initializing dendrite: ", dend.name)
            dend.ind_phi = []  # temp
            dend.ind_s = [] # temp
            dend.spk_print = True # temp

            dendrite_drive_construct(
                dend,
                net.time_params['tau_vec'],
                net.time_params['t_tau_conversion'],
                net.time_params['d_tau']
                )

            rate_array_attachment(dend)
            synapse_initialization(
                dend,
                net.time_params['tau_vec'],
                net.time_params['t_tau_conversion']
                )

        output_synapse_initialization(
            node.neuron,net.time_params['tau_vec'],
            net.time_params['t_tau_conversion']
            )
        
        transmitter_initialization(
            node.neuron,net.time_params['t_tau_conversion']
            )

    finish = time.perf_counter()
    if net.print_times: print(f"Initialization procedure run time: {finish-start}")
    net.init_time = finish-start

    start = time.perf_counter()
    
    if net.backend == 'numba':
        net = numba_net_step(net,net.time_params['tau_vec'],net.time_params['d_tau'])
    else:
        net = net_step(net,net.time_params['tau_vec'],net.time_params['d_tau'])

    finish = time.perf_counter()
    if net.print_times: print(f"Py stepper time: {finish-start}")
    net.run_time = finish-start

    # attach results to dendrite objects
    for node in net.nodes:
        for dend in node.dendrite_list:
            dend.phi_vec = dend.phi_r__vec[:]
            dendrite_data_attachment(dend,net)
    
    return net

def run_julia_backend(net):
    '''
    The originial method of simulation (needs updating)
        - Initializes component parameters with soen_intialize
        - Runs the python backend from soen_jul_stepper
    '''
    # print('Running Julia Backend')
    
    from sim_soens.soen_initialize import make_subarrays, find_shoulders
    
    start = time.perf_counter()

    # interate through all network nodes and initialize all related elements
    net.phi_vec, net.s_array, net.r_array = make_subarrays(net.nodes[0].neuron.ib,'ri')

    # for special indexing trick at update
    neg_idx,pos_idx,neg_min,pos_min = find_shoulders(net.phi_vec)

    net.phi_vals = {
        "neg_idx":int(neg_idx),
        "pos_idx":int(pos_idx),
        "neg_min":neg_min,
        "pos_min":pos_min
    }

    for node in net.nodes:
        # print("Initializing neuron: ", neuron.name)
        node.neuron.time_params = net.time_params
        node.neuron.dend_soma.threshold_flag = False

        for dend in node.dendrite_list:
            dend.beta = dend.circuit_betas[-1]
            synapse_initialization(
                dend,
                net.time_params['tau_vec'],
                net.time_params['t_tau_conversion']
                )

        output_synapse_initialization(
            node.neuron,
            net.time_params['tau_vec'],
            net.time_params['t_tau_conversion']
            )
        
        transmitter_initialization(
            node.neuron,
            net.time_params['t_tau_conversion']
            )

    finish = time.perf_counter()
    if net.print_times: print(f"Initialization procedure run time: {finish-start}")

    net.init_time = finish-start

    distributed = False
    if distributed == False:
        # print("Thread path")
        start = time.perf_counter()            

        import os
        os.environ["JULIA_NUM_THREADS"] = str(net.jul_threading)
        # string = f"$env:JULIA_NUM_THREADS={net.jul_threading}"
        # os.system("$env:JULIA_NUM_THREADS=8")
        # os.system('echo "Hello out there"')

        from julia import Main as jl

        # jl.using("Distributed")
        # jl.addprocs(2)
        # print("Threads:", jl.Threads.nthreads())


        jl.include("../sim_soens/julia_conversion.jl")

        if jl.Threads.nthreads() == 1:
            jl.include("../sim_soens/julia_thread_stepper.jl")
        else:
            jl.include("../sim_soens/julia_thread_stepper.jl")

        jul_net = jl.obj_to_structs(net)

        finish = time.perf_counter()
        if net.print_times: print(f"Julia setup time: {finish-start}")


        start = time.perf_counter()
        jl.stepper(jul_net)
        finish = time.perf_counter()

        if net.print_times: print(f"Julia stepper time: {finish-start}")

        net.run_time = finish-start  

        start = time.perf_counter()
        for node in net.nodes:
            for i,dend in enumerate(node.dendrite_list):
                jul_dend = jul_net["nodes"][node.name]["dendrites"][dend.name]
                dend.s     = jul_dend.s #[:-1]
                dend.phi_r = jul_dend.phir #[:-1]

                dend.ind_phi = jul_dend.ind_phi #[:-1]
                dend.ind_s = jul_dend.ind_s #[:-1]
                dend.phi_vec = jul_dend.phi_vec #[:-1]

                if "soma" in dend.name:
                    spike_times = (
                        (jul_dend.out_spikes-1)
                        * net.dt 
                        * net.time_params['t_tau_conversion']
                        )
                    dend.spike_times        = spike_times
                    node.neuron.spike_times = spike_times
                # # if net.print_times: print(sum(jul_net[node.name][i].s))/net.dt
            for i,syn in enumerate(node.synapse_list):
                jul_syn = jul_net["nodes"][node.name]["synapses"][syn.name]
                syn.phi_spd = jul_syn.phi_spd
        finish = time.perf_counter()
        if net.print_times: print(f"jul-to-py re-attachment time: {finish-start}")

        jul_net = jl.clear_all(jul_net)
        jl.unbindvariables()
    return net