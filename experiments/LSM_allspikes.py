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
from sim_soens.soen_components import *
from sim_soens.argparse import setup_argument_parser

import json
import time
import multiprocessing



'''
Plan
 - Reservoir of N neurons
 - Readout layer of C=num_classes neurons
 - Input dimensionality of C nuerons = N
 - Input dimensionality of N neurons = some grid size (convolutions?)
 - N neurons connected to eachothers' somas directly, arbor reserved for input
 - Arbor update rule only on C neurons
'''

def make_res_node(n,config):
    # print(f"Making node {n}")

    ib      = config['nodes_ib']
    tau     = config['nodes_tau']
    beta    = config['nodes_beta']
    s_th    = config['nodes_s_th']

    res_params = {
                    "ib"        :ib,
                    "ib_n"      :ib,
                    "ib_di"     :ib,
                    "tau"       :tau,
                    "tau_ni"    :tau,
                    "tau_di"    :tau,
                    "beta"      :beta,
                    "beta_ni"   :beta,
                    "beta_di"   :beta,
                    "s_th"      :s_th,
                }

    node = SuperNode(
        name = f'res_neuron_{n}',
        weights = [
            [np.random.rand(2)],
            [np.random.rand(2) for _ in range(2)],
            [np.random.rand(2) for _ in range(4)],
        ],
        **res_params
    )
    node.normalize_fanin(2.25)
    return node

def make_parallel_neurons(n1,n2,return_dict,config):

    nodes = {(f'res_neuron_{n}',make_res_node(n,config)) for n in range(n1,n2)}
    return_dict.update(nodes)

def make_single_readout(c,return_dict,N,C,config):

    ib      = config['codes_ib']
    tau     = config['codes_tau']
    beta    = config['codes_beta']
    s_th    = config['codes_s_th']


    readout_params = {
                    "ib"        :ib,
                    "ib_n"      :ib,
                    "ib_di"     :ib,
                    "tau"       :tau,
                    "tau_ni"    :tau,
                    "tau_di"    :tau,
                    "beta"      :beta,
                    "beta_ni"   :beta,
                    "beta_di"   :beta,
                    "s_th"      :s_th,
                }


    if N == 98:
        code = SuperNode(
            name = f'code_neuron_{c}',
            weights = [
                [np.random.rand(7)],
                [np.random.rand(7) for _ in range(7)],
                [np.random.rand(2) for _ in range(49)],
            ],
            **readout_params
        )
        code.normalize_fanin(config['fan_coeff_codes'])

    return_dict[code.name] = code

def make_readouts(N,C,config):

    ib      = config['codes_ib']
    tau     = config['codes_tau']
    beta    = config['codes_beta']
    s_th    = config['codes_s_th']


    readout_params = {
                    "ib"        :ib,
                    "ib_n"      :ib,
                    "ib_di"     :ib,
                    "tau"       :tau,
                    "tau_ni"    :tau,
                    "tau_di"    :tau,
                    "beta"      :beta,
                    "beta_ni"   :beta,
                    "beta_di"   :beta,
                    "s_th"      :s_th,
                }

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
                ],
                **readout_params
            )
            code.normalize_fanin(config['fan_coeff_codes'])
            codes.append(code)

        elif N == 98:
            code = SuperNode(
                name = f'code_neuron_{c}',
                weights = [
                    [np.random.rand(7)],
                    [np.random.rand(7) for _ in range(7)],
                    [np.random.rand(2) for _ in range(49)],
                ],
                **readout_params
            )
            code.normalize_fanin(config['fan_coeff_codes'])
            codes.append(code)

        elif N == 10:
            code = SuperNode(
                name = f'code_neuron_{c}',
                weights = [
                    [np.random.rand(3)],
                    [np.random.rand(4) for _ in range(3)],
                ],
                **readout_params
            )
            code.normalize_fanin(config['fan_coeff_codes'])
            codes.append(code)


    return codes


def make_neurons(N,C,config):
    
    ib      = config['nodes_ib']
    tau     = config['nodes_tau']
    beta    = config['nodes_beta']
    s_th    = config['nodes_s_th']

    res_params = {
                    "ib"        :ib,
                    "ib_n"      :ib,
                    "ib_di"     :ib,
                    "tau"       :tau,
                    "tau_ni"    :tau,
                    "tau_di"    :tau,
                    "beta"      :beta,
                    "beta_ni"   :beta,
                    "beta_di"   :beta,
                    "s_th"      :s_th,
                }

    nodes = []
    for n in range(N):
        node = SuperNode(
            name = f'res_neuron_{n}',
            weights = [
                [np.random.rand(2)],
                [np.random.rand(2) for _ in range(2)],
                [np.random.rand(2) for _ in range(4)],
            ],
            **res_params
        )
        node.normalize_fanin(config['fan_coeff_nodes'])
        nodes.append(node)


    ib      = config['codes_ib']
    tau     = config['codes_tau']
    beta    = config['codes_beta']
    s_th    = config['codes_s_th']


    readout_params = {
                    "ib"        :ib,
                    "ib_n"      :ib,
                    "ib_di"     :ib,
                    "tau"       :tau,
                    "tau_ni"    :tau,
                    "tau_di"    :tau,
                    "beta"      :beta,
                    "beta_ni"   :beta,
                    "beta_di"   :beta,
                    "s_th"      :s_th,
                }


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
                ],
                **readout_params
            )
            code.normalize_fanin(config['fan_coeff_codes'])
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
            code.normalize_fanin(config['fan_coeff_codes'])
            codes.append(code)

        elif N == 10:
            code = SuperNode(
                name = f'code_neuron_{c}',
                weights = [
                    [np.random.rand(3)],
                    [np.random.rand(4) for _ in range(3)],
                ]
            )
            code.normalize_fanin(config['fan_coeff_codes'])
            codes.append(code)


    return nodes, codes


def connect_nodes(nodes,p_connect,conn_coeff):
    for i,N1 in enumerate(nodes):
        for j,N2 in enumerate(nodes):
            if np.random.rand() < p_connect and i!=j:
                syn = synapse(
                    name=f'{N2.name}_synsoma_from_n{i}'
                    )
                N2.synapse_list.append(syn)
                N2.neuron.dend_soma.add_input(syn,connection_strength=np.random.rand()*conn_coeff)
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

# def run_network(all_nodes):
#     net = network(
#         sim     =True,
#         tf      = 500,
#         nodes   = all_nodes,
#         backend = 'julia',
#         dt=1.0
#     )

#     return net

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

#%%

def run_MNIST(nodes,codes,config):


    digits   = config['digits']
    samples  = config['samples']
    eta      = config['eta']
    duration = config['duration']
    runs     = config['runs']
    exp_name = config['exp_name']
    N        = config['N']

    dataset = picklin("datasets/MNIST/","duration=5000_slowdown=100")

    res_spikes = {}
    for i in range(digits):
        res_spikes[str(i)]=[]
    accs = []
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


        print(f"RUN: {run} -- EXP: {exp_name}")
        successes = 0
        seen = 0
        
        for sample in range(samples):
            print(f"  ------------------------------------------------ ")
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
                    
                    # net = run_network(nodes)
                    net = network(
                        sim     =True,
                        tf      = duration,
                        nodes   = nodes,
                        backend = 'julia',
                        dt=1.0
                    )

                    # if digit==0 and sample == 0:
                    #     # move to within class
                    #     mid_plot = plt
                    #     mid_plot.figure(figsize=(12,4))
                        
                    #     for node in nodes:
                    #         mid_plot.plot(net.t,node.neuron.dend_soma.s)
                    #         mid_plot.plot(net.t,node.neuron.dend_soma.phi_r,'--',linewidth=1)
                    #     # plt.legend()
                    #     mid_plot.xlabel("Time(ns)",fontsize=16)
                    #     mid_plot.ylabel("Signal",fontsize=16)
                    #     mid_plot.title("Network Node Dynamics",fontsize=18)
                    #     mid_plot.show()

                    spikes = net.spikes
                    axs[digit][sample].plot(spikes[1], spikes[0], '.k')
                    # axs[digit][sample].axhline(y = N-.5, color = 'b', linestyle = '-') 
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
                    with open(f'{path}/config.txt', 'w') as file:
                        file.write(json.dumps(config))

                inpt = SuperInput(
                    type="defined",
                    channels=config['N'],
                    defined_spikes=res_spikes[str(digit)][sample]
                    )

                for c,code in enumerate(codes):
                    for s,signal in enumerate(inpt.signals):
                        code.synapse_list[s].add_input(signal)

                c_net = network(
                    sim     =True,
                    tf      = duration,
                    nodes   = codes,
                    backend = 'julia',
                    dt=1.0
                )
                    

                success,outputs,prediction = check_success(codes,digit)

                targets = np.zeros(digits)
                targets[digit] = 15 # --> 10/15

                codes, update_sums = make_updates(codes,targets,eta)

                print(
                    f"   {digit} --> {prediction} :: {outputs} :: {np.round(update_sums,2)} :: {np.round(c_net.run_time,2)} :: {len(res_spikes[str(digit)][sample][0])}"
                    )   

                seen      += 1
                successes += success
                if seen == 30:
                    ep_acc = np.round(successes*100/seen,2)
                    accs.append(ep_acc)
                    print(f"  Epoch accuracy = {ep_acc}%\n")
                    if successes == 30:
                        print("Converged!")
                        all_nodes = nodes + codes
                        picklit(
                            accs,
                            f"results/res_MNIST/{exp_name}/",
                            f"CONVERGED"
                            )
                        picklit(
                            codes,
                            f"results/res_MNIST/{exp_name}/",
                            f"converged_nodes"
                            )
                        return nodes,codes
                nodes,codes = cleanup(c_net,nodes,codes)


        if run == 0:
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(f"{path}/raster_all.png")
            plt.close()

    picklit(
        accs,
        f"results/res_MNIST/{exp_name}/",
        f"performance"
        )
    return nodes, codes, accs


def run_all(config):
    s1=time.perf_counter()

    N = config['N']
    C = config['C']
    
    eta = config["eta"]
    tau_c = config["codes_tau"]
    tau_n = config["nodes_tau"]
    sth_n = config["nodes_s_th"]
    sth_c = config["codes_s_th"]
    fans_n = config["fan_coeff_nodes"]
    fans_c = config["fan_coeff_codes"]
    den = config["density"]
    conn = config["res_connect_coeff"]

    print(f"res_eta_tau_c_tau_n_sth_n_sth_c_fans_n_fans_c_den_conn")
    config['exp_name'] = f"res_{eta}_{tau_c}_{tau_n}_{sth_n}_{sth_c}_{fans_n}_{fans_c}_{den}_{conn}"

    print(config['exp_name'])

    config['path'] = f"results/res_MNIST/{config['exp_name']}"


    np.random.seed(config['seed'])



    ###################################
    #     Making Reservoir Neurons    #
    ###################################
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    thrds = []
    for thrd in range(14):
        thrds.append(
            multiprocessing.Process(
                target=make_parallel_neurons, 
                args=(thrd*7,thrd*7+7,return_dict,config)
                )
                )

    for thrd in thrds:
        thrd.start()

    for thrd in thrds:
        thrd.join()

    nodes = []
    for i in range(98):
        nodes.append(return_dict[f'res_neuron_{i}'])

    s2 = time.perf_counter()
    print(f"Time to make nodes: {np.round(s2-s1,2)}")

    ###################################
    #      Making Readout Neurons     #
    ###################################

    return_dict = manager.dict()
    thrds = []
    for thrd in range(C):
        thrds.append(
            multiprocessing.Process(
                target=make_single_readout, 
                args=(thrd,return_dict,N,C,config)
                )
                )

    for thrd in thrds:
        thrd.start()

    for thrd in thrds:
        thrd.join()

    codes = []
    for i in range(C):
        codes.append(return_dict[f'code_neuron_{i}'])

    # codes = make_readouts(N,C,config)
    # nodes, codes = make_neurons(N,C,config)
    
    s3 = time.perf_counter()
    print(f"Time to make codes: {np.round(s3-s2,2)}")

    ###################################
    #      Connecting Reservoir       #
    ###################################

    nodes = connect_nodes(nodes,config['density'],config['res_connect_coeff'])

    s4 = time.perf_counter()
    print(f"Time to connect nodes: {np.round(s4-s3,2)}")

    # nodes, codes = nodes_to_codes(nodes,codes)

    ###################################
    #       Running Simulation        #
    ###################################

    nodes,codes,accs = run_MNIST(nodes,codes,config)

    s5 = time.perf_counter()
    print(f"Time to run epoch: {np.round(s5-s4,2)}")


if __name__ == "__main__":

    
    config = setup_argument_parser()
    run_all(config.__dict__)

    # etas     = [0.005]
    # tau_cs   = [250,500]
    # tau_ns   = [50,100]

    # sth_ns   = [0.1,.25]
    # sth_cs   = [0.1,.25]

    # fans_ns  = [2.25,2.5]
    # fans_cs  = [1.50,1.75]

    # dens     = [0.1,0.25,.5]
    # conns    = [0.05,0.1,0.15]

    # all_configs = []
    # for eta in etas:
    #     for tau_c in tau_cs:
    #         for tau_n in tau_ns:
    #             for sth_n in sth_ns:
    #                 for sth_c in sth_cs:
    #                     for fans_n in fans_ns:
    #                         for fans_c in fans_cs:
    #                             for den in dens:
    #                                 for conn in conns:
    #                                     config = {

    #                                         'N'                     : 98,
    #                                         'C'                     : 3,
    #                                         'seed'                  : 442,
    #                                         'eta'                   : eta,
    #                                         'digits'                : 3,
    #                                         'samples'               : 10,
    #                                         'duration'              : 100,
    #                                         'runs'                  : 25,

    #                                         'nodes_tau'             : tau_n,
    #                                         'nodes_beta'            : 2*np.pi*10**3,
    #                                         'nodes_ib'              : 1.8,
    #                                         'nodes_s_th'            : sth_n,
    #                                         'fan_coeff_nodes'       : fans_n,

    #                                         'codes_tau'             : tau_c,
    #                                         'codes_beta'            : 2*np.pi*10**3,
    #                                         'codes_ib'              : 1.8,
    #                                         'codes_s_th'            : sth_c,
    #                                         'fan_coeff_codes'       : fans_c,

    #                                         'density'               : den,
    #                                         'res_connect_coeff'     : conn,

    #                                     }

    #                                     N = config['N']
    #                                     C = config['C']
                                        
    #                                     eta = config["eta"]
    #                                     tau_c = config["codes_tau"]
    #                                     tau_n = config["nodes_tau"]
    #                                     sth_n = config["nodes_s_th"]
    #                                     sth_c = config["codes_s_th"]
    #                                     fans_n = config["fan_coeff_nodes"]
    #                                     fans_c = config["fan_coeff_codes"]
    #                                     den = config["density"]
    #                                     conn = config["res_connect_coeff"]

    #                                     config['exp_name'] = f"res_{eta}_{tau_c}_{tau_n}_{sth_n}_{sth_c}_{fans_n}_{fans_c}_{den}_{conn}"


    #                                     # run_all(config)
    #                                     all_configs.append(config)

    # print(len(all_configs))
    # import os
    # path = 'results/res_MNIST/'
    # for i,config in enumerate(all_configs):
    #     exp_path = f"{path}{config['exp_name']}"
    #     # print("Already Run:")
    #     try:
    #         acc = picklin(exp_path,"performance")
    #         # print(f"  {config['exp_name']}")
    #     except:
    #         print("Running Now:")
    #         print(f"  {config['exp_name']} -- file number {i}")
    #         run_all(config)





    #-------------------------------------------------------
    # config = {

    #     'N'                     : 100,
    #     'C'                     : 3,
    #     'seed'                  : np.random.randint(1000),
    #     'eta'                   : 0.005,
    #     'digits'                : 3,
    #     'samples'               : 10,
    #     'duration'              : 100,

    #     'nodes_tau'             : 50,
    #     'nodes_beta'            : 2*np.pi*10**3,
    #     'nodes_ib'              : 1.8,
    #     'nodes_s_th'            : 0.1,
    #     'fan_coeff_nodes'       : 2.25,

    #     'codes_tau'             : 250,
    #     'codes_beta'            : 2*np.pi*10**3,
    #     'codes_ib'              : 1.8,
    #     'codes_s_th'            : 0.1,
    #     'fan_coeff_codes'       : 1.5,

    #     'density'               : .1,
    #     'res_connect_coeff'     : 0.1,

    # }