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
    '''
    For making a single reservoir node of index n
    '''

    # parameters with which the node will be initialized
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
    
    # Create a super node object with an exposed dendritied layer of 8
    node = SuperNode(
        name = f'res_neuron_{n}',
        weights = [
            [np.random.rand(2)],
            [np.random.rand(2) for _ in range(2)],
            [np.random.rand(2) for _ in range(4)],
        ],
        **res_params
    )

    # Normalize the dendritic arbor fanin and weight updward with fanin coeff
    node.normalize_fanin(config['fan_coeff_nodes'])
    return node

def make_parallel_neurons(n1,n2,return_dict,config):
    '''
    This implementation allows pythonin multiprocess for node initializations
    '''
    nodes = {(f'res_neuron_{n}',make_res_node(n,config)) for n in range(n1,n2)}
    return_dict.update(nodes)

def make_single_readout(c,return_dict,N,C,config):
    '''
    Makes a single readout node (called a code here)
    '''

    # init params
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

    # match the fanin to the size of the reservoir from which the code reads
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

    if N == 490:
        code = SuperNode(
            name = f'code_neuron_{c}',
            weights = [
                [np.random.rand(7)],
                [np.random.rand(7) for _ in range(7)],
                [np.random.rand(2) for _ in range(49)],
                [np.random.rand(5) for _ in range(98)],
            ],
            **readout_params
        )
        code.normalize_fanin(config['fan_coeff_codes'])

    return_dict[code.name] = code


def make_readouts(N,C,config):
    '''
    For making readout nodes without parallelization
    '''
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
    '''
    For making reservoir nodes without parallelization
    '''
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
    '''
    Connect any two nodes in the reservoir with probability p_connect
        - Strength of connection is a random number in [0,1], weighted by conn_coeff
        - Connection pairing included in synaptic name
    '''
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
    '''
    Each reservoir node feeds into one synapse of every readout node
    '''
    for n,node in enumerate(nodes):
        for c,code in enumerate(codes):
            node.neuron.add_output(code.synapse_list[n])
    
    return nodes, codes


def connect_input(inp,nodes,method='random_windows'):
    '''
    Connect input object to reservoir
        - random_windows     -> connect in same order but in random spots
        - synapse_sequential -> connect input in order to reservoir, repeat if needed
    '''
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
                    syn.add_input(inp.signals[count%784])
                    count+=1
        # print("Input dimensions met: ", count)
    
    return nodes


def check_success(codes,digit):
    '''
    Collect output spikes from each readout neuron
        - Determine prediction based on code that spiked most
        - Check against correct class
    '''
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
    '''
    For updated the readout layer for better classifcation
        - Take error of outputs and targets at each readout neuron
        - Apply the arbor update rule
        - Bound by some maximal flux_offset size on either end
        - Record updated made
    '''
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

def neuromodulate(nodes,codes,targets,eta):
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

        if targets[c] > 0:
            for node in nodes:
                for dend in code.dendrite_list:
                    if 'ref' not in dend.name:
                        step = error*np.mean(dend.s)*eta
                        old_offset = dend.offset_flux
                        if dend.offset_flux < 0:
                            dend.offset_flux = np.max([dend.offset_flux+step,dend.phi_th])
                        else:
                            dend.offset_flux = np.min([dend.offset_flux+step,dend.phi_th])
                        update_sums[c]+=dend.offset_flux-old_offset    


    return nodes,codes,update_sums

def cleanup(net,nodes,codes):
    '''
    Cleanup nodes for re-use
    '''
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


def run_MNIST(nodes,codes,config):
    '''
    Runs the reservoirs and readouts together on the first pass
        - Thereafter, uses saved reservoir states to train only the readout
    '''
    # params
    digits   = config['digits']
    samples  = config['samples']
    eta      = config['eta']
    duration = config['duration']
    runs     = config['runs']
    exp_name = config['exp_name']
    N        = config['N']

    # Saved spiking MNIST data
    dataset = picklin("datasets/MNIST/","duration=5000_slowdown=100")

    # Prepare a dictionary for collected reservoir spikes for each sample/class
    res_spikes = {}
    for i in range(digits):
        res_spikes[str(i)]=[]
    accs = []

    # Iterate over some number of runs (reservoirs only run on the first pass)
    for run in range(runs):

        # On first run, prepare the reservoir response plot
        if run == 0:
            path = config['path']
            try:
                os.makedirs(path)    
            except FileExistsError:
                pass
            fig, axs = plt.subplots(digits,samples,figsize=(36,12))
            # plt.title(f"MNIST Reservoir + Readout Activity",fontsize=18)
            # plt.xlabel("Time (ns)")
            # plt.ylabel("Neuron Index (Readouts Neurons Above Line)")


        print(f"RUN: {run} -- EXP: {exp_name}")
        successes = 0 # start counting correct predictions
        seen = 0      # and total seen samples
        
        # itereate over all samples
        for sample in range(samples):
            print(f"  ------------------------------------------------ ")

            # iterate over each class
            for digit in range(digits):
                
                # if reservoir data is passed in via the config dict, don't resimulate the reservoir
                if 'res_spikes' not in config.keys() or config['neuromod']==True:

                    # create spiking MNIST input from saved spikes
                    inp = SuperInput(
                        type="defined",
                        channels=784,
                        defined_spikes=dataset[digit][sample]
                        )

                    # connect the input to reservoir nodes
                    nodes = connect_input(inp,nodes,method='synapse_sequential')
                    
                    # run the network --best with julia backend and large time step (for beta^3)
                    net = network(
                        sim     = True,
                        tf      = duration,
                        nodes   = nodes,
                        backend = 'julia',
                        dt=1.0
                    )

                    spks_current = [
                        net.spikes[0].astype(int).tolist(),
                        net.spikes[1].tolist()]
                    
                    # Collect reservoir spikes
                    res_spikes[str(digit)].append(spks_current
                        )

                    if run == 1:
                        spikes = net.spikes
                        axs[digit][sample].plot(spikes[1], spikes[0], '.k')
                        # axs[digit][sample].axhline(y = N-.5, color = 'b', linestyle = '-') 
                        axs[digit][sample].set_xticks([])
                        axs[digit][sample].set_yticks([])
                        # save the spikes
                        picklit(
                            res_spikes,
                            f"{path}/",
                            f"res_spikes"
                            )
                        
                        # save the config
                        with open(f'{path}/config.txt', 'w') as file:
                            file.write(json.dumps(config))

                        # add spikes to config (so that this block will be skipped in future)
                        config['res_spikes'] = res_spikes
                    del(net) # clear the net
                
                else:
                    res_spikes = config['res_spikes']

                if config['neuromod'] == True:
                    trial_spikes = spks_current
                else:
                    trial_spikes = res_spikes[str(digit)][sample]
                # Use the reservoir spikes for this digit and sample to create an input object
                inpt = SuperInput(
                    type="defined",
                    channels=config['N'],
                    defined_spikes=trial_spikes
                    )

                # Add input to readout neurons (Each neuron gets the whole res as input)
                for c,code in enumerate(codes):
                    for s,signal in enumerate(inpt.signals):
                        code.synapse_list[s].add_input(signal)

                # Run the readout simulation
                c_net = network(
                    sim     =True,
                    tf      = duration,
                    nodes   = codes,
                    backend = 'julia',
                    dt=1.0
                )
                    
                # Count successful predictions
                success,outputs,prediction = check_success(codes,digit)

                # Targeted output for each readout neuron given the digit
                targets = np.zeros(digits)
                targets[digit] = 15 # --> 10/15

                # Make the arbor update
                if config['neuromod'] == True:
                    # print('Neurmodulation')
                    nodes, codes, update_sums = neuromodulate(nodes,codes,targets,eta)
                else:
                    codes, update_sums = make_updates(codes,targets,eta)

                # Print out what has transpired
                print(
                    f"   {digit} --> {prediction} :: {outputs} :: {np.round(update_sums,2)} :: {np.round(c_net.run_time,2)} :: {len(res_spikes[str(digit)][sample][0])}"
                    )   

                # If all 30 samples are correctly classified, save the readout neurons and return
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
                            f"{path}/",
                            f"CONVERGED"
                            )
                        picklit(
                            codes,
                            f"{path}/",
                            f"converged_nodes"
                            )
                        return nodes,codes
                nodes,codes = cleanup(c_net,nodes,codes)

        # Save the multiplot of reservoir spikes
        if run == 0:
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(f"{path}/raster_all.png")
            plt.close()

    # Save the performance history of the readout training
    picklit(
        accs,
        f"{path}/",
        f"performance"
        )
    return nodes, codes, accs


def run_all(config):
    '''
    Creates or loads nodes and codes, runs experiment given config, can evolve successful exps
    '''
    s1=time.perf_counter()

    # params
    N       = config['N']
    C       = config['C']
    eta     = config["eta"]
    tau_c   = config["codes_tau"]
    tau_n   = config["nodes_tau"]
    sth_n   = config["nodes_s_th"]
    sth_c   = config["codes_s_th"]
    fans_n  = config["fan_coeff_nodes"]
    fans_c  = config["fan_coeff_codes"]
    den     = config["density"]
    conn    = config["res_connect_coeff"]

    # for reference
    print(f"res_eta_tau_c_tau_n_sth_n_sth_c_fans_n_fans_c_den_conn")

    # if experiment name not passed in through config (which means it is being run for the first time)
    if 'exp_name' not in config.keys() or config['exp_name'] == 'test':
        config['exp_name'] = f"res_N{N}_{eta}_{tau_c}_{tau_n}_{sth_n}_{sth_c}_{fans_n}_{fans_c}_{den}_{conn}"

    print(config['exp_name'])

    # Add path if not already passed in through config
    if 'path' not in config.keys():
        config['path'] = f"results/res_MNIST/{config['exp_name']}"

    # Be consistent with seeding
    np.random.seed(config['seed'])



    ###################################
    #     Making Reservoir Neurons    #
    ###################################
    # Mutliprocessing tools allowing for return of information from multiple threads
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # if creating a new reservoir, use multiprocessing
    nodes = []
    if 'res_spikes' not in config.keys():
        thrds = []

        # use appropriate amount of threads given the number of neurons in reservoir
        # each thread takes some chunk of total
        if config['N'] == 98:
            for thrd in range(14):
                thrds.append(
                    multiprocessing.Process(
                        target=make_parallel_neurons, 
                        args=(thrd*7,thrd*7+7,return_dict,config)
                        )
                )
        elif config['N'] == 490:
            thrd_cnt = 14
            intrvl   = 35
            for thrd in range(thrd_cnt):
                thrds.append(
                    multiprocessing.Process(
                        target=make_parallel_neurons, 
                        args=(thrd*intrvl,thrd*intrvl+intrvl,return_dict,config)
                        )
                )
        # start them
        for thrd in thrds:
            thrd.start()
        # join them
        for thrd in thrds:
            thrd.join()

        # create an ordered list of reservoir nodes from list
        for i in range(config['N']):
            nodes.append(return_dict[f'res_neuron_{i}'])

    s2 = time.perf_counter()
    print(f"Time to make nodes: {np.round(s2-s1,2)}")

    ###################################
    #      Making Readout Neurons     #
    ###################################

    # Apply same procedure to readout neurons
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

    s3 = time.perf_counter()
    print(f"Time to make codes: {np.round(s3-s2,2)}")

    ###################################
    #      Connecting Reservoir       #
    ###################################

    # If it is a new reservoir, make internal connections
    if 'res_spikes' not in config.keys():
        nodes = connect_nodes(nodes,config['density'],config['res_connect_coeff'])

    s4 = time.perf_counter()
    print(f"Time to connect nodes: {np.round(s4-s3,2)}")

    # nodes, codes = nodes_to_codes(nodes,codes)

    ###################################
    #       Running Simulation        #
    ###################################

    # Run the MNIST experiment
    nodes,codes,accs = run_MNIST(nodes,codes,config)

    s5 = time.perf_counter()
    print(f"Time to run epoch: {np.round(s5-s4,2)}")


def evolve(path,N):
    '''
    Takes the top ten percent of reservoir/readout configurations and reruns them for a greater
    number of iterations.
    '''
    import os
    if os.path.exists(path) == True:
        dir_list = os.listdir(path)
        accs = {}
        for directory in dir_list:
            if str(N) in directory:
                try:
                    exp_path = f"{path}{directory}"
                    acc = picklin(exp_path,"performance")
                    accs[directory] = np.max(acc)
                except:
                    print(f"Dir {directory} has not been run.")
    print(f"\n")
    perf_rankings = dict(
        reversed(list({k: v for k, v in sorted(accs.items(), key=lambda item: item[1])}.items()))
        )
    for i,(k,v) in enumerate(perf_rankings.items()):
        if float(i) < len(perf_rankings)*.1:
            print("HERE")
            print("Evolution candidate: ",k,v)
            exp_path = f"{path}{k}"
            print(f"{path}{k}/performance.pickle")
            if (not os.path.isfile(f"{path}{k}/evolved/performance.pickle")
                and os.path.isfile(f"{path}{k}/performance.pickle")==True):
                print(f"{k} has not undergone an evolution.")
                try:
                    res_spikes = picklin(exp_path,"res_spikes")
                except:
                    print(f"Dir {k} is not ready to evolve.")
                try:

                    config = json.load(open(f"{path}{k}/config_dict.json"))
                    # with open(f"{path}{k}/config.txt") as f:

                    #     print(text)
                    #     config = dict(eval())
                    #     print(config)

                    config['path'] = f"results/res_MNIST/{k}/evolved"
                    config['runs'] = 100
                    config['res_spikes'] = res_spikes
                    print(f"Config {k} ready to evolve.")
                    return config
                except:
                    print(f"Dir {k} has no config file.")

                print(f"Running {k}")
                
                

if __name__ == "__main__":

    # Collect command line arguments as the configuration
    config = setup_argument_parser()

    # The evolutionary path
    if config.evolve==True:
        print("Evolve")
        path = 'results/res_MNIST/'
        config = evolve(path,config.N)
        run_all(config)

    # The traditional path
    else:
        config  = config.__dict__
        N       = config['N']
        C       = config['C']
        eta     = config["eta"]
        tau_c   = config["codes_tau"]
        tau_n   = config["nodes_tau"]
        sth_n   = config["nodes_s_th"]
        sth_c   = config["codes_s_th"]
        fans_n  = config["fan_coeff_nodes"]
        fans_c  = config["fan_coeff_codes"]
        den     = config["density"]
        conn    = config["res_connect_coeff"]

        if config['exp_name'] == "test":
            exp_name = f"res_{eta}_{tau_c}_{tau_n}_{sth_n}_{sth_c}_{fans_n}_{fans_c}_{den}_{conn}"
            path = f"results/res_MNIST/{exp_name}"

            alt_name = f"res_{eta}_{tau_c}_{tau_n}_{sth_n}_{sth_c}_{fans_n}_{fans_c}_{den}_{conn}"
            alt_path = f"results/res_MNIST/{alt_name}"
        else:
            exp_name = config['exp_name']
            path = alt_path = f"results/res_MNIST/{exp_name}"
        
        

        # if this configuration has not been run before, run it
        if (not os.path.isfile(f"{path}/performance.pickle") and 
            not os.path.isfile(f"{alt_path}/performance.pickle")):
            print(f"Running {exp_name}")
            run_all(config)

        # otherwise move on to the next config
        else:
            print(f"{exp_name} already run.")

