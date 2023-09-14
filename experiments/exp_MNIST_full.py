import numpy as np
import matplotlib.pyplot as plt

# Import writer class from csv module
from csv import writer
import os
import glob
import sys
sys.path.append('../sim_soens')
sys.path.append('../')

from sim_soens.super_input import SuperInput
from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *
from sim_soens.soen_sim import network, synapse

from sim_soens.soen_utilities import (
    dend_load_arrays_thresholds_saturations, 
    index_finder
)

import time

def main():
    np.random.seed(10)
    # print(np.random.randint(0, 100, 10))


    def make_dataset(digits,samples,slowdown,duration):
        '''
        Creates rate coded spiking MNIST dataset
            - digits   = number of classes (different handwritten digits)
            - samples  = number of examples from each class
            - slowdown = factor by which to reduce rate encoding
            - duration = how long each sample should be (nanoseconds)
        '''
        dataset = [[] for _ in range(digits)]
        fig, axs = plt.subplots(digits,samples,figsize=(36,12))
        for i in range(digits):
            for j in range(samples):
                input_MNIST = SuperInput(
                    type='MNIST',
                    index=i,
                    sample=j,
                    slow_down=slowdown,
                    duration=duration
                    )
                spikes = input_MNIST.spike_arrays
                dataset[i].append([spikes[0],spikes[1]])

                axs[i][j].plot(spikes[1],spikes[0],'.k',ms=.5)
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])
        # saves dataset
        picklit(
            dataset,
            "datasets/MNIST/",
            f"duration={duration}_slowdown={slowdown}"
            )
        # plots dataset
        plt.show()

    def make_audio_dataset(patterns,replicas):
        '''
        For Heidelberg dataset only
        '''
        import tables
        file_path = f"datasets/Heidelberg/shd_train/shd_train.h5"
        fileh = tables.open_file(file_path, mode='r')
        units = fileh.root.spikes.units
        times = fileh.root.spikes.times
        labels = fileh.root.labels
        channels = np.max(np.concatenate(units))+1
        length = np.max(np.concatenate(times))
        classes = np.max(labels)

        names = ['ZERO','ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN',
                    'EIGHT','NINE','TEN','NULL','EINS','ZWEI','DREI','VIER',
                    'FUNF','SECHS','SEBEN','ACHT','NEUN','ZEHN']
        dataset = [[] for _ in range(patterns)]
        for i in range(patterns):
            result = np.where(np.array(labels)==i)[0]
            for j in range(replicas):
                dataset[i].append([units[result[j]],times[result[j]]*1000])

        picklit(
            dataset,
            f"datasets/Heidelberg/",
            f"digits={patterns}_samples={replicas}"
            )
        return dataset
    
    def plot_nodes(nodes,digit,sample,run):
        try:
            os.makedirs(path+name+'plots/')    
        except FileExistsError:
            pass
        for n,node in enumerate(nodes):
            lays = [[] for _ in range(len(node.dendrites))]
            phays = [[] for _ in range(len(node.dendrites))]
            for l,layer in enumerate(node.dendrites):
                for g,group in enumerate(layer):
                    for d,dend in enumerate(group):
                        lays[l].append(dend.s)
                        phays[l].append(dend.phi_r)
            # try:
            #     plt.style.use('seaborn-muted')  
            # except MatplotlibDeprecationWarning:
            #     plt.style.use("seaborn-v0_8")

            # plt.style.use('seaborn-muted')
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            plt.figure(figsize=(8,4))
            for l,lay in enumerate(lays):
                if l == 0:
                    lw = 4
                else:
                    lw = 2
                plt.plot(
                    np.mean(lay,axis=0),
                    linewidth=lw,
                    color=colors[l],
                    label=f'Layer {l} Mean Signal'
                    )
                plt.plot(
                    np.mean(phays[l],axis=0),
                    '--',
                    linewidth=.5,
                    color=colors[l],
                    # label=f'Layer {l} Mean Flux'
                    )
            plt.legend(loc='upper right')
            plt.title(f'Node {n} - Digit {digit} Sample {sample} - Run {run}')
            plt.savefig(path+name+f'plots/node_{n}_digit_{digit}_sample_{sample}_run_{run}.png')
            plt.close()

    def make_weights(size,exin,fixed):
        ones = np.ones(size)
        symm = 1

        if exin != None:
            # print(exin)
            symm = np.random.choice([-1,0,1], p=[exin[0]/100,exin[1]/100,exin[2]/100], size=size)

        if fixed is not None:
            # print("fixed")
            w = ones*fixed*symm
        else:
            w = np.random.rand(size)*symm
        return w
    
    def add_inhibition_counts(node):
    
        def recursive_downstream_inhibition_counter(dendrite,superdend):
            for out_name,out_dend in dendrite.outgoing_dendritic_connections.items():
                cs = out_dend.dendritic_connection_strengths[dendrite.name]
                if cs < 0:
                    superdend.downstream_inhibition += 1
                recursive_downstream_inhibition_counter(out_dend,superdend)

        for dendrite in node.dendrite_list:
            dendrite.downstream_inhibition = 0
            recursive_downstream_inhibition_counter(dendrite,dendrite)

    def max_s_finder(dendrite):
        d_params_ri = dend_load_arrays_thresholds_saturations('default_ri')
        ib_list = d_params_ri["ib__list"]
        s_max_plus__vec = d_params_ri["s_max_plus__vec"]
        _ind_ib = index_finder(ib_list[:],dendrite.ib) 
        return s_max_plus__vec[_ind_ib]

    def normalize_fanin(node):
        for dendrite in node.dendrite_list:
            if len(dendrite.dendritic_connection_strengths) > 0:
                print(dendrite.name)
                max_s = max_s_finder(dendrite) - dendrite.phi_th
                cs_list = []
                max_list = []
                influence = []
                for in_name,in_dend in dendrite.dendritic_inputs.items():
                    cs = dendrite.dendritic_connection_strengths[in_name]
                    max_in = max_s_finder(in_dend)
                    cs_list.append(cs)
                    max_list.append(max_in)
                    influence.append(cs*max_in)
                if sum(influence) > max_s:
                    norm_fact = sum(influence)/max_s
                    cs_normed = cs_list/norm_fact
                    for i,(in_name,cs) in enumerate(dendrite.dendritic_connection_strengths.items()):
                        dendrite.dendritic_connection_strengths[in_name] = cs_normed[i]

    def get_nodes(
            path,
            name,
            config,
            ):
        '''
        Either creates or loads nodes for training
        '''

        # importing os module
        place = path+name+'nodes/'
        if os.path.exists(place) == True:
            print("Loading nodes...")
            files = glob.glob(place+'*')
            latest = max(files, key=os.path.getctime)
            # print("latest",latest)
            file_name = latest[len(place):len(latest)-len('.pickle')]
            print("file name: ",file_name)
            nodes = picklin(place,file_name)

            new_nodes = False


        else:
            new_nodes=True


        if new_nodes == True:
            print("Making new nodes...")
            if config.inh_counter:
                print("Inhibition counter")  
            if config.norm_fanin:
                print("Fanin Normalization")  
            saved_run = 0
            start = time.perf_counter()
            np.random.seed(10)

            exin = config.exin
            fixed = config.fixed
            print(f" Excitatory, zeroed, inhibitory ratios: {exin}")
            print(f" Fixed uniform Jij (coupling strength) factor: {fixed}")


            # initialize a neuron of each class with this structure
            nodes = []
            for node in range(config.digits):
                # branching factor
                f_idx = 28

                if config.layers == 3:
                    layer_1_weighting = 1/4
                    layer_2_weighting = 3/4
                    exin = config.exin
                    fixed = config.fixed
                    # create random weights for each layer
                    layer_1 = [make_weights(f_idx,exin,fixed)*layer_1_weighting]
                    layer_2 = [make_weights(f_idx,exin,fixed)*layer_2_weighting for _ in range(f_idx)]

                    # place them in a weight structure (defines structure and weighing of a neuron)
                    weights = [
                        layer_1,
                        layer_2
                    ]

                elif config.layers == 2:
                    layer_1_weighting = 3/4

                    # create random weights for each layer
                    layer_1 = [np.random.rand(f_idx**2)*layer_1_weighting]
                    # layer_2 = [np.random.rand(f_idx)*layer_2_weighting for _ in range(f_idx)]

                    # place them in a weight structure (defines structure and weighing of a neuron)
                    weights = [
                        layer_1
                    ]

                elif config.layers == 4:
                    layer_1_weighting = 1/4
                    layer_2_weighting = 3/4

                    # create random weights for each layer
                    layer_1 = [np.random.rand(f_idx)*layer_1_weighting]
                    layer_2 = [np.random.rand(f_idx)*layer_2_weighting for _ in range(f_idx)]
                    layer_3 = [np.random.rand(1)*layer_2_weighting for _ in range(f_idx**2)]

                    # place them in a weight structure (defines structure and weighing of a neuron)
                    weights = [
                        layer_1,
                        layer_2,
                        layer_3
                    ]

                elif config.layers == 5:
                    # l1_weighting = 1/4
                    # l2_weighting = 3/4
                    # l3_weighting = 2

                    # # create random weights for each layer
                    # layer_1 = np.array([np.random.rand(f_idx)*l1_weighting])
                    # layer_2 = np.array([np.random.rand(2)*l2_weighting for _ in range(f_idx)])
                    # layer_3 = np.array([np.random.rand(int(f_idx/2))*l3_weighting for _ in range(f_idx*2)])

                    l1_weighting = 1/4
                    l2_weighting = 3/4
                    l3_weighting = 2

                    # create random weights for each layer
                    layer_1 = np.array([make_weights(f_idx,exin,fixed)*l1_weighting])
                    layer_2 = np.array([make_weights(2,exin,fixed)*l2_weighting for _ in range(f_idx)])
                    layer_3 = np.array([make_weights(int(f_idx/2),exin,fixed)*l3_weighting for _ in range(f_idx*2)])

                    print(layer_1.shape)
                    print(layer_2.shape)
                    print(layer_3.shape)

                    # place them in a weight structure (defines structure and weighing of a neuron)
                    weights = [
                        layer_1,
                        layer_2,
                        layer_3
                    ]


                # internal node parameters
                mutual_inhibition = True
                ib      = 1.8
                tau     = 50
                beta    = 2*np.pi*10**2
                s_th    = 0.5
                params = {
                    "ib"        :ib,
                    "ib_n"      :ib,
                    "ib_di"     :ib,
                    "tau"       :tau,
                    "tau_ni"    :tau,
                    "tau_di"    :tau,
                    "beta"      :beta,
                    "beta_ni"   :beta,
                    "beta_di"   :beta,
                    "s_th"      :s_th
                }
                    
                nodes.append(SuperNode(name=f'node_{node}',weights=weights,**params))
                # if node == 0:
                #     # nodes[node].plot_structure()


            if mutual_inhibition == True:
                inhibition = -(1/config.digits)
                for i,node in enumerate(nodes):
                    syn_soma = synapse(name=f'{node.name}_somatic_synapse')
                    node.synapse_list.append(syn_soma)
                    node.neuron.dend_soma.add_input(syn_soma,connection_strength=inhibition)
                for i,node in enumerate(nodes):
                    for other_node in nodes:
                        if other_node.name != node.name:
                            node.neuron.add_output(other_node.synapse_list[-1])
                            print(other_node.synapse_list[-1].name)

            if config.rand_flux is not None:
                print(f" Random flux factor: {config.rand_flux}")

                for n,node in enumerate(nodes):
                    for l,layer in enumerate(node.dendrites):
                        for g,group in enumerate(layer):
                            for d,dend in enumerate(group):
                                if 'ref' not in dend.name and 'soma' not in dend.name:
                                    sign = np.random.choice([-1,1], p=[.5,.5], size=1)[0]
                                    dend.offset_flux = np.random.rand()*config.rand_flux*sign

                    if config.inh_counter:
                        add_inhibition_counts(node)

                    if config.norm_fanin:
                        normalize_fanin(node)


            finish = time.perf_counter()
            print("Time to make neurons: ", finish-start)

            # save the nodes!
            picklit(
                nodes,
                f"{path}{name}/nodes/",
                f"init_nodes"
                )
        return nodes
    
    def make_update(nodes,config,digit,sample,errors):
        # print("Functional Update")
        s = time.perf_counter()
        offset_sums = [0 for _ in range(config.digits)]
        
        for n,node in enumerate(nodes):
            for l,layer in enumerate(node.dendrites):
                for g,group in enumerate(layer):
                    for d,dend in enumerate(group):
                        if 'ref' not in dend.name and 'soma' not in dend.name:
                            
                            if hasattr(dend, 'hebb'):
                                hebb = dend.hebb*20
                            else:
                                hebb = 1

                            step = errors[n]*np.mean(dend.s)*config.eta*hebb
                            flux = np.mean(dend.phi_r) + step
                            
                            if config.hebbian == "True":
                                for in_dend in dend.dendritic_inputs.keys():
                                    in_dendrite = dend.dendritic_inputs[in_dend]
                                    if "ref" not in in_dend:
                                        in_dendrite.hebb = np.mean(dend.s)
                                        # print(np.mean(dend.s))


                            if config.elasticity=="elastic":
                                if flux > 0.5 or flux < config.low_bound:
                                    step = -step
            
                            elif config.elasticity=="inelastic":
                                if flux > 0.5 or flux < config.low_bound:
                                    step = 0

                            if config.inh_counter:
                                if dend.downstream_inhibition%2!=0:
                                    step = -step

                            dend.offset_flux += step
                            offset_sums[n] += step

                        # else:
                        #     print("soma")

                        dend.s = []
                        dend.phi_r = []

        f = time.perf_counter()
        # print(f"Update time = {f-s}")
        return nodes, offset_sums

    
    def train_MNIST_neurons(nodes,dataset,path,name,config):
        '''
        Trains nodes on MNIST dataset
        '''
        if config.dataset=='MNIST':
            desired = [
                [5,0,0],
                [0,5,0],
                [0,0,5],
            ]
        elif config.dataset=='Heidelberg':
            desired = [
                [20,10,10],
                [10,20,10],
                [10,10,20],
            ]
        if config.digits != 3:
            desired = []
            for idx in range(config.digits):
                desired.append([0 for _ in range(config.digits)])
            for idx in range(config.digits):
                desired[idx][idx] = 5
            # print(desired)

        backend = 'julia'
        print('Backend: ', backend)

        # tracks ongoing timing costs
        # run_times = []
        # init_times = []

        # itereate over some number of epochs
        # for run in range(next_run,1):
        print("Run: ",config.run)

        # initialize epoch success count
        samples_passed=0

        # itereate over each sample
        for sample in range(config.samples):
            
            # count errors for this sample
            # total_errors = [[] for i in range(3)]

            # track outputs for this samples
            outputs = [[] for i in range(config.digits)]

            # iterate over each digit-class
            np.random.seed(None)
            shuffled = np.arange(0,config.digits,1)
            np.random.shuffle(shuffled)

            for digit in range(config.digits):
                digit = shuffled[digit]

                start = time.perf_counter()

                # create input opject for appropriate class and sample
                input_ = SuperInput(
                    type="defined",
                    channels=784,
                    defined_spikes=dataset[digit][sample]
                    )
                
                # attach same input to all neurons
                for node in nodes:
                    for i,channel in enumerate(input_.signals):
                        node.synapse_list[i].add_input(channel)

                # f0 = time.perf_counter()
                # print("Input time: ", f0-start)

                # run the network
                net = network(
                    name=config.name,
                    sim=True,
                    dt=.1,
                    tf=config.duration,
                    nodes=nodes,
                    backend=backend,
                    # print_times=True,
                    jul_threading=config.jul_threading
                    )
                
                # save one set of plots for all nodes for each digit of sample 0
                if config.plotting == 'sparse':
                    if sample == 0 and config.run%10==0:
                        plot_nodes(nodes,digit,sample,config.run)
                elif config.plotting == 'full':
                    plot_nodes(nodes,digit,sample,config.run)
                

                # keep track of run time costs
                # run_times.append(net.run_time)
                # init_times.append(net.init_time)

                # check spiking output
                spikes = array_to_rows(net.spikes,config.digits)

                # define error by difference of desired with actual spiking for each node
                errors = []
                for nd in range(config.digits):
                    errors.append(desired[nd][digit] - len(spikes[nd]))

                # output spike totals from each class
                output = []
                for nd in range(config.digits):
                    output.append(len(spikes[nd]))

                # track outputs associated with each class
                outputs[digit].append(output)

                # clear data
                for node in nodes:
                    node.neuron.spike_times                         = []
                    node.neuron.spike_indices                       = []
                    node.neuron.electroluminescence_cumulative_vec  = []
                    node.neuron.time_params                         = []

                s = time.perf_counter()
                
                offset_sums = [0 for _ in range(config.digits)]

                # on all but every tenth run, make updates according to algorithm 1 with elasticity
                if config.run%10 != 0 or config.run == 0:

                    if config.probabilistic == 1:
                        nodes, offset_sums = make_update(nodes,config,digit,sample,errors)
                        # # print("Determined update")
                        # if config.elasticity=="elastic":
                        #     if sample == 0 and config.run == 0: print("elastic")
                        #     for n,node in enumerate(nodes):
                        #         for l,layer in enumerate(node.dendrites):
                        #             for g,group in enumerate(layer):
                        #                 for d,dend in enumerate(group):
                        #                     if 'ref' not in dend.name and 'soma' not in dend.name:
                        #                         step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
                        #                         flux = np.mean(dend.phi_r) + step #dend.offset_flux
                        #                         if flux > 0.5 or flux < config.low_bound:
                        #                             step = -step
                        #                         dend.offset_flux += step
                        #                         offset_sums[n] += step
                        #                     dend.s = []
                        #                     dend.phi_r = []

                        # if config.elasticity=="inelastic":
                        #     if sample == 0 and config.run == 0: print("inealstic")
                        #     for n,node in enumerate(nodes):
                        #         for l,layer in enumerate(node.dendrites):
                        #             for g,group in enumerate(layer):
                        #                 for d,dend in enumerate(group):
                        #                     if 'ref' not in dend.name and 'soma' not in dend.name:
                        #                         step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
                        #                         flux = np.mean(dend.phi_r) + step #dend.offset_flux
                        #                         if flux > 0.5 or flux < config.low_bound:
                        #                             step = 0
                        #                         dend.offset_flux += step
                        #                         offset_sums[n] += step
                        #                     dend.s = []
                        #                     dend.phi_r = []

                        # if config.elasticity=="unbounded":
                        #     if sample == 0 and config.run == 0: print("unbounded")
                        #     for n,node in enumerate(nodes):
                        #         for l,layer in enumerate(node.dendrites):
                        #             for g,group in enumerate(layer):
                        #                 for d,dend in enumerate(group):
                        #                     if 'ref' not in dend.name and 'soma' not in dend.name:
                        #                         step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
                        #                         dend.offset_flux += step
                        #                         offset_sums[n] += step
                        #                     dend.s = []
                        #                     dend.phi_r = []
                    else:
                        # print("Probabilistic update")
                        bool_array = np.random.rand(len(nodes)*len(nodes[0].dendrite_list)) < config.probabilistic
                        dend_counter = 0
                        if config.elasticity=="elastic":
                            if sample == 0 and config.run == 0: print("elastic")
                            for n,node in enumerate(nodes):
                                for l,layer in enumerate(node.dendrites):
                                    for g,group in enumerate(layer):
                                        for d,dend in enumerate(group):
                                            print(bool_array[dend_counter])
                                            if bool_array[dend_counter] == True:
                                                if 'ref' not in dend.name and 'soma' not in dend.name:
                                                    step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
                                                    flux = np.mean(dend.phi_r) + step #dend.offset_flux
                                                    if flux > 0.5 or flux < config.low_bound:
                                                        step = -step
                                                    dend.offset_flux += step
                                                    offset_sums[n] += step
                                            dend.s = []
                                            dend.phi_r = []
                                            dend_counter += 1

                        if config.elasticity=="inelastic":
                            if sample == 0 and config.run == 0: print("inealstic")
                            for n,node in enumerate(nodes):
                                for l,layer in enumerate(node.dendrites):
                                    for g,group in enumerate(layer):
                                        for d,dend in enumerate(group):
                                            bool_array[dend_counter]
                                            if bool_array[dend_counter] == True:
                                                if 'ref' not in dend.name and 'soma' not in dend.name:
                                                    step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
                                                    flux = np.mean(dend.phi_r) + step #dend.offset_flux
                                                    if flux > 0.5 or flux < config.low_bound:
                                                        step = 0
                                                    dend.offset_flux += step
                                                    offset_sums[n] += step
                                            dend.s = []
                                            dend.phi_r = []
                                            dend_counter += 1

                        if config.elasticity=="unbounded":
                            if sample == 0 and config.run == 0: print("unbounded")
                            for n,node in enumerate(nodes):
                                for l,layer in enumerate(node.dendrites):
                                    for g,group in enumerate(layer):
                                        for d,dend in enumerate(group):
                                            bool_array[dend_counter]
                                            if bool_array[dend_counter] == True:
                                                if 'ref' not in dend.name and 'soma' not in dend.name:
                                                    step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
                                                    dend.offset_flux += step
                                                    offset_sums[n] += step #dend.offset_flux
                                            dend.s = []
                                            dend.phi_r = []
                                            dend_counter += 1

                # on the tenth run test, but don't update -- save full nodes with data
                else:
                    # print("Skipping Update")
                    if sample == 0 and config.run%50 == 0:
                        # save the nodes!
                        picklit(
                            nodes,
                            f"{path}{name}/full_nodes/",
                            f"full_0_{digit}_nodes_at_{config.run}"
                            )
                        for node in nodes:
                            for dend in node.dendrite_list:
                                dend.s = []
                                dend.phi_r = []

                f = time.perf_counter()
                # print("Update time: ", f-s)
                # print("Total runtime", f-start)
                for o,offset in enumerate(offset_sums):
                    offset_sums[o] = np.round(offset,2)

                print(f"  {sample}  -  [{digit} -> {np.argmax(output)}]  -  {np.round(f-start,1)}  -  {output} - {offset_sums} ")

                # CSV data
                List = [sample,digit,output,errors,np.argmax(output),f-start,net.init_time,net.run_time,offset_sums]
                with open(f'{path}{name}/learning_logger.csv', 'a') as f_object:
                    writer_object = writer(f_object)
                    writer_object.writerow(List)
                    f_object.close()

                # delete old objects
                del(net)
                del(input_)

                # check if sample was passed (correct prediction)
                # if np.argmax(output) == digit:
                #     samples_passed+=1

                # allow no ties
                sub = np.array(output) - output[digit] 
                if sum(n > 0 for n in sub) == 0 and sum(n == 0 for n in sub) == 1:
                    samples_passed+=1

        # samples passed out of total epoch
        print(f"samples passed: {samples_passed}/{config.digits*config.samples}\n\n")

        # save the nodes!
        picklit(
            nodes,
            f"{path}{name}/nodes/",
            f"eternal_nodes"
            )
        
        # if all samples passed, task complete!
        if samples_passed == config.digits*config.samples:
            print("converged!\n\n")
            picklit(
                nodes,
                f"{path}{name}/nodes/",
                f"CONVERGED_at_{config.run}"
                )

    from sim_soens.argparse import setup_argument_parser
    config = setup_argument_parser()
    # if config.jul_threading > 1:
    #     import os
    #     print("Multi-threading")
    #     os.system("$env:JULIA_NUM_THREADS=4")

    # call in previously generated dataset
    path    = 'results/MNIST/'
    name    = config.name+'/'
    if config.dataset=='MNIST':
        dataset = picklin("datasets/MNIST/","duration=5000_slowdown=100")
    elif config.dataset=='Heidelberg':
        print("Heidelberg dataset!")
        dataset = picklin("datasets/Heidelberg/",f"digits=3_samples=10")
        # dataset = make_audio_dataset(config.digits,config.samples)
    # new_nodes=True

    # load_start = time.perf_counter()

    if config.decay == "True":
        config.eta = np.max([1/(500+15*config.run),0.0001])

    nodes = get_nodes(path,name,config)
    # load_finish = time.perf_counter()
    # print("Load time: ", load_finish-load_start)
    print(
        config.name,
        " -- ",
        config.elasticity,
        " -- ",
        config.eta,
        " -- ",
        config.digits,
        " -- ",
        config.samples, 
        " -- ", 
        config.eta
        )
    train_MNIST_neurons(nodes,dataset,path,name,config)

if __name__=='__main__':
    main()
