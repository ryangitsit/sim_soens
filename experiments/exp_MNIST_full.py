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
            saved_run = 0
            start = time.perf_counter()
            np.random.seed(10)

            # branching factor
            f_idx = 28

            if config.layers == 3:
                layer_1_weighting = 1/4
                layer_2_weighting = 3/4

                # create random weights for each layer
                layer_1 = [np.random.rand(f_idx)*layer_1_weighting]
                layer_2 = [np.random.rand(f_idx)*layer_2_weighting for _ in range(f_idx)]

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
                l1_weighting = 1/4
                l2_weighting = 3/4
                l3_weighting = 2

                # create random weights for each layer
                layer_1 = np.array([np.random.rand(f_idx)*l1_weighting])
                layer_2 = np.array([np.random.rand(2)*l2_weighting for _ in range(f_idx)])
                layer_3 = np.array([np.random.rand(int(f_idx/2))*l3_weighting for _ in range(f_idx*2)])
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
            

            # initialize a neuron of each class with this structure
            nodes = []
            for node in range(config.digits):
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

            finish = time.perf_counter()
            print("Time to make neurons: ", finish-start)

            # save the nodes!
            picklit(
                nodes,
                f"{path}{name}/nodes/",
                f"init_nodes"
                )
        return nodes
    
    def train_MNIST_neurons(nodes,dataset,path,name,config):
        '''
        Trains nodes on MNIST dataset
        '''
        desired = [
            [3,0,0],
            [0,3,0],
            [0,0,3],
        ]
        if config.digits > 3:
            desired = []
            for idx in range(config.digits):
                desired.append([0 for _ in range(config.digits)])
                desired[idx][idx] = 5

        backend = 'julia'
        print(backend)

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

            # track outputs for this sample
            outputs = [[] for i in range(config.digits)]

            # iterate over each digit-class
            for digit in range(config.digits):
                
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
                    sim=True,
                    dt=.1,
                    tf=250,
                    nodes=nodes,
                    backend=backend,
                    print_times=True
                    )
                
                # save one set of plots for all nodes for each digit of sample 0
                if sample == 0 and config.run%10==0:
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
                        plt.style.use('seaborn-v0_8-muted')
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
                        plt.title(f'Node {n} - Digit {digit} - Run {config.run}')
                        plt.savefig(path+name+f'plots/node_{n}_digit_{digit}_run_{config.run}.png')
                        plt.close()

                

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
                if config.run%10 != 0 and config.run != 0:

                    if config.probabilistic == 1:
                        # print("Determined update")
                        if config.elasticity=="elastic":
                            if sample == 0 and config.run == 0: print("elastic")
                            for n,node in enumerate(nodes):
                                for l,layer in enumerate(node.dendrites):
                                    for g,group in enumerate(layer):
                                        for d,dend in enumerate(group):
                                            if 'ref' not in dend.name and 'soma' not in dend.name:
                                                step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
                                                flux = np.mean(dend.phi_r) + step #dend.offset_flux
                                                if flux > 0.5 or flux < -0.5:
                                                    step = -step
                                                dend.offset_flux += step
                                                offset_sums[n] += dend.offset_flux
                                            dend.s = []

                        if config.elasticity=="inelastic":
                            if sample == 0 and config.run == 0: print("inealstic")
                            for n,node in enumerate(nodes):
                                for l,layer in enumerate(node.dendrites):
                                    for g,group in enumerate(layer):
                                        for d,dend in enumerate(group):
                                            if 'ref' not in dend.name and 'soma' not in dend.name:
                                                step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
                                                flux = np.mean(dend.phi_r) + step #dend.offset_flux
                                                if flux > 0.5 or flux < -0.5:
                                                    step = 0
                                                dend.offset_flux += step
                                                offset_sums[n] += dend.offset_flux
                                            dend.s = []
                                            dend.phi_r = []

                        if config.elasticity=="unbounded":
                            if sample == 0 and config.run == 0: print("unbounded")
                            for n,node in enumerate(nodes):
                                for l,layer in enumerate(node.dendrites):
                                    for g,group in enumerate(layer):
                                        for d,dend in enumerate(group):
                                            if 'ref' not in dend.name and 'soma' not in dend.name:
                                                step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
                                                dend.offset_flux += step
                                                offset_sums[n] += dend.offset_flux
                                            dend.s = []
                                            dend.phi_r = []
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
                                                    if flux > 0.5 or flux < -0.5:
                                                        step = -step
                                                    dend.offset_flux += step
                                                    offset_sums[n] += dend.offset_flux
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
                                                    if flux > 0.5 or flux < -0.5:
                                                        step = 0
                                                    dend.offset_flux += step
                                                    offset_sums[n] += dend.offset_flux
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
                                                    offset_sums[n] += dend.offset_flux
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

    # call in previously generated dataset
    path    = 'results/MNIST/'
    name    = config.name+'/'
    dataset = picklin("datasets/MNIST/","duration=5000_slowdown=100")
    # new_nodes=True

    # load_start = time.perf_counter()

    if config.decay == "True":
        config.eta = 1/(1000+15*config.run)

    nodes = get_nodes(path,name,config)
    # load_finish = time.perf_counter()
    # print("Load time: ", load_finish-load_start)
    print(config.name," -- ",config.elasticity," -- ",config.eta," -- ",config.digits," -- ",config.samples, " -- ", config.eta)
    train_MNIST_neurons(nodes,dataset,path,name,config)

if __name__=='__main__':
    main()
