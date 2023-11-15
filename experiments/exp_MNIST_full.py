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
from sim_soens.soen_components import network, synapse

from sim_soens.soen_utilities import (
    dend_load_arrays_thresholds_saturations, 
    index_finder
)
from sim_soens.neuron_library import MNISTNode
from sim_soens.super_algorithms import *
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
            # print("Loading nodes...")
            files = glob.glob(place+'*')
            latest = max(files, key=os.path.getctime)
            # print("latest",latest)
            file_name = latest[len(place):len(latest)-len('.pickle')]
            # print("file name: ",file_name)
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


                # internal node parameters
                mutual_inhibition = True
                ib      = 1.8
                tau     = config.tau
                beta    = 2*np.pi*10**config.beta
                s_th    = config.s_th
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
                    "s_th"      :s_th,
                    "name"      :f'node_{node}'
                }
                params.update(config.__dict__)
                nodes.append(MNISTNode(**params))
                    
                # nodes.append(SuperNode(name=f'node_{node}',weights=weights,**params))
                print("Ref: ",nodes[0].neuron.tau_ref,nodes[0].neuron.ib_ref)
                # if node == 0:
                #     nodes[node].plot_structure()


            if mutual_inhibition == True:
                inhibition = -(1/config.digits)
                for i,node in enumerate(nodes):
                    syn_soma = synapse(name=f'{node.name}_somatic_synapse')
                    node.synapse_list.append(syn_soma)
                    node.neuron.dend_soma.add_input(syn_soma,connection_strength=inhibition)
                for i,node in enumerate(nodes):
                    for other_node in nodes:
                        if other_node.name != node.name:
                            node.neuron.add_output(other_node.synapse_list[-1],out_node_name=other_node.name)
                            print("-- ",other_node.synapse_list[-1].name)

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
        if 'unbounded' == config.exp_name:
            desired = [
                [30,10,10],
                [10,30,10],
                [10,10,30],
            ]

        if 'unbounded_fan' == config.exp_name:
            desired = [
                [60,40,40],
                [40,60,40],
                [40,40,60],
            ]

        elif config.dataset=='Heidelberg':
            desired = [
                [30,10,10],
                [10,30,10],
                [10,10,30],
            ]
        else:
            desired = []
            for idx in range(config.digits):
                desired.append([0 for _ in range(config.digits)])
            target = 10
            # if 'long' in config.exp_name: target=10
            for idx in range(config.digits):
                desired[idx][idx] = config.target

            if config.run ==1: print(desired)

        # backend = 'julia'
        print('Backend: ', config.backend)

        # tracks ongoing timing costs
        # run_times = []
        # init_times = []

        # itereate over some number of epochs
        # for run in range(next_run,1):
        print("Run: ",config.run)

        # initialize epoch success count
        samples_passed=0
        if 'full' in config.exp_name:
            mod = 500
        else:
            mod = 10
        # itereate over each sample
        sample = config.run%50
        for sample in range(sample,sample+1):
            
            # count errors for this sample
            # total_errors = [[] for i in range(3)]

            # track outputs for this samples
            outputs = [[] for i in range(config.digits)]

            # iterate over each digit-class
            np.random.seed(10)
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
                    name=config.exp_name,
                    sim=True,
                    dt=config.dt,
                    tf=config.duration,
                    nodes=nodes,
                    backend=config.backend,
                    # print_times=True,
                    jul_threading=config.jul_threading
                    )
            
                
                # save one set of plots for all nodes for each digit of sample 0
                if config.plotting == 'sparse':
                    if sample == 0 and config.run%mod==0:
                        plot_nodes(nodes,digit,sample,config.run)
                elif config.plotting == 'full':
                    plot_nodes(nodes,digit,sample,config.run)
                # if "fanin" in config.exp_name and sample == 9:
                #     plot_nodes(nodes,digit,sample,config.run)
                

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
                if config.run%mod != 0 or config.run == 0:

                    if config.probabilistic == 1:
                        nodes, offset_sums = arbor_update(nodes,config,digit,sample,errors)

                    else:
                        # print("Probabilistic update")
                        nodes, offset_sums = probablistic_arbor_update(nodes,config,digit,sample,errors)

                # on the tenth run test, but don't update -- save full nodes with data
                else:
                    # print("Skipping Update")
                    if (sample == 0 and config.run%50 == 0):
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


        if hasattr(nodes[0],'seen'):
            nodes[0].seen = ((config.run % 50))*10 + 10
        else:
            nodes[0].seen = 10

        if hasattr(nodes[0],'passed'):
            nodes[0].passed += samples_passed
        else:
            nodes[0].passed = samples_passed


        # samples passed out of total epoch
        if 'full' not in config.exp_name: 
            print(f" samples passed: {samples_passed}/{config.digits*config.samples}\n")
        else:
            print(f" samples passed: {samples_passed}/{config.digits} -- running epoch accuracy: {np.round(nodes[0].passed*100/(nodes[0].seen),2)}%\n")

        
        # if all samples passed, task complete!
        # if samples_passed == config.digits*config.samples:
         
        if samples_passed == config.digits*config.samples:
            print("converged!\n\n")
            picklit(
                nodes,
                f"{path}{name}/nodes/",
                f"CONVERGED_at_{config.run}"
                )

        if 'full' in config.exp_name and nodes[0].seen == config.digits*config.samples:

            if nodes[0].passed == nodes[0].seen:
                print("converged!\n\n")
                picklit(
                    nodes,
                    f"{path}{name}/nodes/",
                    f"CONVERGED_at_{config.run}"
                    )
            else:
                nodes[0].passed = 0
                nodes[0].seen = 0

        if (config.run+1) % 50 == 0:
            nodes[0].passed = 0
            nodes[0].seen = 0

        # save the nodes!
        picklit(
            nodes,
            f"{path}{name}/nodes/",
            f"eternal_nodes"
            )

    from sim_soens.argparse import setup_argument_parser
    config = setup_argument_parser()

    # call in previously generated dataset
    path    = 'results/MNIST/'
    name    = config.exp_name+'/'
    if config.dataset=='MNIST':
        dataset = picklin("datasets/MNIST/","duration=5000_slowdown=100")
    elif config.dataset=='Heidelberg':
        print("Heidelberg dataset!")
        dataset = picklin("datasets/Heidelberg/",f"digits=3_samples=10")
        # dataset = make_audio_dataset(config.digits,config.samples)

    # load_start = time.perf_counter()

    if config.decay == "True":
        decay = np.ceil((config.run+1)/50)
        # print("decay = ",decay)
        config.eta = np.max([1/(250+15*decay),0.00001])
        # config.eta = 0.003389830508474576

    nodes = get_nodes(path,name,config)

    # inhibition = -(1/config.digits)
    # # for i,node in enumerate(nodes):
    # #     syn_soma = synapse(name=f'{node.name}_somatic_synapse')
    # #     node.synapse_list.append(syn_soma)
    #     # node.neuron.dend_soma.add_input(syn_soma,connection_strength=inhibition)

    
    # for i,node in enumerate(nodes):
    #     node.neuron.dend_soma.firing_targets = {}

    # for i,node in enumerate(nodes):
    #     for other_node in nodes:
    #         if other_node.name != node.name:
    #             out_node_name=other_node.name
    #             soma = node.neuron.dend_soma
    #             if out_node_name not in list(soma.firing_targets.keys()):
    #                 soma.firing_targets[out_node_name] = []
    #             soma.firing_targets[out_node_name].append(other_node.synapse_list[-1].name)

    # load_finish = time.perf_counter()
    # print("Load time: ", load_finish-load_start)
    if config.run%50 == 0: print(
        config.exp_name,
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
    
    if config.run == 1:
        import json
        with open(f'{path}/{name}/config.txt', 'w') as convert_file:
            convert_file.write(json.dumps(config.__dict__))

    train_MNIST_neurons(nodes,dataset,path,name,config)

if __name__=='__main__':
    main()
