import numpy as np
import matplotlib.pyplot as plt
from wakepy import keep
# Import writer class from csv module
from csv import writer
import os
import glob
import sys
sys.path.append('../sim_soens')
sys.path.append('../')

from sim_soens.super_input import SuperInput
# from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *
from sim_soens.soen_components import network, synapse
from sim_soens.soen_plotting import plot_MNIST_nodes
from sim_soens.argparse import setup_argument_parser

# from sim_soens.soen_utilities import (
# )
from sim_soens.neuron_library import MNISTNode
from sim_soens.super_algorithms import *
from sim_soens.input_library import *
import time
# import multiprocessing
import multiprocess as mp

def main():
    np.random.seed(10)

    def offset_readin(nodes,config):
        
        loaded_weights = picklin('./saved_data/',config.offset_transfer)
        for n,node in enumerate(nodes):
            for i,layer in enumerate(node.dendrites[1:]):
                for j,dens in enumerate(layer):
                    for k,d in enumerate(dens):
                        d.offset_flux = loaded_weights[n][i][j][k]
        return nodes
    
    def make_nodes(path,name,config):
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

            if config.weight_transfer is not None:
                lw = picklin('./saved_data/',config.weight_transfer)[node]

                if config.no_negative_jij == True:
                    print("Asymmetric weight transfer.")
                    for l,layer in enumerate(lw):
                        for g,group in enumerate(layer):
                            for d,dend in enumerate(group):
                                if dend < 0: lw[l][g][d] = 0

                params['loaded_weights'] = lw
                                        
            nodes.append(MNISTNode(**params))
            # if node == 0:
            #     nodes[0].plot_structure()
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
                        node.neuron.add_output(other_node.synapse_list[-1])
                        # print("-- ",other_node.synapse_list[-1].name)

        finish = time.perf_counter()
        print("Time to make neurons: ", finish-start)

        # save the nodes!
        picklit(
            nodes,
            f"{path}{name}/nodes/",
            f"init_nodes"
            )
        
        return nodes
    

    def make_single_node(n,return_dict,config):
        print(f"Making new node {n}...")

        # internal node parameters
        ib      = 1.8
        tau     = config.tau
        beta    = 2*np.pi*10**config.beta
        s_th    = config.s_th
        params = {
            "ib"          :ib,
            "ib_n"        :ib,
            "ib_di"       :ib,
            "tau"         :tau,
            "tau_ni"      :tau,
            "tau_di"      :tau,
            "beta"        :beta,
            "beta_ni"     :beta,
            "beta_di"     :beta,
            "s_th"        :s_th,
            "name"        :f'node_{n}',
            "fan_coeff"   :config.fan_coeff
        }
        if config.weight_transfer is not None:
            lw = picklin('./saved_data/',config.weight_transfer)[n]

            if config.no_negative_jij == True:
                print("Asymmetric weight transfer.")
                for l,layer in enumerate(lw):
                    for g,group in enumerate(layer):
                        for d,dend in enumerate(group):
                            if dend < 0: lw[l][g][d] = 0

            params['loaded_weights'] = lw

        params.update(config.__dict__)
        node = MNISTNode(**params)
        return_dict[node.name] = node


    def get_nodes(
            path,
            name,
            config,
            ):
        '''
        Either creates or loads nodes for training
        '''
        s1 = time.perf_counter()
        # importing os module
        place = path+name+'nodes/'
        if os.path.exists(place) == True:
            # print("Loading nodes...")
            files = glob.glob(place+'*')
            print("\n",config.exp_name)
            latest = max(files, key=os.path.getctime)
            # print("latest",latest)
            file_name = latest[len(place):len(latest)-len('.pickle')]
            # print("file name: ",file_name)
            nodes = picklin(place,file_name)

        else:
            if config.multi == True:
                manager = mp.Manager()
                return_dict = manager.dict()
                return_dict = manager.dict()
                thrds = []
                for thrd in range(config.digits):
                    thrds.append(
                        mp.Process(
                            target=make_single_node, 
                            args=(thrd,return_dict,config)
                            )
                        )

                for thrd in thrds:
                    thrd.start()

                for thrd in thrds:
                    thrd.join()

                nodes = []
                for i in range(config.digits):
                    print(f'Adding node_{i}')
                    nodes.append(return_dict[f'node_{i}'])

                mutual_inhibition = True
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
                                # print("-- ",other_node.synapse_list[-1].name)]
                
                if config.offset_transfer is not None:
                    nodes = offset_readin(nodes,config)

            else:
                make_nodes(path,name,config)
                if config.offset_transfer is not None:
                    nodes = offset_readin(nodes,config)
            

            

            # save the nodes!
            picklit(
                nodes,
                f"{path}{name}/nodes/",
                f"init_nodes"
                )
            
        s2 = time.perf_counter()

        print(f"Total node acquisition time: {s2-s1}")



        return nodes


    def train_MNIST_neurons(nodes,dataset,path,name,config):
        '''
        Trains nodes on MNIST dataset
        '''
        pass_arr = np.zeros(config.digits)

        # if 'unbounded' == config.exp_name:
        #     desired = [
        #         [30,10,10],
        #         [10,30,10],
        #         [10,10,30],
        #     ]

        # if 'unbounded_fan' == config.exp_name:
        #     desired = [
        #         [60,40,40],
        #         [40,60,40],
        #         [40,40,60],
        #     ]

        # elif config.dataset=='Heidelberg':
        #     desired = [
        #         [30,10,10],
        #         [10,30,10],
        #         [10,10,30],
        #     ]

        # else:

        desired = []
        for idx in range(config.digits):
            desired.append([config.off_target for _ in range(config.digits)])

        for idx in range(config.digits):
            desired[idx][idx] = config.target

        if config.run ==0: print(desired)

        if config.tiling == True:
            idx_groups,idx_list = tile()


        # backend = 'julia'
        # print('Backend: ', config.backend)

        # tracks ongoing timing costs
        # run_times = []
        # init_times = []

        # itereate over some number of epochs
        # for run in range(next_run,1):
        print("Run: ",config.run)

        # initialize epoch success count
        samples_passed=0
        mod = config.samples*config.digits
        # itereate over each sample
        sample = config.run%50

        for sample in range(sample,sample+1):
            

            # track outputs for this samples
            outputs = [[] for i in range(config.digits)]

            # iterate over each digit-class
            np.random.seed(None)
            shuffled = np.arange(0,config.digits,1)
            np.random.shuffle(shuffled)
            mhs = np.zeros(config.digits)

            for digit in range(config.digits):
                digit = shuffled[digit]
                start_1 = time.perf_counter()

                if config.dataset!='keras':
                    # create input opject for appropriate class and sample
                    input_ = SuperInput(
                        type="defined",
                        channels=784,
                        defined_spikes=dataset[digit][sample]
                        )
                    
                else:
                    input_ = SuperInput(
                        type="defined",
                        channels=784,
                        defined_spikes=[[],[]]
                        )
                    
                    (X_train, y_train), (X_test, y_test) = dataset
                    X = (X_train[(y_train == digit)][sample]).reshape(784)
                    X_max = np.max(X)
                    x = (X/X_max)*.5
                    for i,x in enumerate(X):
                        for node in nodes:
                            node.dendrite_list[::-1][i].offset_flux = x


                # attach same input to all neurons
                for node in nodes:
                    
                    if config.tiling == True:                        
                        for i,syn in enumerate(node.synapse_list[:784]):
                            # print(f"Tiling -- synapse {i} <-- {idx_list[i]} channel")
                            syn.add_input(input_.signals[idx_list[i]])

                    else:
                        if config.layers == 42:
                            # print("Special Inputs")
                            for start in range(4):
                                node.doubled_input(input_,start=start*784*2)
                        elif config.double_dends == True:
                            # print("Doubled Input")
                            node.doubled_input(input_)
                        else:
                            for i,channel in enumerate(input_.signals):
                                node.synapse_list[i].add_input(channel)
                            if config.extended_arbor==True:
                                for i,channel in enumerate(input_.signals):
                                    node.synapse_list[i+784].add_input(channel)

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
                

                record_penultimate_signals=False
                if record_penultimate_signals==True:
                    penultimte_signals = []
                    for itr, node in enumerate(nodes):
                        signal_last = []           
                        for dend in node.dendrites[1][0]:
                            signal_last.append(np.mean(dend.s))
                        # append them to the confusion matrix
                        penultimte_signals.append(signal_last)
                    
                    with open(f'{path}{name}/penultimate.csv', 'a') as f_object:
                        writer_object = writer(f_object)
                        writer_object.writerow([nodes[0].run,digit,penultimte_signals])
                        f_object.close()
                    
                
                # save one set of plots for all nodes for each digit of sample 0
                if config.plotting == 'sparse':
                    if sample == 0 and config.run%mod==0:
                        plot_MNIST_nodes(nodes,digit,sample,config.run,name,path)
                elif config.plotting == 'full':
                    plot_MNIST_nodes(nodes,digit,sample,config.run,name,path)
                # if config.exp_name=='double_dends_slim':
                #     plot_MNIST_nodes(nodes,digit,sample,config.run,name,path)
                

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
                    for syn in node.synapse_list:
                        syn.phi_spd = []
                    

                s = time.perf_counter()
                
                offset_sums = [0 for _ in range(config.digits)]

                if ((config.exp_name=='double_dends_slim_nonrand_lim' or
                    config.exp_name=='double_dends_slim_nonrand_chooser' or
                    config.exp_name=='disynaptic_extened_fanin_4') 
                    and digit==0 and (config.run==1 or sample%500==0 or config.run==1)):
                    picklit(
                        nodes,
                        f"{path}{name}/full_nodes_prime/",
                        f"full_{sample}_{digit}_nodes_at_{config.run}"
                        )
                    
                # on all but every tenth run, make updates according to algorithm 1 with elasticity
                if config.run%mod != 0 or config.run == 0:

                    if config.probabilistic == 1:
                        nodes, offset_sums, max_hits = arbor_update(nodes,config,digit,sample,errors,config.updater)

                    else:
                        # print("Probabilistic update")
                        nodes, offset_sums, max_hits = probablistic_arbor_update(nodes,config,digit,sample,errors,config.updater)
                    # mhs[digit] = max_hits
                        
                    for node in nodes:
                        for dend in node.dendrite_list:
                            dend.s = []
                            dend.phi_r = []
                        
                # on the modth run test, but don't update -- save full nodes with data   
                else:
                    max_hits = np.zeros(config.digits)
                    # print("Skipping Update")
                    if (sample == 0 and (config.run%1000 == 0 or config.run==1)):
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

                print(f"  {sample}  -  [{digit} -> {np.argmax(output)}]  -  {np.round(f-start_1,1)}  -  {output} - {errors} - {max_hits}")#- {offset_sums} - {max_hits}")
                del(net)
                del(input_)

                # CSV data
                # List = [sample,digit,output,errors,np.argmax(output),f-start,net.init_time,net.run_time,offset_sums]
                # with open(f'{path}{name}/learning_logger.csv', 'a') as f_object:
                #     writer_object = writer(f_object)
                #     writer_object.writerow(List)
                #     f_object.close()

                # check if sample was passed (correct prediction)
                # if np.argmax(output) == digit:
                #     samples_passed+=1

                # allow no ties
                sub = np.array(output) - output[digit] 
                if sum(n > 0 for n in sub) == 0 and sum(n == 0 for n in sub) == 1:
                    samples_passed+=1
                    pass_arr[digit] += 1



        if hasattr(nodes[0],'seen'):
            # nodes[0].seen = ((config.run % 50))*10 + 10
            nodes[0].seen = ((config.run % config.samples))*config.digits + config.digits
        else:
            # nodes[0].seen = 10
            nodes[0].seen = config.digits

        if hasattr(nodes[0],'passed'):
            nodes[0].passed += samples_passed
        else:
            nodes[0].passed = samples_passed

        if hasattr(nodes[0],'all_passed'):
            nodes[0].all_passed += pass_arr
        else:
            nodes[0].all_passed = pass_arr

        acc = np.round(nodes[0].passed*100/(nodes[0].seen),2)
        accs = np.round(100*nodes[0].all_passed/(nodes[0].seen/config.digits),2)
        # samples passed out of total epoch
        # if 'full' not in config.exp_name: 
        #     print(f" samples passed: {samples_passed}/{config.digits*config.samples}\n")
        # else:
        print(f" samples passed: {samples_passed}/{config.digits} -- running epoch accuracy: {acc}%")
        # print(f" digit performance {accs}%\n")


        # if all samples passed, task complete!
        if nodes[0].seen == config.digits*config.samples:
            print("acc check")
            with open(f'{path}{name}/performance_log.csv', 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow([acc,accs])
                f_object.close()

            if nodes[0].passed == config.digits*config.samples:
                print("converged!\n\n")
                picklit(
                    nodes,
                    f"{path}{name}/nodes/",
                    f"CONVERGED_at_{config.run}"
                    )
            else:
                nodes[0].passed = 0
                nodes[0].seen = 0
                nodes[0].all_passed = np.zeros(config.digits)
        

        # save the nodes!
        picklit(
            nodes,
            f"{path}{name}/nodes/",
            f"eternal_nodes"
            )



    config = setup_argument_parser()
    if not hasattr(config,'updater'): config.updater = "classic"

    exin_name = 'excit'
    if config.exin is not None:
        exin_name = 'inhib'
    if "arbor_sweep" in config.exp_name:
        config.exp_name = f'arbor_sweep_{config.s_th}_{config.tau}_{config.fan_coeff}_{config.target}_{config.rand_flux}_{config.max_offset}_{exin_name}'
        file = f"results/MNIST/{config.exp_name}/performance_log.csv"
        if os.path.isfile(file) == True:
            import pandas as pd
            df = pd.read_csv(file,names=['total_perf','by_dig_perf'])
            # print(len(df['total_perf']))
            if len(df) == 30:
                print((f"Config {config.exp_name} alreadry run!"))
                sys.exit()
            else:
                pass
            

    # if config.exin != [0,0,100]:
    #     # print("Inhibition")
    #     config.inh_counter=True
    # call in previously generated dataset
            

    if config.alternode is not None:
        alt_name = config.alternode
        config = picklin(f"results\\MNIST\\{alt_name}\\","config.pickle")
                
    path    = 'results/MNIST/'
    name    = config.exp_name+'/'
    if config.dataset=='MNIST':
        dataset = picklin("datasets/MNIST/","duration=5000_slowdown=100")
    elif config.dataset=='Heidelberg':
        print("Heidelberg dataset!")
        dataset = picklin("datasets/Heidelberg/",f"digits=3_samples=10")
        # dataset = make_audio_dataset(config.digits,config.samples)
    elif config.dataset=='keras':
        from keras.datasets import mnist
        dataset = mnist.load_data()

        # config.exp_name = config.alternode
    # load_start = time.perf_counter()

    if config.decay == "True":
        decay = np.ceil((config.run+1)/50)
        # print("decay = ",decay)
        config.eta = np.max([1/(250+15*decay),0.00001])
        # config.eta = 0.003389830508474576
    if config.run==0:
        for i,(k,v) in enumerate(config.__dict__.items()):
            print(f"{i}  --  {k}      {v}")

    nodes = get_nodes(path,name,config)

    # if config.exp_name =='weight_transfer_inh_counting':
    #     for n,node in enumerate(nodes):
    #         node.add_inhibition_counts()


    # for  i, node in enumerate(nodes):
    #     print(f"\nNode {i}")  
    #     print(node.neuron.dend_soma.dendritic_connection_strengths)
    
    if config.alternode is not None and config.run==0:
        nodes[0].run = 0
    elif hasattr(nodes[0],'run'):
        nodes[0].run +=1
        config.run = nodes[0].run
    # elif config.name == 'thresh_full_rerun':
    #     print("restart")
    #     nodes[0].run = 0
    else:
        nodes[0].run = config.run


    # if config.exp_name == "thresh_full_rerun":
    #     nodes[0].run = 0
    #     config.run = nodes[0].run

    # if config.exp_name == "updates_cobuff":
    #     picklit(config,path+config.exp_name+'/',"config")
    
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

    with keep.running() as k:
        train_MNIST_neurons(nodes,dataset,path,name,config)

if __name__=='__main__':
    main()
