import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.super_node import SuperNode
from sim_soens.super_input import SuperInput
from sim_soens.soen_components import network
from sim_soens.soen_plotting import raster_plot

from sim_soens.super_functions import make_letters, pixels_to_spikes, plot_letters, picklit

'''
Simple test case for two-neuron predictive-processing method
 - Any function that might be relevant for testing is brought to the surface
 - Adjustable parameters and run calls are at the bottom of the script
 - Synaptic weights, dendritic weights, and neuron parameters all easily adjustable
 - Plotting function gives activity gist for both neurons
 - Just run this script to test
 - For a more thorough overview of NeuralZoo objects, see library_tour.py
'''

def main():

    def make_ZN_node():
        W = [
            [[.5,.4]],
            [[.3,-.3,.3],[-.3,.3,.3]],
            [[.3,.3,-.3],[-.3,.3,-.3],[-.3,.3,.3],[-.3,.3,-.3],[.3,-.3,.3],[.3,-.3,.3]]
        ]

        params = {
            "s_th": 0.15,
            "ib": 1.8,
            "tau_ni": 500,
            "tau_di": 250,
            "beta_ni": 2*np.pi*1e2,
            "beta_di": 2*np.pi*1e2,
            "weights": W,
        }

        node = SuperNode(**params)
        node.normalize_fanin(1.5)
        node.plot_structure()

        return node

    def make_rand_node():

        # W = [
        #     [np.random.random(2)],
        #     np.random.random((2,3)),
        #     np.random.random((6,3))
        # ]

        W = [
            [np.ones(2)],
            np.ones((2,3)),
            np.ones((6,3))
        ]

        params = {
            "s_th": 0.15,
            "ib": 1.8,
            "tau_ni": 500,
            "tau_di": 250,
            "beta_ni": 2*np.pi*1e2,
            "beta_di": 2*np.pi*1e2,
            "weights": W,
        }

        # params = {
        #     "s_th": 0.5,
        #     "ib": 1.8,
        #     "tau_ni": 500,
        #     "tau_di": 150, 
        #     "beta_ni": 2*np.pi*1e3,
        #     "beta_di": 2*np.pi*1e3,
        #     "weights": W,
        # }

        node = SuperNode(**params)
        node.normalize_fanin(1.5)
        node.random_flux(0.05)
        return node
    
    def run_context_and_event(node,l1,l2):
        letters = make_letters(patterns='zvnx+')

        persistent_context = False
        if persistent_context == False:
            prime_times = np.arange(50,251,50)
            event_times = np.arange(300,600,50)
        else:
            prime_times = np.arange(50,600,50)
            event_times = np.arange(300,600,50)    

        primer = pixels_to_spikes(letters[l1],prime_times)
        event  = pixels_to_spikes(letters[l2],event_times)

        proximal_input = SuperInput(channels=9,type='defined',defined_spikes=primer,duration=500)
        basal_input    = SuperInput(channels=9,type='defined',defined_spikes=event, duration=500)

        separate_input=True
        if separate_input==True:
            # print("separate input")
            proximal_connections = [(i,i) for i in range(9)]
            basal_connections    = [(i+9,i) for i in range(9)]
        else:
            # print("global input")
            proximal_connections = basal_connections = [(i,i%9) for i in range(18)]


        node.multi_channel_input(proximal_input,proximal_connections)
        node.multi_channel_input(basal_input,basal_connections)

        net = network(sim=True,dt=.1,tf=600,nodes=[node],backend='julia')
        # print(len(net.spikes[0]))
        # node.plot_neuron_activity(net=net,phir=True,dend=False,spikes=False,title=f"{l1} - {l2}")
        # node.plot_arbor_activity(net,phir=True)
        return node
    
    def make_update(node,error,eta=0.005,max_offset=None,bounds=None):
        
        for dend in node.dendrite_list:
            if 'ref' not in dend.name:

                if max_offset is not None:
                    max_offset = dend.phi_th

                step = error*eta*np.mean(dend.s)
            
                dend.offset_flux = np.clip(
                    dend.offset_flux+step,
                    a_min=-dend.phi_th,
                    a_max=dend.phi_th
                    )
                
        return node


    def run_and_plot(node,names):
        fig, axs = plt.subplots(len(names), len(names),figsize=(12,6))
        
        fig.subplots_adjust(wspace=0,hspace=0)
        basin  = np.arange(50,251,50)
        proxin = np.arange(300,600,50)
        x = np.arange(0,600.1,.1)

        for i,name1 in enumerate(names):
            for j,name2 in enumerate(names):
                node = run_context_and_event(node,name1,name2)
                s,phi = node.neuron.dend_soma.s,node.neuron.dend_soma.phi_r
                axs[i][j].set_ylim(-0.01,.225)
                axs[i][j].set_title(f"{name1} - {name2}", y=1.0, x=.15, pad=-14)
                if i==0 and j == 0:
                    axs[i][j].plot(x,phi,color='orange',label='flux')
                    axs[i][j].plot(x,s,linewidth=4,color='b',label='signal')
                    axs[i][j].plot(
                        basin,np.zeros(len(basin)),'x',color='red', markersize=8, label='basal input event'
                        )
                    axs[i][j].plot(
                        proxin,np.zeros(len(proxin)),'x',color='purple', markersize=8, label='proximal input event'
                        )
                else:
                    axs[i][j].plot(x,phi,color='orange')
                    axs[i][j].plot(x,s,linewidth=4,color='b')
                    axs[i][j].plot(
                        basin,np.zeros(len(basin)),'x',color='red', markersize=8
                        )
                    axs[i][j].plot(
                        proxin,np.zeros(len(proxin)),'x',color='purple', markersize=8
                        )          

                axs[i][j].axhline(
                    y = 0.15, 
                    color = 'purple', 
                    linestyle = '--',
                    linewidth=.5
                    )
                if i != len(axs):
                    axs[i][j].set_xticklabels([])
                if j != 0:
                    axs[i][j].set_yticklabels([])

        plt.suptitle("Z-to-N Sequence Detector",fontsize=22)
        lines = [] 
        labels = []     
        for ax in fig.axes: 
            Line, Label = ax.get_legend_handles_labels() 
            # print(Label) 
            lines.extend(Line) 
            labels.extend(Label) 
        
        fig.text(0.5, 0.04, 'Time (ns)', ha='center', fontsize=18)
        fig.text(0.04, 0.5, 'Unitless Signal and Flux', va='center', rotation='vertical', fontsize=18)
        fig.legend(lines, labels, bbox_to_anchor=(.15, 0.15), loc='lower left', borderaxespad=0) 

        plt.show()

    def clear_node(node):
        for dend in node.dendrite_list:
            dend.s = []
            dend.phi_r = []
        for syn in node.synapse_list:
            syn.phi_spd = []
        return node


    def learn(names,pattern1,pattern2):

        node = make_rand_node()
        targets = np.zeros((len(names),len(names)))
        targets[names.index(pattern1)][names.index(pattern2)] = 10
        
        accuracy = 0
        epochs = 0
        while accuracy!=100.00:
            outputs = np.zeros(targets.shape)
            correct  = 0
            seen     = 0
            print(f"\nEpoch {epochs}\n--------------------")
            for i,name1 in enumerate(names):
                for j,name2 in enumerate(names):
                    node = run_context_and_event(node,name1,name2)
                    output = len(node.neuron.spike_times)
                    outputs[i][j] = output
                    print(f"  [{name1}, {name2}] : {targets[i][j]} -> {output}")
                    error = targets[i][j] - output
                    node = make_update(node,error)
                    node = clear_node(node)
                    seen+=1
                    if error==0: correct+=1
            accuracy = np.round(correct*100/seen,2)
            epochs+=1
            if np.argmax(targets)==np.argmax(outputs):
                sub = np.concatenate(outputs) - np.concatenate(outputs)[np.argmax(outputs)]
                if sum(n > -3 for n in sub) == 1 and sum(n == 0 for n in sub) == 1:
                    print("Converged!")
                    run_and_plot(node,names)
                    picklit(node,'results/sequencing/',f'node_converged_{epochs}')
                    return node
        run_and_plot(node)


    def make_sequence_indices(sequence,iterations):
        indices = []
        seq_dct = {
            "A":0,
            "B":1,
            "C":2,
        }
        for seq in sequence:
            for i in range(iterations):
                indices.append(seq_dct[seq])
        return indices

    def make_letter_sequence(sequence,letters):
        indices = []
        spikes = []
        for i,pattern in enumerate(sequence):

            if pattern=='A': letter = 'z'
            if pattern=='B': letter = 'v'
            if pattern=='C': letter = 'n'

            spk_times = np.arange(
                50+i*150,
                150*(i+1)+1,
                50
                )
            pixels = letters[letter]
            sub_spikes = pixels_to_spikes(pixels,spk_times)
            # print(pattern,spikes)
            spikes.append(sub_spikes)
            indices.append(spikes[0])
        
        all_spikes = list(map(list.__add__, spikes[0],spikes[1]))
        all_spikes = list(map(list.__add__, all_spikes,spikes[2]))
        
        return all_spikes


    
    
    
    def timer_func(func): 
        from time import time 
        # This function shows the execution time of  
        # the function object passed 
        def wrap_func(*args, **kwargs): 
            t1 = time() 
            result = func(*args, **kwargs) 
            t2 = time() 
            print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
            return result 
        return wrap_func 
    
    @timer_func
    def run_sequence():

        letters = make_letters(patterns='zvnx+')

        W_z = [
            [[.3,.3,.3]],
            [[.3,.3,-.3],[-.3,.3,-.3],[-.3,.3,.3]]
        ]
        node_z = SuperNode(weights=W_z)
        node_z.normalize_fanin(1.5)

        W_v = [
            [[.3,.3,.3]],
            [[.3,-.3,.3],[.3,-.3,.3],[-.3,.3,-.3]]
        ]


        W_n = [
            [[.3,.3,.3]],
            [[-.3,.3,-.3],[.3,-.3,.3],[.3,-.3,.3]]
        ]


        W = [
            [[.1,.175,.25]],
        ]
        taus = [
            [[500,100,10]],
        ]

        params = {
            "s_th": 0.1,
            "ib": 1.8,
            "tau_ni": 100,
            "tau_di": 100,
            "beta_ni": 2*np.pi*1e2,
            "beta_di": 2*np.pi*1e2,
            "weights": W,
            "taus": taus,
        }

        
        # node.normalize_fanin(1.5)
        # node.plot_structure()

        sequence = ['A','B','C']

        # make_letter_sequence(sequence,letters)


        import itertools
        combos = list(itertools.product(sequence, repeat=3))

        print(f"\n     Sequence     |   z  v  n   |   Timing Neuron   ")
        print(  f"----------------------------------------------------")
        for c  in combos:

            make_letter_sequence(c,letters)

            timing_node = SuperNode(**params)

            nodes = [
                SuperNode(name='node_z',tau_di=50,weights=W_z),
                SuperNode(name='node_v',tau_di=50,weights=W_v),
                SuperNode(name='node_n',tau_di=50,weights=W_n),
            ]

            for node in nodes:
                node.normalize_fanin(1.5)

            ### branch test inputs ###
            # indices = make_sequence_indices(c,3)
            # times = list(np.arange(10,len(indices)*50+10,50))
            # inp_spikes = np.array([indices,times])

            ### pattern sequence inputs ###
            inp_spikes = make_letter_sequence(c,letters)

            duration = np.max(inp_spikes[1])+100
            inp = SuperInput(type='defined',defined_spikes = inp_spikes, duration = duration,channels=9)
            # raster_plot(inp.spike_arrays)
            # timing_node.one_to_one(inp)

            for i,node in enumerate(nodes):
                node.one_to_one(inp)
                node.neuron.add_output(timing_node.synapse_list[i])

            run_nodes = nodes+[timing_node]
            
            net = network(sim=True,nodes=run_nodes,dt=0.1,tf=duration,backend='julia')

            outputs = []
            for node in nodes:
                spikes = node.neuron.spike_times
                outputs.append(len(spikes))


            sequence_detection = len(timing_node.neuron.spike_times)
            print(f"  {c}    {outputs}           {sequence_detection}")

            if c==('A', 'B', 'C'):
                timing_node.plot_neuron_activity(net=net,input=inp,title=f"{c}")
                timing_node.plot_arbor_activity(net,phir=True,title=f"{c}")
                # node_z.plot_arbor_activity(net,phir=True,title=f"{c}")
                # node_z.plot_neuron_activity(net=net,input=inp)
            del(net)
            for node in run_nodes:
                del(node)
        print("\n\n")
            
            

    @timer_func
    def run_learning_sequence():

        letters = make_letters(patterns='zvnx+')

        nodes = []
        for i in range(3):
            # W_init = [
            #     [np.ones(3)*.3],
            #     [np.ones(3)*.3 for _ in range(3)]
            # ]
            W_init = [
                [np.random.uniform(-1,1,[3,])*.3],
                [np.random.uniform(-1,1,[3,])*.3 for _ in range(3)]
            ]
            node = SuperNode(weights=W_init)
            node.normalize_fanin(1.5)
            node.random_flux(0.15)
            nodes.append(node)

        W = [
            [[.1,.175,.25]],
        ]
        taus = [
            [[500,100,10]],
        ]

        params = {
            "s_th": 0.1,
            "ib": 1.8,
            "tau_ni": 100,
            "tau_di": 100,
            "beta_ni": 2*np.pi*1e2,
            "beta_di": 2*np.pi*1e2,
            "weights": W,
            "taus": taus,
        }

        
        # node.normalize_fanin(1.5)
        # node.plot_structure()

        sequence = ['A','B','C']

        # make_letter_sequence(sequence,letters)


        import itertools
        combos = list(itertools.product(sequence, repeat=3))
        acc = 0
        while acc != 100.00:

            print(f"\n     Sequence     |   z  v  n   |   Timing Neuron   ")
            print(  f"----------------------------------------------------")
            seen    = 0
            correct = 0 
            for c  in combos:

                make_letter_sequence(c,letters)

                timing_node = SuperNode(**params)

                ### branch test inputs ###
                # indices = make_sequence_indices(c,3)
                # times = list(np.arange(10,len(indices)*50+10,50))
                # inp_spikes = np.array([indices,times])

                ### pattern sequence inputs ###
                inp_spikes = make_letter_sequence(c,letters)

                duration = np.max(inp_spikes[1])+100
                inp = SuperInput(type='defined',defined_spikes = inp_spikes, duration = duration,channels=9)
                # raster_plot(inp.spike_arrays)
                # timing_node.one_to_one(inp)

                for i,node in enumerate(nodes):
                    node.one_to_one(inp)
                    node.neuron.add_output(timing_node.synapse_list[i])

                run_nodes = nodes+[timing_node]
                
                net = network(sim=True,nodes=run_nodes,dt=0.1,tf=duration,backend='julia')
                if c==('A', 'B', 'C'):
                    target = 5
                else:
                    target = 0

                sequence_detection = len(timing_node.neuron.spike_times)
                error = target - sequence_detection
                if error == 0: correct+=1

                outputs = []
                for node in nodes:
                    spikes = node.neuron.spike_times
                    outputs.append(len(spikes))

                    node = make_update(node,error)
                    node = clear_node(node)

                
                print(f"  {c}    {outputs}           {sequence_detection}")

                del(net)
                # for node in run_nodes:
                #     del(node)
            print("\n\n")
        
            acc = np.round(100*correct/27,2)
            print(f"Epoch accuracy = {acc}")
        print(f"Final accuracy = {acc}")
            
            
    # run_learning_sequence()
    run_sequence()
    # names = ['z','v','n','x','+']          
    # node = learn(names,'z','n')
    


if __name__=='__main__':
    main()