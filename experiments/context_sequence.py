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
        import seaborn as sns
        # plt.style.use('seaborn-muted')
        colors = sns.color_palette("muted")
        # print(plt.__dict__['pcolor'].__doc__)
        # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, axs = plt.subplots(len(names), len(names),figsize=(8,4))
        
        fig.subplots_adjust(wspace=0,hspace=0)
        basin  = np.arange(50,251,50)
        proxin = np.arange(300,501,50)
        x = np.arange(0,600.1,.1)

        avgs = np.array([np.zeros((len(names),)) for _ in range(len(names))])
        for i,name1 in enumerate(names):
            for j,name2 in enumerate(names):
                print(f" {name1} - {name2}")
                node = run_context_and_event(node,name1,name2)
                s,phi = node.neuron.dend_soma.s,node.neuron.dend_soma.phi_r
                # print(node.neuron.dend_soma.dendritic_connection_strengths)#_dict__.keys())
                cs = node.neuron.dend_soma.dendritic_connection_strengths
                avgs[i][j] = np.mean(phi-node.neuron.dend__ref.s*cs[list(cs.keys())[0]]) #len(node.neuron.spike_times) #
                axs[i][j].set_ylim(-0.01,.3)
                axs[i][j].set_title(f"{name1} - {name2}", y=1.0, x=.15, pad=-14)
                spike_times = node.neuron.spike_times
                if i==0 and j == 0:
                    axs[i][j].plot(x,phi,color=colors[1],label='soma flux')
                    axs[i][j].plot(x,s,linewidth=2,color=colors[0],label='soma signal')
                    axs[i][j].scatter(
                        basin,np.zeros(len(basin)),marker='x',color=colors[2], s=50,linewidths=2,  label='basal input event'
                        )
                    axs[i][j].scatter(
                        proxin,np.zeros(len(proxin)),marker='x',color=colors[3], s=50,linewidths=2, label='proximal input event'
                        )
                    axs[i][j].axhline(
                        y = 0.15, 
                        color = 'purple', 
                        linestyle = '--',
                        linewidth=.5,
                        label='threshold'
                        )
                else:
                    axs[i][j].plot(x,phi,color=colors[1])
                    axs[i][j].plot(x,s,linewidth=2,color=colors[0])
                    axs[i][j].scatter(
                        basin,np.zeros(len(basin)),marker='x',color=colors[2], s=50,linewidths=2,  
                        )
                    axs[i][j].scatter(
                        proxin,np.zeros(len(proxin)),marker='x',color=colors[3], s=50,linewidths=2,
                        )          

                    axs[i][j].axhline(
                        y = 0.15, 
                        color = 'purple', 
                        linestyle = '--',
                        linewidth=.5,
                        )
                if len(spike_times) > 0:
                    axs[i][j].scatter(
                        spike_times,np.ones(len(spike_times))*node.s_th,marker='x',color='black', s=60 ,linewidths=2, label='output spike',zorder=10
                        )
                if i != len(axs)-1:
                    axs[i][j].set_xticklabels([])
                if j != 0:
                    axs[i][j].set_yticklabels([])
        


        # plt.figure(figsize=((8,7.5)))
        # plt.title("Soma Activity After Learning n-x Sequence",fontsize=20)
        # plt.ylabel("Basal Input Pattern",fontsize=16)
        # plt.xlabel("Proximal Input Pattern",fontsize=16)
        # plt.xticks(np.arange(0.5,len(names)+0.5,1),names,fontsize=18)
        # plt.yticks(np.arange(0.5,len(names)+0.5,1),names,fontsize=18)
        # print(avgs)
        # print(avgs.shape)
        # heatmap = plt.pcolor(avgs, cmap=plt.cm.Oranges)
        # plt.colorbar(heatmap, cmap=plt.cm.Oranges)
        # # plt.imshow(avgs)
        # plt.tight_layout()
        # plt.show()
                    

        plt.suptitle("Basal Proximal Neuron z-n",fontsize=20)
        lines = [] 
        labels = []     
        for ax in fig.axes: 
            Line, Label = ax.get_legend_handles_labels() 
            # print(Label) 
            lines.extend(Line) 
            labels.extend(Label) 
        
        fig.text(0.5, 0.04, 'Time (ns)', ha='center', fontsize=18)
        fig.text(0.04, 0.5, 'Signal and Flux', va='center', rotation='vertical', fontsize=18)
        fig.legend(lines, labels, bbox_to_anchor=(.2, 0.185), loc='lower left', borderaxespad=0) 
        plt.subplots_adjust(bottom=.15)
        # plt.tight_layout()

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

            # print(f"Max sequence: {np.argmax(outputs)} == {np.argmax(targets)}")
            if np.argmax(targets)==np.argmax(outputs):
                # print("Converged!")
                sub = np.concatenate(outputs) - np.concatenate(outputs)[np.argmax(outputs)]
                # print(sub)
                if sum(n > 0 for n in sub) == 0 and sum(n == 0 for n in sub) == 1:
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
            # print(i,pattern)
            if pattern=='A': letter = 'z'
            if pattern=='B': letter = 'v'
            if pattern=='C': letter = 'n'
            if pattern=='D': letter = 'x'
            if pattern=='E': letter = '+'

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
        
        all_spikes = [[],[]]
        for i in range(len(sequence)):
            all_spikes = list(map(list.__add__, all_spikes,spikes[i]))
        
        return all_spikes

    def branch_test():
            
        # branch test
        W = [
            [[.1,.175,.25]],
        ]
        taus = [
            [[100,50,25]],
        ]

        # W = [
        #     [[.142,.225,.25]],
        # ]
        # taus = [
        #     [[300,90,10]],
        # ]

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


        timing_node = SuperNode(**params)
        c = ['A','B','C']
        indices = make_sequence_indices(c,1)
        times = list(np.arange(10,len(indices)*50+10,50))
        inp_spikes = np.array([indices,times])
        duration = np.max(inp_spikes[1])+75
        inp = SuperInput(type='defined',defined_spikes = inp_spikes, duration = duration,channels=3)
        timing_node.one_to_one(inp)
        
        net = network(sim=True,nodes=[timing_node],dt=0.01,tf=duration,backend='julia')

        import seaborn as sns
        # plt.style.use('seaborn-muted')
        colors = sns.color_palette("muted")
        print(len(colors))

        plt.figure(figsize=(8,3))
        count = 0
        for i,dend in enumerate(timing_node.dendrite_list):
            if 'soma' not in dend.name and 'ref' not in dend.name:
                plt.plot(net.t,dend.s*W[0][0][count],'--',color=colors[count+2],label=f"branch {count+1}",zorder=count+100)

                plt.scatter(
                        inp.spike_rows[count],np.zeros(len(inp.spike_rows[count])),marker='x',
                        color=colors[count+2], s=70,linewidths=2,  zorder=count+150 #label=f'input event branch {count+1}',
                        )
                count+=1
        plt.plot(net.t,timing_node.neuron.dend_soma.s,linewidth=4,label="soma signal")
        # plt.plot(net.t,timing_node.neuron.dend_soma.phi_r,linewidth=2,label="soma  flux")
        spike_times = timing_node.neuron.spike_times
        if len(spike_times) > 0:
                plt.scatter(
                    spike_times,
                    np.ones(len(spike_times))*timing_node.s_th,
                    marker='x',color='black', s=60,linewidths=2, label='output spike',zorder=90
                    )
        plt.axhline(
                    y = timing_node.s_th, 
                    color = colors[7    ], 
                    linestyle = '--',
                    linewidth=.5,
                    label='threshold',
                    )
        plt.title("Sequential Branch Excitation of Timing Neuron",fontsize=20)
        plt.ylabel("Signal",fontsize=18)
        plt.xlabel("Time (ns)",fontsize=18)
        # plt.legend(loc='upper left', bbox_to_anchor=(1, 1.01))
        # plt.subplots_adjust(bottom=.25)
        plt.tight_layout()
        plt.legend()
        plt.show()
    
    
    
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
        # plot_letters(letters)
        W_z = [
            [[.3,.3,.3]],
            [[.3,.3,-.3],[-.3,.3,-.3],[-.3,.3,.3]]
        ]

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

        # W = [
        #     [[.142,.225,.25]],
        # ]
        # taus = [
        #     [[300,90,10]],
        # ]

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


        timing_node = SuperNode(**params)

        nodes = [
            SuperNode(name='node_z',tau_di=50,weights=W_z),
            SuperNode(name='node_v',tau_di=50,weights=W_v),
            SuperNode(name='node_n',tau_di=50,weights=W_n),
        ]

        for node in nodes:
            node.normalize_fanin(1.5)

        # make_letter_sequence(sequence,letters)


        import itertools
        import random
        from sim_soens.soen_plotting import activity_plot
        # combos = list(itertools.product(sequence, repeat=3))
        combos = [random.choices('ABCDE', k=10) for _ in range(1000)]
        # print(combos)
        # combos.insert(0, ['D','E','A','B','C','B','C','E'])
        combos.insert(0, ['A','B','C','A','B','C','A','B','C','A','D','B','C','A','B','C'])

        print(f"\n     Sequence     |   z  v  n   |   Timing Neuron   ")
        print(  f"----------------------------------------------------")
        excit = 0.125
        inhib = -.75
        W_anom = [[[excit,excit,excit,excit,excit,excit,excit,excit,excit,inhib]]]
        tau_anom = [[[20,20,20,20,20,20,20,20,20,150]]]
        anomaly_detector = SuperNode(
            weights = W_anom,
            taus    = tau_anom
            )

        # for i,syn in enumerate(anomaly_detector.synapse_list):
        #     print(syn.name, syn.__dict__.keys())

        # for i,dend in enumerate(anomaly_detector.dendrite_list):
        #     if 'ref' not in dend.name and 'soma' not in dend.name:
        #         # print(dend.name, dend.output_connection_strength, dend.tau_di)
        #         if i < 9:
        #             dend.name = f"excite_{i}"
        #         else:
        #             dend.name = "inhibit"


        combos=[['A','B','C']]
        for c  in combos:
            # print(f"Current sequence: {c}")
            # make_letter_sequence(c,letters)

            ### branch test inputs ###
            # indices = make_sequence_indices(c,3)
            # times = list(np.arange(10,len(indices)*50+10,50))
            # inp_spikes = np.array([indices,times])

            ### pattern sequence inputs ###
            inp_spikes = make_letter_sequence(c,letters)
            # print(inp_spikes[1])
            duration = np.max(inp_spikes[1])+100
            inp = SuperInput(type='defined',defined_spikes = inp_spikes, duration = duration,channels=9)

            times = list(set(inp.spike_arrays[1]))
            times.sort()
            times = times[::3]
            print(
                len(times),times
                )
            # raster_plot(inp.spike_arrays)
            # timing_node.one_to_one(inp)

            for i,node in enumerate(nodes):
                node.one_to_one(inp)
                node.neuron.add_output(timing_node.synapse_list[i])

            
            for i,sig in enumerate(inp.signals):
                anomaly_detector.synapse_list[i].add_input(sig)

            timing_node.neuron.add_output(anomaly_detector.synapse_list[-1])

            anomaly_detector.synapse_list[-1].input_signal.spike_times.append(0.0)

            run_nodes = nodes+[timing_node,anomaly_detector]
            
            net = network(sim=True,nodes=run_nodes,dt=0.1,tf=duration,backend='julia')

            # timing_node.plot_neuron_activity(net=net)

            import seaborn as sns
            # plt.style.use('seaborn-muted')
            colors = sns.color_palette("muted")
            print(len(colors))
            branches = ['z','v','n']
            plt.figure(figsize=(8,3))
            count = 0
            for i,dend in enumerate(timing_node.dendrite_list):
                if 'soma' not in dend.name and 'ref' not in dend.name:
                    plt.plot(net.t,dend.s*W[0][0][count],'--',color=colors[count+2],label=f"branch {branches[count]}",zorder=count+100)

                    # plt.scatter(
                    #         inp.spike_rows[count],np.zeros(len(inp.spike_rows[count])),marker='x',
                    #         color=colors[count+2], s=70,linewidths=2,  zorder=count+150 #label=f'input event branch {count+1}',
                    #         )
                    count+=1
            plt.plot(net.t,timing_node.neuron.dend_soma.s,linewidth=4,label="soma signal")
            # plt.plot(net.t,timing_node.neuron.dend_soma.phi_r,linewidth=2,label="soma  flux")
            spike_times = timing_node.neuron.spike_times
            if len(spike_times) > 0:
                    plt.scatter(
                        spike_times,
                        np.ones(len(spike_times))*timing_node.s_th,
                        marker='x',color='black', s=60,linewidths=2, label='output spike',zorder=90
                        )
            plt.axhline(
                        y = timing_node.s_th, 
                        color = colors[7], 
                        linestyle = '--',
                        linewidth=.5,
                        label='threshold',
                        )
            plt.title("Timing Neuron During a z-v-n Sequence",fontsize=20)
            plt.ylabel("Signal",fontsize=18)
            plt.xlabel("Time (ns)",fontsize=18)
            # plt.legend(loc='upper left', bbox_to_anchor=(1, 1.01))
            # plt.subplots_adjust(bottom=.25)
            plt.tight_layout()
            plt.legend()
            plt.show()

            # anomaly_detector.plot_arbor_activity(net,phir=True,title=c)
            # anomaly_detector.plot_neuron_activity(net=net,phir=True,legend=False,size=(8,4),title='Anomaly Detection')#,legend_out=True,size=(10,4))

            ##################

            import seaborn as sns
            # plt.style.use('seaborn-muted')
            colors = sns.color_palette("muted")
            print(len(colors))
            timing_node = anomaly_detector

            plt.figure(figsize=(8,4))
            count = 0
            for i,dend in enumerate(timing_node.dendrite_list):
                if 'soma' not in dend.name and 'ref' not in dend.name:
                    if count==7:
                        plt.plot(net.t,dend.s*W_anom[0][0][count],'--',color=colors[(count+2)%len(colors)],label=f"excitatory branch",zorder=count+10)
                    elif count == 9:
                        plt.plot(net.t,dend.s*W_anom[0][0][count],'--',color=colors[(count+2)%len(colors)],label=f"inhibitory branch",zorder=count+10)
                    else:
                        plt.plot(net.t,dend.s*W_anom[0][0][count],'--',color=colors[(count+2)%len(colors)],zorder=count+10)


                    # plt.scatter(
                    #         inp.spike_rows[count],np.zeros(len(inp.spike_rows[count])),marker='x',
                    #         color=colors[count+2], s=70,linewidths=2,  zorder=count+150 #label=f'input event branch {count+1}',
                    #         )
                    count+=1
            plt.plot(net.t,timing_node.neuron.dend_soma.s,linewidth=4,label="soma signal",zorder=20)
            plt.plot(net.t,timing_node.neuron.dend_soma.phi_r,color=colors[2],linewidth=2,label="soma  flux")
            spike_times = timing_node.neuron.spike_times
            if len(spike_times) > 0:
                    plt.scatter(
                        spike_times,
                        np.ones(len(spike_times))*.5,
                        marker='x',color='black', s=70,linewidths=3, label='output spike',zorder=25
                        )
            plt.axhline(
                        y = timing_node.neuron.s_th, 
                        color = colors[7], 
                        linestyle = '--',
                        linewidth=.5,
                        label='threshold',
                        )
            
            plt.axvline(
                        x = 1550, 
                        color = colors[3], 
                        linestyle = ':',
                        linewidth=2,
                        label='anomaly',
                        zorder=24
                        )
            
            plt.title("Anomaly Detection in Sequence of Patterns",fontsize=20)
            plt.ylabel("Signal",fontsize=18)
            plt.xlabel("Pattern Sequence",fontsize=18)
            plt.xticks(times,c,fontsize=16)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1.01))
            plt.subplots_adjust(bottom=.25)
            plt.tight_layout()
            # plt.legend()
            plt.show()











            ###################

            outputs = []
            for node in nodes:
                spikes = node.neuron.spike_times
                outputs.append(len(spikes))


            sequence_detection = len(timing_node.neuron.spike_times)
            print(f"  {c}    {outputs}           {sequence_detection}")

            # if c == combos[0] or sequence_detection > 0: # or c == combos[0]: #c==('A', 'B', 'C'):
            #     # timing_node.plot_neuron_activity(net=net,phir=True,phi_th=True,input=inp,title=f"{c}")
            #     # timing_node.plot_arbor_activity(net,phir=True,title=f"{c}")
            #     # activity_plot(nodes,spikes=True,title=f"{c}",legend_all=True)#,legend_out=True)
            #     # node_z.plot_arbor_activity(net,phir=True,title=f"{c}")
            #     # node_z.plot_neuron_activity(net=net,input=inp)

            #     fig, axs = plt.subplots(len(nodes), 1,figsize=(8,4))
            #     fig.subplots_adjust(hspace=0)
            #     for node in nodes:

            del(net)
            for node in run_nodes:
                del(node)
        print("\n\n")
            
            

    @timer_func
    def run_learning_sequence():

        letters = make_letters(patterns='zvnx+')

        nodes = []

        params_nodes = {
            "beta_ni": 2*np.pi*1e3,
            "beta_di": 2*np.pi*1e3,
            "s_th": 0.1
        }


        for i in range(3):
            # W_init = [
            #     [np.ones(3)*.3],
            #     [np.ones(3)*.3 for _ in range(3)]
            # ]
            W_init = [
                [np.random.uniform(-1,1,[3,])*.3],
                [np.random.uniform(-1,1,[3,])*.3 for _ in range(3)]
            ]
            node = SuperNode(weights=W_init,**params_nodes)
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
            "beta_ni": 2*np.pi*1e3,
            "beta_di": 2*np.pi*1e3,
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
                
                net = network(sim=True,nodes=run_nodes,dt=1.0,tf=duration,backend='julia')
                targets = np.zeros((3,))
                for i,let in enumerate(c):
                    if let == "A": targets[0]+=3
                    if let == "B": targets[1]+=3
                    if let == "C": targets[2]+=3

                sequence_detection = len(timing_node.neuron.spike_times)
                # error = target - sequence_detection
                # if error == 0: correct+=1

                outputs = []
                for i,node in enumerate(nodes):
                    spikes = node.neuron.spike_times
                    outputs.append(len(spikes))
                    error = targets[i] - len(spikes)
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
            
            


    # make node and run all combinations of named input patters, generate a multiplot
    node = make_ZN_node()
    names = ['z','n']
    run_and_plot(node,names)

    # l1 = 'z'
    # l2 = 'n'
    # names=['z','v','n']
    # node=make_ZN_node()
    # run_and_plot(node,names)

    # node = make_rand_node()
    # run_context_and_event(node,l1,l2)

    # run_learning_sequence()
        
    # run_sequence()

    # names = ['z','v','n','x','+']          
    # node = learn(names,'n','x')

    # branch_test()
    # from sim_soens.super_functions import picklin
    # node = picklin('results/sequencing/','node_converged_8')
    # run_and_plot(node,names)
    


if __name__=='__main__':
    main()