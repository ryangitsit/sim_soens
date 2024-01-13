import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.super_node import SuperNode
from sim_soens.super_input import SuperInput
from sim_soens.soen_components import network
# from sim_soens.soen_plotting import raster_plot

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


    names = ['z','v','n','x','+']          
    node = learn(names,'z','n')

if __name__=='__main__':
    main()