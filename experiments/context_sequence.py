import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.super_node import SuperNode
from sim_soens.super_input import SuperInput
from sim_soens.soen_components import input_signal, synapse, neuron, network
from sim_soens.soen_plotting import raster_plot

from sim_soens.super_functions import make_letters, pixels_to_spikes, plot_letters

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

    letters = make_letters()
    # plot_letters(letters)
    # for letter,pixels in letters.items():
    #     print(letter)
    #     spikes = pixels_to_spikes(pixels,[20,60,80])
    #     print(spikes)

    def run_context_and_event(l1,l2):
        prime_times = np.arange(50,251,50)
        event_times = np.arange(300,600,50)

        primer = pixels_to_spikes(letters[l1],prime_times)
        event  = pixels_to_spikes(letters[l2],event_times)

        proximal_input = SuperInput(channels=9,type='defined',defined_spikes=primer,duration=500)
        basal_input    = SuperInput(channels=9,type='defined',defined_spikes=event, duration=500)

        proximal_connections = [(i,i) for i in range(9)]
        basal_connections    = [(i+9,i) for i in range(9)]

        node.multi_channel_input(proximal_input,proximal_connections)
        node.multi_channel_input(basal_input,basal_connections)

        net = network(sim=True,dt=.1,tf=600,nodes=[node],backend='julia')
        print(len(net.spikes[0]))
        # node.plot_neuron_activity(net=net,phir=True,dend=False,spikes=False,title=f"{l1} - {l2}")
        # node.plot_arbor_activity(net,phir=True)
        return node.neuron.dend_soma.s,node.neuron.dend_soma.phi_r
    
    names = ['z','v','n']
    fig, axs = plt.subplots(3, 3,figsize=(12,6))
    
    fig.subplots_adjust(wspace=0,hspace=0)
    basin  = np.arange(50,251,50)
    proxin = np.arange(300,600,50)
    x = np.arange(0,600.1,.1)
    for i,name1 in enumerate(names):
        for j,name2 in enumerate(names):
            s,phi = run_context_and_event(name1,name2)
            axs[i][j].set_ylim(-0.01,.225)
            axs[i][j].set_title(f"{name1} - {name2}", y=1.0, x=.1, pad=-14)
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
            if i != 2:
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
    
    # rotating x-axis labels of last sub-plot 
    plt.xticks(rotation=45) 
    
    fig.legend(lines, labels, bbox_to_anchor=(.15, 0.15), loc='lower left', borderaxespad=0) 

    plt.show()
if __name__=='__main__':
    main()