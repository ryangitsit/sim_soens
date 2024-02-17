import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.super_node import SuperNode
from sim_soens.super_input import SuperInput
from sim_soens.soen_components import network
from sim_soens.super_functions import make_letters, pixels_to_spikes

'''
Script for 9-pixel-pattern-driven basal-proximal context-stimulus demonstration
'''

def main():

    def make_ZN_node():
        '''
        Makes and return a node that will only fire when z-then-n simuli are received
        '''
        # weighting such that proximal branch responds to z-stimulus and proximal to n-stimulues
        W = [
            [[.5,.4]],
            [[.3,-.3,.3],[-.3,.3,.3]],
            [[.3,.3,-.3],[-.3,.3,-.3],[-.3,.3,.3],[-.3,.3,-.3],[.3,-.3,.3],[.3,-.3,.3]]
        ]

        # tau such that the more strongly connected branch (basal) must be excited first to fire
        params = {
            "s_th": 0.15,
            "ib": 1.8,
            "tau_ni": 500,
            "tau_di": 250,
            "beta_ni": 2*np.pi*1e2,
            "beta_di": 2*np.pi*1e2,
            "weights": W,
        }

        # make node with params
        node = SuperNode(**params)

        # normalize fanin
        node.normalize_fanin(1.5)

        # visualize
        node.plot_structure()

        return node

    
    def run_context_and_event(node,l1,l2,dt):
        '''
        Runs trial where first context pattern l1 and then stimulus l2 are received by a given node
        '''
        # input patterns
        letters = make_letters(patterns='zvnx+')

        # spike times
        context_times = np.arange(50,600,50)
        stimulus_times = np.arange(300,600,50)    

        # convert patterns to spikes
        primer = pixels_to_spikes(letters[l1],context_times)
        event  = pixels_to_spikes(letters[l2],stimulus_times)

        # create input objects
        proximal_input = SuperInput(channels=9,type='defined',defined_spikes=primer,duration=500)
        basal_input    = SuperInput(channels=9,type='defined',defined_spikes=event, duration=500)

        # connect to appropriate synapses
        proximal_connections = [(i,i) for i in range(9)]
        basal_connections    = [(i+9,i) for i in range(9)]

        node.multi_channel_input(proximal_input,proximal_connections)
        node.multi_channel_input(basal_input,basal_connections)

        # run network
        net = network(sim=True,dt=dt,tf=600,nodes=[node],backend='julia')

        # other output if interested
        # print(len(net.spikes[0]))
        # node.plot_neuron_activity(net=net,phir=True,dend=False,spikes=False,title=f"{l1} - {l2}")
        # node.plot_arbor_activity(net,phir=True)

        return node



    def run_and_plot(node,names,):
        '''
        Runs input combinations defined by names on a given node and produces a multiplot
        '''
        dt = 0.01
        # plotting detials
        colors = sns.color_palette("muted")
        fig, axs = plt.subplots(len(names), len(names),figsize=(8,4))
        fig.subplots_adjust(wspace=0,hspace=0)

        # x-axis for spikes
        basin  = np.arange(50,251,50)
        proxin = np.arange(300,501,50)

        # for signals/phis
        x = np.arange(0,600.001,dt)

        # for a heatmap if needed (usually for a larger combination space)
        avgs = np.array([np.zeros((len(names),)) for _ in range(len(names))])

        # iterate over all possible input orderings
        for i,name1 in enumerate(names):
            for j,name2 in enumerate(names):
                print(f" {name1} - {name2}")

                # run the trial with the given order
                node = run_context_and_event(node,name1,name2,dt)

                # collect relevant data
                s,phi = node.neuron.dend_soma.s,node.neuron.dend_soma.phi_r

                # for heatmap
                cs = node.neuron.dend_soma.dendritic_connection_strengths
                avgs[i][j] = np.mean(phi-node.neuron.dend__ref.s*cs[list(cs.keys())[0]]) 

                # plotting details
                axs[i][j].set_ylim(-0.01,.3)
                axs[i][j].set_title(f"{name1} - {name2}", y=1.0, x=.15, pad=-14)

                # collect output spikes
                spike_times = node.neuron.spike_times
                
                # only record these labels for shared legend
                if i==0 and j == 0:

                    # phi
                    axs[i][j].plot(x,phi,color=colors[1],label='soma flux')

                    # signal
                    axs[i][j].plot(x,s,linewidth=2,color=colors[0],label='soma signal')

                    # basal input spikes
                    axs[i][j].scatter(
                        basin,np.zeros(len(basin)),
                        marker='x',color=colors[2], s=50,linewidths=2,  label='basal input event'
                        )
                    
                    # proximal input spikes
                    axs[i][j].scatter(
                        proxin,np.zeros(len(proxin)),
                        marker='x',color=colors[3], s=50,linewidths=2, label='proximal input event'
                        )
                    
                    # # threshold line
                    axs[i][j].axhline(
                        y = 0.15, 
                        color = 'purple', 
                        linestyle = '--',
                        linewidth=.5,
                        label='threshold'
                        )
                    
                # repeat for rest without labels
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
                        spike_times,np.ones(len(spike_times))*node.s_th,
                        marker='x',color='black', s=60 ,linewidths=2, label='output spike',zorder=10
                        )
                    
                # only ticks for outter plot edges
                if i != len(axs)-1:
                    axs[i][j].set_xticklabels([])
                if j != 0:
                    axs[i][j].set_yticklabels([])
        
                    
        # plotting details
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
        # fig.legend(lines, labels, bbox_to_anchor=(1.05, 1.1), loc='upper left')#, borderaxespad=0) 
        plt.subplots_adjust(bottom=.15)
        # plt.tight_layout()
        plt.show()


    # make node and run all combinations of named input patters, generate a multiplot
    node = make_ZN_node()
    names = ['z','n']
    run_and_plot(node,names)

if __name__=='__main__':
    main()