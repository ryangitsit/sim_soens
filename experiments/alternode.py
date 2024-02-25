#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import multiprocess as mp
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.super_functions import *


def main():

    def get_S_trajectories(exp_name,rng):

        digs = 10
        all_S = []
        runs = range(rng[0],rng[1],1)
        S = []
        length=len(runs)


        all_S = [ [[] for _ in range(digs)] for _ in range(digs)]

        for i,run in enumerate(runs):
            for dig in range(digs):
                print(f"Digit {dig} - Run {run}",end="\r")
                print("  ["+"="*i+">"+" "*(length-i)+"]"+f"{i}/{length}",end="\r")

                # Nodes at a given run seeing a given sample of a given digit
                nodes = picklin(f"results\\MNIST\\{exp_name}\\full_nodes_prime",f"full_{10+i}_{dig}_nodes_at_{run}")

                # iterate over each node in this situation
                for d in range(digs):

                    node = nodes[d]

                    # collect the 7 average signals of the penultimate layer 
                    signal_last = []
                    for dend in node.dendrites[1][0]:
                        signal_last.append(np.mean(dend.s))

                    # append them to the confusion matrix
                    all_S[dig][d].append(signal_last)

        return all_S

    def plot_trajects(exp_name,all_S,rng):
        digs=10
        runs = range(rng[0],rng[1],1)
        loc = f"results\\MNIST\\{exp_name}\\"
        colors = sns.color_palette("muted")
        fig, axs = plt.subplots(digs,digs,figsize=(20,12))
        fig.subplots_adjust(wspace=0,hspace=0)    
        for i in range(digs):
            for j in range(digs):
                S = np.array(all_S[i][j])
                if i == j == 0:
                    axs[i][j].plot(S,label=np.arange(1,len(S[0])+1,1))
                else:
                    axs[i][j].plot(S)
                axs[i][j].set_xticklabels([])
                axs[i][j].set_yticklabels([])
                # axs[i][j].set_ylim(0,0.5)
                if i == j:
                    axs[i][j].set_facecolor("lavender")
                else:
                    axs[i][j].set_facecolor("whitesmoke")
                # axs[i][j].set_title(f"ith digit {i} :: jth node {j}")
        # plt.suptitle(f"Digit Seen vs Class Node for Signal Trajectories of Pre-Somatic Dendritic Layer",fontsize=28)
        lines = [] 
        labels = []     
        for ax in fig.axes: 
            Line, Label = ax.get_legend_handles_labels() 
            lines.extend(Line) 
            labels.extend(Label) 
        fig.text(0.5, .9, f"Digit vs Node for Signal Trajectories of Pre-Somatic Dendritic Layer: Runs {min(runs)} -> {max(runs)}", ha='center',fontsize=24)
        fig.text(0.5, 0.1, 'Class Node', ha='center', fontsize=24)
        fig.text(0.09, 0.5, 'Digit Seen', va='center', rotation='vertical', fontsize=24)
        fig.legend(lines, labels, bbox_to_anchor=(0.94, 0.75), loc='upper right', borderaxespad=0) 
        plt.subplots_adjust(bottom=.15)
        # plt.tight_layout()
        plt.savefig(loc+f"DNsignals_{min(runs)}-{max(runs)}")
        plt.show()

    def renorm_single(thrd,return_dict,nodes,config):
        node = nodes[thrd]
        node.normalize_fanin_symmetric(buffer=0,coeff=1.5)
        return_dict[f"node_{thrd}"] = node
        # return node

    def alter(nodes,factor):
        for  i, node in enumerate(nodes):
            print(f"\nNode {i}")
            # print(len(all_S[i][i]))
            avg_s = np.round(np.mean(all_S[i][i],axis=0),2)
            print(avg_s)
            for ii,s in enumerate(avg_s):
                if s < 0.1:
                    print(f"  {ii} --> {node.dendrites[1][0][ii].name}")
                    # print(node.dendrites[1][0][ii].name)
                    node.neuron.dend_soma.dendritic_connection_strengths[node.dendrites[1][0][ii].name] *= factor 
                else:
                    node.neuron.dend_soma.dendritic_connection_strengths[node.dendrites[1][0][ii].name] *= 1.25
            print(node.neuron.dend_soma.dendritic_connection_strengths)
        return nodes
    
    def normalize_dendrite(node,dendrite,buffer=0,coeff=1)
        if len(dendrite.dendritic_connection_strengths) > 0:  

                        # print(f"{dendrite.name} =>  phi_th = {dendrite.phi_th} :: max_s = {node.max_s_finder(dendrite)}")
                        max_phi = 0.5 - dendrite.phi_th*buffer

                        negatives = []
                        neg_max   = []
                        neg_dends = []

                        positives = []
                        pos_max   = []
                        pos_dends = []


                        for in_name,in_dend in dendrite.dendritic_inputs.items():
                            cs = dendrite.dendritic_connection_strengths[in_name]
                            if 'ref' in in_name: cs = 0
                            max_in = node.max_s_finder(in_dend)
                            # print(f"  {in_name} -> {cs}") 

                            if cs<0:
                                # print(cs)
                                negatives.append(cs)
                                neg_max.append(cs*max_in)
                                neg_dends.append(in_dend)

                            elif cs>0:
                                positives.append(cs)
                                pos_max.append(cs*max_in)
                                pos_dends.append(in_dend)
                    

                        if sum(pos_max) > max_phi:
                            # print(f" Normalizing input to {dendrite.name} from {sum(pos_max)} to {max_phi}")
                            for pos_dend in pos_dends:
                                cs = dendrite.dendritic_connection_strengths[pos_dend.name]
                                cs_max = cs*node.max_s_finder(pos_dend)
                                cs_proportion = cs_max/sum(pos_max)
                                cs_normalized = max_phi*cs_proportion/node.max_s_finder(pos_dend) 
                                # print(f"   {pos_dend} -> {cs_normalized}")
                                dendrite.dendritic_connection_strengths[pos_dend.name] = cs_normalized*coeff
                        # print(sum(np.abs(neg_max)))
                        if sum(np.abs(neg_max)) > max_phi:
                            # print(f" Normalizing input to {dendrite.name} from {sum(neg_max)} to {max_phi}")

                            for neg_dend in neg_dends:
                                cs = np.abs(dendrite.dendritic_connection_strengths[neg_dend.name])
                                cs_max = np.abs(cs*node.max_s_finder(neg_dend))
                                cs_proportion = cs_max/sum(np.abs(neg_max))
                                cs_normalized = np.abs(max_phi*cs_proportion/node.max_s_finder(neg_dend))*-1
                                # print(f"   {neg_dend} -> {cs_normalized}")
                                dendrite.dendritic_connection_strengths[neg_dend.name] = cs_normalized*coeff

    def renormalize_fanin(nodes,config):
        length = len(nodes)
        manager = mp.Manager()
        return_dict = manager.dict()
        return_dict = manager.dict()
        thrds = []
        for thrd in range(config.digits):
            thrds.append(
                mp.Process(
                    target=renorm_single, 
                    args=(thrd,return_dict,nodes,config)
                    )
                )

        for thrd in thrds:
            thrd.start()

        for thrd in thrds:
            thrd.join()

        nodes = []
        for i in range(length):
            print(f'Adding node_{i}')
            nodes.append(return_dict[f'node_{i}'])
        return nodes

    def alternodes_inhibition(exp_name,all_S,alt_type='inhibit',renorm_fanin=False):

        nodes = picklin(f"results\\MNIST\\{exp_name}\\nodes\\","init_nodes.pickle")
        config = picklin(f"results\\MNIST\\{exp_name}\\","config.pickle")
        node = nodes[0]
        print(node.neuron.dend_soma.dendritic_connection_strengths)

        if alt_type=="inhibit":
            factor = -1
            
        nodes = alter(nodes,factor)

        if renorm_fanin==True:
            nodes = renormalize_fanin(nodes,config)
            renorm_str='_renormed'
        else:
            renorm_str=''
            
        print(node.neuron.dend_soma.dendritic_connection_strengths)
        new_name = exp_name + '_alternode_inh' + renorm_str
        new_path = f"results\\MNIST\\{new_name}\\nodes\\"
        config.exp_name  = new_name
        config.alternode = new_name
        picklit(
            nodes,
            new_path,
            "init_nodes")
        picklit(
            config,
            f"results\\MNIST\\{new_name}\\",
            "config"
        )
        

    exp_name = "updates_cobuff"
    rng = (360,390)
    all_S = get_S_trajectories(exp_name,rng)

    alternodes_inhibition(exp_name,all_S,renorm_fanin=True)

    def alternodes_pruning(exp_name):
        pass


if __name__=='__main__':
    main()

