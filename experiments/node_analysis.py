#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.super_functions import *

#%%
def load_nodes(run,digit,sample,name):
    nodes = picklin(f"results\\MNIST\\{name}\\full_nodes",f"full_{sample}_{digit}_nodes_at_{run}")
    # print("Loaded nodes:")
    # for node in nodes:
    #     print(" ",node.name)
    return nodes
nodes = picklin(f"results\\MNIST\\updates_cobuff\\full_nodes_prime",f"full_{10}_{0}_nodes_at_{360}")

#%%

loaded_weights = picklin('./saved_data/','W_symmetric_relu_nobias_1000')
print(len(loaded_weights))
print(loaded_weights[0][0])

#%%
W = loaded_weights[0]
ws = []
for i,layer in enumerate(W):
    for j,group in enumerate(layer):
        for k,w in enumerate(group):
            ws.append(w)
print(np.array(ws).shape)
# ws = np.concatenate(np.concatenate(np.concatenate(loaded_weights[0])))
plt.hist(ws,bins=100,label='w')
plt.show()

#%%
# print(len(loaded_weights[0]))
# print(len(loaded_weights[0][0]))
# print(len(loaded_weights[0][0][0]))
# print(len(loaded_weights[0][0][0]))
# print(len(nodes[0].dendrites))

nodes = [nodes[0]]
for n,node in enumerate(nodes):
    for i,layer in enumerate(node.dendrites[1:]):
        print(len(layer))
        for j,dens in enumerate(layer):
            for k,d in enumerate(dens):
                print(d.name,n,i,j,k)
                # print(loaded_weights[n][i][j][k])
                d.offset_flux = loaded_weights[n][i][j][k]
#%%

loc = f"results\\MNIST\\updates_cobuff\\"
digs = 10
all_S = []

runs = range(360,390,1)
# runs = range(360,362,1)
# runs = [20,30]
S = []
length=len(runs)


all_S = [ [[] for _ in range(digs)] for _ in range(digs)]

#%%
for i,run in enumerate(runs):
    for dig in range(digs):
        print(f"Digit {dig} - Run {run}",end="\r")
        print("  ["+"="*i+">"+" "*(length-i)+"]"+f"{i}/{length}",end="\r")

        # Nodes at a given run seeing a given sample of a given digit
        nodes = picklin(f"results\\MNIST\\updates_cobuff\\full_nodes_prime",f"full_{10+i}_{dig}_nodes_at_{run}")


        # iterate over each node in this situation
        for d in range(digs):

            node = nodes[d]

            # collect the 7 average signals of the penultimate layer 
            signal_last = []
            for dend in node.dendrites[1][0]:
                signal_last.append(np.mean(dend.s))

            # append them to the confusion matrix
            all_S[dig][d].append(signal_last)

    
#%%
loc = f"results\\MNIST\\updates_cobuff\\"
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

#%%

def alternodes_inhibition(exp_name,all_S,alt_type='inhibit',renorm_fanin=False):
    print("HERE")
    nodes = picklin(f"results\\MNIST\\{exp_name}\\nodes\\","init_nodes.pickle")
    config = picklin(f"results\\MNIST\\{exp_name}\\","config.pickle")
    node = nodes[0]
    print(node.neuron.dend_soma.dendritic_connection_strengths)

    if alt_type=="inhibit":
        factor = -1
        

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

        print(node.neuron.dend_soma.dendritic_connection_strengths)

    if renorm_fanin==True:
        for  i, node in enumerate(nodes):
            node.normalize_fanin_symmetric(buffer=config.fan_buffer,coeff=config.fan_coeff)
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
    
    

print(np.array(all_S).shape)
exp_name = "updates_cobuff"
alternodes_inhibition(exp_name,all_S,alt_type='inhibit',renorm_fanin=True)

# def alternodes_pruning(exp_name):
#     pass

# def alternodes_renormalization(exp_name):
#     pass








#%%
# #%%
# nodes = picklin(f"results\\MNIST\\updates_cobuff\\full_nodes_prime",f"full_10_3_nodes_at_360")
# node = nodes[0]
# #%%

# plt.style.use('seaborn-muted')

# phis = [max(dend.s) for dend in node.dendrite_list]
# plt.hist(phis,bins=50,label='max_s')
# # plt.show()

# phis = [max(dend.phi_r) for dend in node.dendrite_list]
# plt.hist(phis,bins=50,label='max_phis')
# # plt.show()


# offsets = [dend.offset_flux for dend in node.dendrite_list]
# plt.hist(offsets,bins=50,label='offsets')
# # plt.show()

# plt.legend()
# plt.show()

# #%%

# for lay in range(1,6):
#     offsets = []
#     phis    = []
#     signals = []
#     for dend in node.dendrite_list:
#         if f'lay{lay}' in dend.name:
#             offsets.append(dend.offset_flux)
#             phis.append(max(dend.phi_r))
#             signals.append(max(dend.s))

#     plt.hist([signals,phis,offsets],bins=15,label=['signals','phis','offsets'])
#     # plt.hist(phis,   bins=50,label='max_phis')
#     # plt.hist(phis,   bins=50,label='max_s')  
#     plt.legend()
#     plt.title(f"Layer {lay}")
#     plt.show()

# #%%
# node=nodes[0]
# lays = [[] for _ in range(len(node.dendrites))]
# phays = [[] for _ in range(len(node.dendrites))]
# for l,layer in enumerate(node.dendrites):
#     for g,group in enumerate(layer):
#         for d,dend in enumerate(group):
#             if np.mean(dend.s) != 0:
#                 lays[l].append(dend.s)
#             if np.mean(dend.phi_r) != 0:
#                 phays[l].append(dend.phi_r)

# # plt.style.use('seaborn-muted')
# colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# plt.figure(figsize=(8,4))
# for l,lay in enumerate(lays):
#     if l == 0:
#         lw = 4
#     else:
#         lw = 2
#     plt.plot(
#         np.mean(lay,axis=0),
#         linewidth=lw,
#         color=colors[l%len(colors)],
#         label=f'Layer {l} Mean Signal'
#         )
#     plt.plot(
#         np.mean(phays[l],axis=0),
#         '--',
#         linewidth=.5,
#         color=colors[l%len(colors)],

#         # label=f'Layer {l} Mean Flux'
#         )
# plt.legend()
# plt.show()
# #%%

# def heat_map(node):
#     data = np.zeros((784,7))

#     count = 0
#     for l,layer in enumerate(node.dendrites[::-1]):

#         if l == 6:
#             # print("soma")
#             midpoint  = 784/2
#             halflayer = 1

#         elif l==5:
#             # print("penultimate")
#             midpoint  = 784/2 
#             halflayer = 4

#         elif l==4:
#             # print("penultimate")
#             midpoint  = 784/2 
#             halflayer = 24

#         else:
#             midpoint  = 784/2 
#             halflayer = np.ceil(len(layer))

#         # print(f"layer = {l} :: groups = {len(layer)} :: mid = {midpoint} :: halflayer = {halflayer}")
#         for g,group in enumerate(layer):
#             for d,dend in enumerate(group):

#                 idx = int(count + midpoint - halflayer)
#                 s = np.mean(dend.s)*1000
#                 # print(idx,l,s)
#                 data[idx][l] = s

#                 count+=1
#         count=0


#     # plt.figure(figsize=(10,10))
#     # plt.imshow( data, extent=[0, 7, 0, 784], aspect=7/784 )
#     # plt.title(node.name)
#     # plt.show()
        
#     return data  

# #%%
# import seaborn as sns 
# from matplotlib.colors import ListedColormap
# cmap = ListedColormap(sns.color_palette("ch:s=.25,rot=-.25"))
# # cmap = sns.light_palette("Greens", as_cmap=True)


# sums = np.zeros((10,10))
# fig, axs = plt.subplots(10,10,figsize=(18,18), sharex=True, sharey=True)
# fig.subplots_adjust(hspace=0,wspace=0)
# for i in range(10):
#     print(r"["+"="*i+">"+" "*(10-i)+"]")
#     # nodes = load_nodes(3000,i,0,"thresh_full")
#     for j,node in enumerate(nodes):
#         data = heat_map(node)
#         # plt.imshow(data, extent=[0, 7, 0, 784], aspect=7/784) #, cmap=cmap)
#         # plt.show()
#         sums[i][j]=sum(sum(data))
#         axs[i][j].imshow(data, extent=[0, 7, 0, 784], aspect=7/784) #, cmap=cmap)
#         axs[i][j].set_xticklabels([])
#         axs[i][j].set_yticklabels([])
        
#         if i == len(axs)-1:
#             axs[i][j].set_xticks([3.5],[str(j)],fontsize=18)
#         # if j==0:
#         #     axs[i][j].set_yticks([1],[str(i)],fontsize=18)

# fig.xticks(np.arange(0,10,1),np.arange(0,10,1),fontsize=18)
# fig.yticks(np.arange(0,10,1),np.arange(0,10,1),fontsize=18)
# plt.show()


# plt.imshow(sums)
# plt.show()

#%%
def mem_analysis(span):
    nodes = load_nodes(10,0,'inelast')

    plt.style.use('seaborn-muted')
    plt.figure(figsize=(8,4))

    import sys

    mem_tracker = {}
    for i,(k,v) in enumerate(nodes[0].__dict__.items()):
        if k != 'dend_dict':
            mem_tracker[k] = []

    for i in range(span[0],span[1])[::10]:
        nodes = load_nodes(i,0,'inelast')
        # print(len(nodes[0].synapse_list))
        for i,(k,v) in enumerate(nodes[0].__dict__.items()):
            if k != 'dend_dict':
                mem_tracker[k].append(sys.getsizeof(v))
            # if i in range(5,6):
            # print(i,k,type(v),sys.getsizeof(v))#,f"\n\n\n")
        # print("\n\n")

    for i,(k,v) in enumerate(nodes[0].__dict__.items()):
        plt.plot(mem_tracker[k],label=k)
    plt.legend()
    plt.show()

#%%
# mem_analysis([10,50])
    

def offset_analysis(path,files,digit,layer):

    # file_name = files[0][len(path):len(files[0])-len('.pickle')]
    nodes = picklin(path,files[0][:len(files[0])-len('.pickle')])

    for node in nodes:
        total = 0
        on = 0
        for dend in nodes[0].dendrite_list:
            if np.mean(dend.s) > 0:
                on+=1
            total +=1
        print(f"Node activity ratio = {on/total}")


    plt.style.use('seaborn-muted')
    plt.figure(figsize=(8,4))
    
    import sys

    # dend_offsets = [[] for _ in range(len(nodes[0].dendrite_list))]
    length = len(np.concatenate(nodes[0].dendrites[layer]))
    # print(length)
    dend_offsets = [[] for _ in range(length)]

    for file_name in files:
        # nodes = load_nodes(i,0,'inelast')
        # file_name = f[len(path):len(f)-len('.pickle')]
        if digit == 'any':
            nodes = picklin(path,file_name[:len(file_name)-len('.pickle')])

            print(file_name)
            for node in nodes:
                total = 0
                on = 0
                for dend in nodes[0].dendrite_list:
                    if np.mean(dend.s) > 0:
                        on+=1
                    total +=1
                print(f"  Node activity ratio = {on/total}")
            print("\n\n")


    #         d_count = 0
    #         for ii,dend in enumerate(nodes[1].dendrite_list):
    #             if f'lay{str(layer)}' in dend.name:
    #                 # print(" ",dend.name)
    #                 dend_offsets[d_count].append(dend.offset_flux)
    #                 d_count+=1

    #     elif file_name[7] == f'{digit}':
    #         # print(file_name)
    #         nodes = picklin(path,file_name[:len(file_name)-len('.pickle')])

    #         d_count = 0
    #         for ii,dend in enumerate(nodes[1].dendrite_list):
    #             if f'lay{str(layer)}' in dend.name:
    #                 # print(" ",dend.name)
    #                 dend_offsets[d_count].append(dend.offset_flux)
    #                 d_count+=1


    # plt.plot(np.transpose(dend_offsets))
    # plt.show()

def get_ordered_files(path):
    import os
    import time
    import sys
    # import os
    import glob
    if os.path.exists(path) == True:
        file_list = os.listdir(path)
        run_list = []
        fl = []
        for file in file_list:
            if'eternal' not in file and 'init' not in file:

                start = file.index('t')+2
                finish = len(file)-len(".pickle")

                run_list.append(int(file[start:finish]))

                fl.append(file)

        sorted_inds = np.argsort(np.array(run_list))

        ordered_files = []
        for ind in sorted_inds:
            ordered_files.append(fl[ind])

    return ordered_files

### MNIST ###
# name = "unbounded_deep"
# # name = "MNIST_asymmetic"
# path = f"results\\MNIST\\{name}\\full_nodes\\"
# files = get_ordered_files(path)
# digit = 'any'
# layer = 1

# nodes = load_nodes(0,0,0,'unbounded_deep')
# offset_analysis(path,files,digit,layer)

# name = 'thresh_full'
# # name = 'target50_maxflux_full'
# # name = 'MNIST_large'
# # name = 'MNIST_asymmetic'
# nodes = picklin(f"results\\MNIST\\{name}\\nodes",f"eternal_nodes")
# # for node in nodes:
# #     print(node.name)
# node = nodes[0]
# print(node.__dict__.keys())
# print(node.seen,node.passed)

# node.plot_structure()
# print(node.neuron.name)
# print(len(node.dendrite_list))
# print(len(node.synapse_list))
# print(node.synapse_list[0].__dict__.keys())

# print("Node size = ", sys.getsizeof(node))
# for i,(k,v) in enumerate(nodes[0].__dict__.items()):
#     print(k,sys.getsizeof(v))
# print(node.dendrite_list[0].__dict__.keys())
# print(node.synapse_list[0].spd_duration_converted)

# dend = node.dendrite_list[0]

# for i,(k,v) in enumerate(dend.__dict__.items()):
#     print(k,sys.getsizeof(v))

# print(dend.doubleroll)

# for digit in digits:

def mean_layer_analysis(name):
    nodes = load_nodes(run,0,0,name)
    runs = np.arange(50,1000,50)
    digits = [0,1,2]
    MEANS = [[[] for i in range(len(np.concatenate(nodes[0].dendrites[1])))] for i in range(len(nodes))]
    for run in runs:
        nodes = load_nodes(run,0,name)
        trace_vecs = [[] for i in range(len(nodes))]
        for i,node in enumerate(nodes):
            # plt.title(node.name)
            for d,dend in enumerate(np.concatenate(node.dendrites[1])):
                # plt.plot(dend.s)
                trace_vecs[i].append(np.mean(dend.s))
                MEANS[i][d].append(np.mean(dend.s))
            # plt.show()

        # colors = ['r','b','g']
        # for c,trace in enumerate(trace_vecs):
        #     for t in trace:
        #         plt.plot(np.arange(0,10,1)+c*10,np.ones(10)*t)#,color=colors[c])
        # plt.show()
        # print(trace)
    print(MEANS)
    for n,nodes in enumerate(MEANS):
        for m,mns in enumerate(nodes):
            plt.plot(np.arange(0,len(mns),1)+n*len(mns),mns)
    plt.show()

def vector_analysis(name,run,digit):
    nd = load_nodes(run,0,0,name)
    digits = [0,1,2]
    digit = 0
    MEANS = [[[] for i in range(len(np.concatenate(nd[0].dendrites[1])))] for i in range(len(nd))]


    for sample in range(10):
        nodes = load_nodes(run,digit,sample,name)
        for i,node in enumerate(nodes):
            for d,dend in enumerate(np.concatenate(node.dendrites[1])):
                MEANS[i][d].append(np.mean(dend.s))
    # print(MEANS)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for n,nodes in enumerate(MEANS):
        for m,mns in enumerate(nodes):
            plt.plot(np.arange(0,len(mns),1)+n*len(mns),mns,color=colors[m%(len(colors))])
    plt.show()

# name = 'vector_train'
# run = 1
# # name = 'modern_inh_counter'
# # run = 1031
# for digit in [0,1,2]:
#     vector_analysis(name,run,digit)


### Pixels ###
def plot_offsets(trajects):
    plt.style.use('seaborn-muted')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sub_colors = ['r','b','g']
    sub_colors_lay2_z = ['g','g','r','r','g','r','r','g','g']
    sub_colors_lay2_n = ['g','r','g','g','r','g','r','g','r']
    sub_colors_lay2_v = ['r','g','r','g','r','g','g','r','g']
    lay2_cols = [sub_colors_lay2_z,sub_colors_lay2_v,sub_colors_lay2_n]
    names = ['Z','V','N']
    for i,traject in enumerate(trajects):
        if i >=  0:
            
            plt.figure(figsize=(8,4))
            count1=0
            count2=0
            for name,offset in reversed(traject.items()):
                if 'soma' in name:
                    name = 'soma'
                    converge_length = len(offset)
                    # plt.plot(offset,color=colors[i],label=name,linewidth=4)
                    plt.plot(offset,color=colors[0],label=name,linewidth=4)
                elif 'lay1' in name:
                    col = colors[3]
                    # col = sub_colors[count1]

                    if count1 == 0:
                        # plt.plot(offset,'--',linewidth=2,label='Layer 1')
                        plt.plot(offset,'--',color=col,linewidth=2,label='Layer 1')
                    else:
                        # plt.plot(offset,color=colors[0],label=name,linewidth=3)
                        plt.plot(offset,'--',color=col,linewidth=2)
                    count1+=1

                elif 'lay2' in name:
                    # col = colors[2]
                    # col = sub_colors_lay2_z[count2%len(colors)]
                    col = lay2_cols[i][count2]
                    if count2 == 0:
                        plt.plot(offset,':',color=col,label='Layer 2',linewidth=1)
                    else:
                        plt.plot(offset,':',color=col,linewidth=1)
                    # plt.plot(offset,color=colors[4],label=name)
                    count2+=1

            # plt.title(
            #     f"Noisy 9-Pixel Classifier {regime} {converge_type} Convergence - {names[i]}",
            #     fontsize=16
            #     )
            plt.title(
                f"Dendritic Learning of 9-Pixel Patterns - {names[i]}",
                fontsize=20
                )
            
            plt.xlabel("Total Iterations",fontsize=18)
            plt.ylabel("Flux Offset",fontsize=18)
            plt.subplots_adjust(bottom=.15)
            plt.legend()
            plt.show()

def get_trajects(path):
    import os
    file_list = os.listdir(path)
    success = 0
    for file in file_list:
        if 'accs' in file and 'png' not in file:
            acc = picklin(path,file)
            if acc[-1] == 100:
                success = 1
            else:
                success = 0
        
        if 'trajects' in file and success == 1:
            print(file)
            plot_offsets(picklin(path,file))


# path = 'results/jul_testing/early_plots'
# get_trajects(path)



