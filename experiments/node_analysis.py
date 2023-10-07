import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.super_functions import *


def load_nodes(run,digit,sample,name):
    nodes = picklin(f"results\\MNIST\\{name}\\full_nodes",f"full_{sample}_{digit}_nodes_at_{run}")
    # print("Loaded nodes:")
    # for node in nodes:
    #     print(" ",node.name)
    return nodes

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

name = 'fanin_1.5_full'
# name = 'MNIST_large'
# name = 'MNIST_asymmetic'
nodes = picklin(f"results\\MNIST\\{name}\\nodes",f"eternal_nodes")
node = nodes[0]
for i,(k,v) in enumerate(nodes[0].__dict__.items()):
    print(k,sys.getsizeof(v))
print(node.dendrite_list[0].__dict__.keys())
print(node.synapse_list[0].spd_duration_converted)


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
                f"Noisy 9-Pixel Classifier - {names[i]}",
                fontsize=16
                )
            
            plt.xlabel("Total Iterations",fontsize=14)
            plt.ylabel("Flux Offset",fontsize=14)
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



