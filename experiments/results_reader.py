import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.super_functions import *

# # print(df.to_string())   
# # spikes = np.array(np.array(df["spikes"][11]))
# spike_str = df["spikes"][11][1:-1]
# spikes = np.fromstring(spike_str, dtype=int, sep=',')
# print(spikes)
# for spk in spikes:
#     print(spk)
# digit = int(df["digit"][11])
# print(digit)
# print(spikes[digit])

# sub = spikes - spikes[0] 
# print(sub)
# print(sum(n > 0 for n in sub))


def allow_ties(df,index):
    spike_str = df["spikes"][index][1:-1]
    spikes = np.fromstring(spike_str, dtype=int, sep=',')
    digit = int(df["digit"][index])
    sub = spikes - spikes[digit] 
    if sum(n > 0 for n in sub) == 0:
        return True
    else:
        return False

# print(np.array(df["spikes"][11]) - np.array(df["spikes"])[df["digit"][11]]) 



def ongoing_performance(df):
    percents = []
    procents = []

    for start in range(len(df['sample'])-30):
        runs = 0
        run_wins = 0
        count = 0
        total = 0
        counts = np.array([0,0,0])
        totals = np.array([0,0,0])
        for index, row in df.iterrows():
            if index > start:
                total += 1
                totals[df["digit"][index]] += 1
                if df["digit"][index] == df["prediction"][index]:
                # if allow_ties(df,index) == True:
                    count+=1
                    counts[df["digit"][index]] += 1

        percents.append(count/total)
        procents.append(counts/totals)
    return percents, procents


def by_run_performance(df):
    by_run = []
    by_dig_runs = [[] for _ in range(3)]
    dig_runs = [0,0,0]
    run_wins = 0
    for index, row in df.iterrows():

        if df["digit"][index] == df["prediction"][index]:
            run_wins+=1
            dig_runs[df["digit"][index]]+=1


        if (index+1)%30 == 0:
            by_run.append(run_wins/30)

            for i,dig in enumerate(dig_runs):
                by_dig_runs[i].append(dig/10)

            dig_runs = [0,0,0]
            run_wins = 0
    return by_run, by_dig_runs

def load_nodes(run):
    nodes = picklin("results\MNIST_WTA_julia",f"nodes_at_{run}")
    # print("Loaded nodes:")
    # for node in nodes:
    #     print(" ",node.name)
    return nodes

def node_analysis(span):
    nodes = load_nodes(1)
    # save_one = 0
    # for node in nodes:
    #     for s, syn in enumerate(node.synapse_list):
    #         if 'refraction' in syn.name:
    #             save_one +=1
    #             if save_one > 1:
    #                 # print('delete')
    #                 node.synapse_list.remove(syn)
    #                 del(node.synapse_list[s])
    #                 del(syn)
    # del nodes[0].synapse_list[786:]
    # for syn in nodes[0].synapse_list:
    #     print(syn.name)
    # nodes = load_nodes(72)
    # nodes[0].plot_neuron_activity()

    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(8,4))
    # for i in range(span[0],span[1]):
    #     nodes = load_nodes(i) 
    #     extra_syns = 0
    #     for syn in nodes[0].synapse_list:
    #         if 'refraction' in syn.name:
    #             extra_syns+=1
    #     # print(len(nodes[0].synapse_list),extra_syns)
    #     extra_syns = 0

    #     print(syn.name)
    import sys

    mem_tracker = {}
    for i,(k,v) in enumerate(nodes[0].__dict__.items()):
        if k != 'dend_dict':
            mem_tracker[k] = []

    for i in range(span[0],span[1]):
        nodes = load_nodes(i)
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

# node_analysis([0,98])
# print(nodes[0].params["params"]["params"]["dend_dict"].keys())
# print(len(nodes[0].offset_flux[1]))

# df = pd.read_csv(
#     'MNIST_ongoing_julia.csv',
#     names=['sample','digit','spikes','error','prediction','time','init_time','run_time']
#     )

df = pd.read_csv(
    'results\MNIST\julia_inhibit\learning_logger.csv',
    names=['sample','digit','spikes','error','prediction','time','init_time','run_time','offsets']
    )

# percents, procents = ongoing_performance(df)
by_run, digs = by_run_performance(df)

plt.style.use('seaborn-v0_8-muted')
plt.figure(figsize=(8,4))
plt.title("MNIST Classification Performance")
plt.xlabel("Performance Measure Starting Point")
plt.ylabel("Classification Accuracy on Remaining Iterations")
# plt.plot(percents, linewidth = 4,label='total')
# plt.plot(procents, label=['0','1','2'])
plt.plot(by_run, linewidth = 4, label="Total")
plt.plot(np.transpose(digs), '--', label=['0','1','2'])
plt.ylim(0,1)
plt.legend()
plt.show()


plt.plot(df["run_time"])
plt.show()