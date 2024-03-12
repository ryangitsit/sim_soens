#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../sim_soens')

from sim_soens.soen_components import network
from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *

import time


#%%

letters = make_letters(patterns='all')

del letters['|  ']
del letters['  |']
del letters['_']
del letters['[]']

# plot_letters(letters)
# plot_letters(letters,'v')
inputs = make_inputs(letters,20)

key_list = list(letters.keys())
print(len(set( key_list )))
keys = '  '.join(key_list)
classes = len(key_list)

def make_rand_weights():
    W = [
    [np.random.rand(3)],
    np.random.rand(3,3)
    ]
    return W


def make_update(node,error,eta,offmax=0):
    for i,dend in enumerate(node.dendrite_list):
        if not hasattr(dend,'update_traj'): dend.update_traj = []
        if 'ref' not in dend.name:
            update = np.mean(dend.s)*error*eta
            dend.offset_flux += update
            if offmax==0: offmax = dend.phi_th
            if dend.offset_flux > 0:
                dend.offset_flux = np.min([dend.offset_flux, offmax])
            elif dend.offset_flux < 0:
                dend.offset_flux = np.max([dend.offset_flux, -1*offmax])
            dend.update_traj.append(dend.offset_flux)
    


runs = 250
duration = 100
eta = 0.025

fans = np.arange(0,6,1)
offs = [0,.25,.5]

for fan in fans:
    for offmax in offs:
        nodes = []
        for i,(k,v) in enumerate(letters.items()):
            print(i,k)
            nodes.append(
                SuperNode(
                    name='node_'+k,
                    weights=make_rand_weights(),
                    beta_di=2*np.pi*1e3,
                    beta_ni=2*np.pi*1e3,
                    )
                    )
            if fan != 0:
                nodes[i].normalize_fanin_symmetric(buffer=0,coeff=1)

        accs=[]
        class_accs = [[] for _ in range(classes)]
        class_successes = np.zeros(classes)
        for run in range(runs):
            print(" "*15,keys)
            s1 = time.perf_counter()
            shuffled = np.arange(0,classes,1)
            np.random.shuffle(shuffled)
            success = 0
            seen = 0
            for i in shuffled:
                letter = key_list[i]

                targets = np.zeros(classes)
                targets[i] = 5

                for node in nodes:
                    node.one_to_one(inputs[letter])
                
                net = network(
                    sim     = True,
                    nodes   = nodes,
                    tf      = duration,
                    dt      = 1.0,
                    backend = 'python'

                )

                spikes = array_to_rows(net.spikes,classes)
                outputs = []
                for nd in range(classes):
                    outputs.append(len(spikes[nd]))

                pred_idx = np.argmax(outputs)
                pred = key_list[pred_idx]
                errors = targets - outputs

                if pred_idx == i: 
                    success += 1
                    class_successes[i]+=1
                    class_accs.append(class_successes[i]/(run+1))

                seen += 1
                for n,node in enumerate(nodes):
                    make_update(node,errors[n],eta,offmax)
                    clear_node(node)

                print(f"{run} -- {letter} --> {pred}   {outputs}  --  {errors}")
                del(net)
            acc = success/seen
            accs.append(acc)
            s2 = time.perf_counter()
            print(f"Run performance:  {np.round(acc,2)}   Run time = {np.round(s2-s1,2)}")

            print("\n=============")


        loc = "results/games/"
        picklit(accs,loc,f"accs_{fan}_{offmax}")
        picklit(class_accs,loc,f"classes_{fan}_{offmax}")
        picklit(nodes,loc,f"nodes_{fan}_{offmax}")
        del(nodes)
        
# #%%
    

# plt.plot(accs,linewidth=2,label="Total Performance")
# # plt.plot(class_accs,'--',label=key_list) 


# plt.title(f"{len(key_list)} Class 9 Pixel Classifier")
# plt.xlabel("Epoch")
# plt.ylabel("Prediction Accuracy")
# plt.ylim(0,1)
# plt.legend()
# plt.show()


# #%%

# n_idx = 0
# for n_idx in range(len(nodes)):
#     node = nodes[n_idx]
#     # print(node.neuron.dend_soma.update_traj)
#     plt.figure(figsize=(8,4))
#     for dend in node.dendrite_list:
#         if 'ref' not in dend.name:

#             if 'lay1' in dend.name:
#                 plt.plot(dend.update_traj,'--', label=dend.name)
#             elif 'lay2' in dend.name:
#                 plt.plot(dend.update_traj,':', label=dend.name)
#             else:
#                 plt.plot(dend.update_traj, linewidth=2, label=dend.name)
#     plt.title(f"{node.name} Update Trajectories")
#     plt.legend(loc=(1.05,0.05))
#     plt.show()


