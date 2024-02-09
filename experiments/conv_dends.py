#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../sim_soens')

# from super_library import NeuralZoo
from sim_soens.super_input import SuperInput
from sim_soens.soen_components import network
from sim_soens.soen_plotting import activity_plot, arbor_activity, structure,raster_plot
from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *

from sim_soens.super_functions import *

from sim_soens.neuron_library import *
from sim_soens.network_library import *


'''
Plan
 - 
'''

#%%


# dataset = picklin("datasets/MNIST/","duration=5000_slowdown=100")
from keras.datasets import mnist

def single_kernel(img,coordinates):
    (x1,x2),(y1,y2) = coordinates
    kernel = img[x1:x2,y1:y2] #.transpose()
    return kernel

def get_coordinates(x,y,size):
    return (x,x+size[0]),(y,y+size[1])

def make_row(img,size,y,kern):
    x_axis = len(img[0])
    row = [single_kernel(img, get_coordinates(i,y,size))*kern for i in range(x_axis-size[0])]
    return row

def get_kern(kernel):
    if kernel == 'vertical':
        kern = np.array([
        [0,1,0],
        [0,1,0],
        [0,1,0]
        ])
    elif kernel == 'horizontal':
        kern = np.array([
        [0,0,0],
        [1,1,1],
        [0,0,0]
        ])
    elif kernel == 'up':
        kern = np.array([
        [0,0,1],
        [0,1,0],
        [1,0,0]
        ])
    elif kernel == 'down':
        kern = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
        ])
    else:
        kern = np.ones((3,3))
    return kern

def kernelize(img,size,kernel=None):

    kern = get_kern(kernel)

    y_axis = len(img)
    all_rows = [make_row(img,size,j,kern) for j in range(y_axis-size[1])] #[::-1]
    return np.rot90(np.array(all_rows))[::-1]

def kerns_to_img(kernels):
    side = kernels.shape[0]*kernels.shape[2]
    kern_img = [[] for _ in range(side)]
    for i,row in enumerate(kernels):
        for j,kern in enumerate(row):
            for k in range(kernels.shape[2]):
                kern_img[i*kernels.shape[3]+k].extend(kern[k])
    return kern_img

def plot_kernels(kernels):
    s1=25
    s2=25
    fig, axs = plt.subplots(s1,s2,figsize=(14,14), sharex=True, sharey=True)
    print("here")
    fig.subplots_adjust(wspace=0,hspace=0)
    for i,row in enumerate(kernels):
        for j,kern in enumerate(row):
            if i<s1 and j < s2:
                print(i,j)
                axs[i][j].imshow(kern)
                axs[i][j].set_xticklabels([])
                axs[i][j].set_yticklabels([])
    plt.title("Kernelization of The MNIST Image \"0\"",fontsize=24)
    plt.show()

# plot_kernels(kernels)


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X = (X_train[(y_train == 0)][0])
print(X.shape)
# plt.imshow(X)
# plt.show()

all_kernels = [None,'vertical','horizontal','up','down']

fig, axs = plt.subplots(1,len(all_kernels),figsize=(14,14), sharex=True, sharey=True)
for i,kern in enumerate(all_kernels):
    kernels = kernelize(X,(3,3),kernel=kern)
    kern_img = kerns_to_img(kernels)
    axs[i].imshow(kern_img)
    axs[i].set_title(f"{kern} Kernel",fontsize=16)
plt.show()

#%%

duration = 100
# patterns = {
#     "vertical": 
#         [0,1,0,
#          0,1,0,
#          0,1,0],
#     "horizontal": 
#         [0,0,0,
#          1,1,1,
#          0,0,0],
# }

# # inpt_spikes = pixels_to_spikes(patterns["vertical"],[20])
# inpt_spikes = pixels_to_spikes(patterns["horizontal"],[20])

# inpt = SuperInput(
#     type           ='defined',
#     defined_spikes = inpt_spikes, 
#     duration       = duration,
#     channels       = 9
#     )

# raster_plot(inpt.spike_arrays)

letters = make_letters(patterns='all')
plot_letters(letters)
print(type({'z':letters['z']}))
plot_letters(letters,'x')
inputs = make_inputs(letters,20)





#%%

vert_positive = [
    [
        [.3,.3,.3]
        ],
    [
        [0.0,1.0,0.0],
        [0.0,1.0,0.0],
        [0.0,1.0,0.0]
        ]
]

vert_symmetric = [

    [
        [.3,.3,.3]
        ],
    [
        [-.25,1.0,-0.25],
        [-.25,1.0,-0.25],
        [-.25,1.0,-0.25]
        ]
]

heavy_neg = [
    [
        [-1.0,-1.0,-1.0]
        ],
    [
        [-1.0,-1.0,-1.0,],
        [-1.0,-1.0,-1.0,],
        [-1.0,-1.0,-1.0,]
        ]
]

heavy = [
    [
        [1.0,1.0,1.0]
        ],
    [
        [1.0,1.0,1.0,],
        [1.0,1.0,1.0,],
        [1.0,1.0,1.0,]
        ]
]

kernel_node_vertical = SuperNode(
    name = 'vert_positive',
    weights=heavy,
    beta_di=2*np.pi*1e3,beta_ni=2*np.pi*1e3,
    normalize_input_connection_strengths=False
    )
# kernel_node_vertical.parameter_print()
# kernel_node_vertical.normalize_fanin(1)

@timer_func
def normfan(node,buffer=0,verbose=False):
    
    for dendrite in node.dendrite_list:
        if len(dendrite.dendritic_connection_strengths) > 0:  

            if verbose==True: print(f"{dendrite.name} =>  phi_th = {dendrite.phi_th} :: max_s = {node.max_s_finder(dendrite)}")
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
                if verbose==True: print(f"  {in_name} -> {cs}") 

                if cs<0:
                    if verbose==True: print(cs)
                    negatives.append(cs)
                    neg_max.append(cs*max_in)
                    neg_dends.append(in_dend)

                elif cs>0:
                    positives.append(cs)
                    pos_max.append(cs*max_in)
                    pos_dends.append(in_dend)
        

            if sum(pos_max) > max_phi:
                if verbose==True: print(f" Normalizing input to {dendrite.name} from {sum(pos_max)} to {max_phi}")
                for pos_dend in pos_dends:
                    cs = dendrite.dendritic_connection_strengths[pos_dend.name]
                    cs_max = cs*node.max_s_finder(pos_dend)
                    cs_proportion = cs_max/sum(pos_max)
                    cs_normalized = max_phi*cs_proportion/node.max_s_finder(pos_dend) 
                    if verbose==True: print(f"   {pos_dend} -> {cs_normalized}")
                    dendrite.dendritic_connection_strengths[pos_dend.name] = cs_normalized
            if verbose==True:print(sum(np.abs(neg_max)))

            if sum(np.abs(neg_max)) > max_phi:
                if verbose==True: print(f" Normalizing negative input to {dendrite.name} from {sum(neg_max)} to {max_phi}")
                for neg_dend in neg_dends:
                    cs = np.abs(dendrite.dendritic_connection_strengths[neg_dend.name])
                    cs_max = cs*node.max_s_finder(neg_dend)
                    cs_proportion = cs_max/sum(np.abs(neg_max))
                    cs_normalized = max_phi*cs_proportion/node.max_s_finder(neg_dend)*-1
                    if verbose==True: print(f"   {neg_dend} -> {cs_normalized}")
                    dendrite.dendritic_connection_strengths[neg_dend.name] = cs_normalized

            if verbose==True:print("\n")
    return node

# kernel_node_vertical = normfan(kernel_node_vertical,buffer=0)

#%%
def steady_input(node,inpt):
    for i,syn in enumerate(node.synapse_list):
        # print(inpt.signals[i].spike_times)
        # print(syn.__dict__.keys())

        # print(dend.name, inpt.signals[i].spike_times)
        if 'ref' not in syn.name: # and len(inpt.signals[i].spike_times) > 0:

            for dend in node.dendrite_list:
                if syn.name in list(dend.synaptic_inputs.keys()):
                    dend.offset_flux = 0.5 #+ dend.phi_th#- dend.phi_th


                    # print(dend.name,dend.phi_th)

    return node

# kernel_node_vertical.plot_structure()

#%%
for k,v in inputs.items():


    if k=="|": #1==1: #k=="|": #
        print(k,letters[k])

        kernel_node_vertical = SuperNode(
            name = 'vert_symmetric',
            weights=heavy,
            beta_di=2*np.pi*1e3,beta_ni=2*np.pi*1e3,
            normalize_input_connection_strengths=False
            )
        kernel_node_vertical = normfan(kernel_node_vertical,buffer=0,verbose=True)

        plot_letters(letters,k)

        # raster_plot(v.spike_arrays)
        # kernel_node_vertical.one_to_one(v)



        kernel_node_vertical = steady_input(kernel_node_vertical,v)

        net = network(
            sim     = True,
            nodes   = [kernel_node_vertical],
            tf      = duration+500,
            dt      = 1.0,
            backend = 'julia'

        )

        kernel_node_vertical.plot_arbor_activity(net,phir=True)
        kernel_node_vertical.plot_neuron_activity(phir=True,ref=True)

        # del(net)
        # del(kernel_node_vertical)

#%%
for dend in kernel_node_vertical.dendrite_list:
    if 'lay2' in dend.name:
        plt.plot(dend.phi_r,'--',label=dend.name)
        plt.plot(dend.s)
        # print(dend.name)
plt.ylim(0,1)
plt.legend()
plt.show()


#%%

# node = SuperNode()
# node.parameter_print()

offsets = np.arange(.1,2.01,.1)
# offsets = [0.7]
nodes = []
for i,off in enumerate(offsets):
    off = np.round(off,2)
    node = SuperNode(name=f'node_{off}',beta_di=2*np.pi*1e3,beta_ni=2*np.pi*1e3,beta=2*np.pi*1e3,s_th=1)
    node.neuron.dend_soma.offset_flux = off
    nodes.append(node)

net = network(sim=True,nodes=nodes,tf=500,dt=1.0,backend='python')
maxes = []
plt.figure(figsize=(8,6))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i,node in enumerate(nodes):
    dend = node.neuron.dend_soma
    maxes.append(np.max(dend.s))
    plt.plot(dend.s,label=node.name,color=colors[i%len(colors)])
    plt.plot(dend.phi_r,'--',color=colors[i%len(colors)])
plt.legend()
plt.show()

plt.plot(offsets,maxes)
plt.show()