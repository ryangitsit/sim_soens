
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from sim_soens.soen_plotting import raster_plot, activity_plot
from sim_soens.super_input import SuperInput
from sim_soens.super_library import NeuralZoo
from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *
from sim_soens.soen_sim import network, dendrite, HardwareInTheLoop

print(np.random.randint(0, 100, 10))
np.random.seed(None)


# non-noisy nine-pixel letters
z = [1,1,0,
     0,1,0,
     0,1,1]

v = [1,0,1,
     1,0,1,
     0,1,0]

n = [0,1,0,
     1,0,1,
     1,0,1]

letters = [z,v,n]

# convert to indices of nine-channel input
z_idx = np.where(np.array(z)==1)[0]
v_idx = np.where(np.array(v)==1)[0]
n_idx = np.where(np.array(n)==1)[0]

# all non-zero indices will spike at spike_time
spike_time = 20
spike_times = (np.ones(len(z_idx))*spike_time).astype(int)


def one_pixel_noise(letter):
    noise_idx = np.random.randint(0,len(letter))
    if letter[noise_idx] == 1:
        letter[noise_idx] = 0
    else:
        letter[noise_idx] = 1
    return letter

n = one_pixel_noise(n)
let_idxs = [z_idx,v_idx,n_idx]
np.random.seed(10)

# print(np.random.randint(0, 100, 10))
def plot_letter(letter):
    import matplotlib.cm as cm
    arrays = [[] for i in range(3)]
    count = 0
    for col in range(3):
        for row in range(3):
            arrays[col].append(letter[count])
            count+=1

    pixels = np.array(arrays).reshape(3,3)
    plt.figure()
    plt.title("pixel_plot")
    plt.xticks([])
    plt.yticks([])
    pixel_plot = plt.imshow(
        pixels,
        interpolation='nearest',
        cmap=cm.Blues
        )
    
    plt.show()


for let in letters:
    plot_letter(let)




weights = [
    [[.5,.5,.5]],
    [[.3,.3,.3],[.3,.3,.3],[.3,.3,.3]]
]

node_z = SuperNode(weights=weights)
# node_z.plot_structure()

correct_z = {'rand_neuron_77132_soma': -0.17018820352206204, 
             'rand_neuron_77132_lay1_branch0_den0': -0.1005137433783909, 
             'rand_neuron_77132_lay1_branch0_den1': -0.15793987522376604, 
             'rand_neuron_77132_lay1_branch0_den2': -0.09882142011047565, 
             'rand_neuron_77132_lay2_branch0_den0': -0.03885238533504326, 
             'rand_neuron_77132_lay2_branch0_den1': 0.1779857429759991, 
             'rand_neuron_77132_lay2_branch0_den2': -0.5318486717211699, 
             'rand_neuron_77132_lay2_branch1_den0': -0.5951666245254208, 
             'rand_neuron_77132_lay2_branch1_den1': 0.22541383358572367, 
             'rand_neuron_77132_lay2_branch1_den2': -0.5951666245254208, 
             'rand_neuron_77132_lay2_branch2_den0': -0.5366149403665759, 
             'rand_neuron_77132_lay2_branch2_den1': -0.03885238533504326, 
             'rand_neuron_77132_lay2_branch2_den2': 0.1779857429759991}

correct_v = {'rand_neuron_77132_soma': 0.044736378133675835, 
             'rand_neuron_77132_lay1_branch0_den0': 0.0982066950642038, 
             'rand_neuron_77132_lay1_branch0_den1': -0.0023132695052485467, 
             'rand_neuron_77132_lay1_branch0_den2': -0.08514403864197762, 
             'rand_neuron_77132_lay2_branch0_den0': 0.10824719509768488, 
             'rand_neuron_77132_lay2_branch0_den1': -0.24870807367155753, 
             'rand_neuron_77132_lay2_branch0_den2': 0.2661981451385887, 
             'rand_neuron_77132_lay2_branch1_den0': 0.023510432122448473, 
             'rand_neuron_77132_lay2_branch1_den1': -0.17347386222102582, 
             'rand_neuron_77132_lay2_branch1_den2': 0.023510432122448473, 
             'rand_neuron_77132_lay2_branch2_den0': -0.20381303305920312, 
             'rand_neuron_77132_lay2_branch2_den1': 0.10824719509768488, 
             'rand_neuron_77132_lay2_branch2_den2': -0.24870807367155753}

correct_n = {'rand_neuron_77132_soma': 0.04693267924276726, 
             'rand_neuron_77132_lay1_branch0_den0': -0.08684334793735499, 
             'rand_neuron_77132_lay1_branch0_den1': -0.006008094102381742, 
             'rand_neuron_77132_lay1_branch0_den2': 0.09389567597814484, 
             'rand_neuron_77132_lay2_branch0_den0': -0.2533917538265178, 
             'rand_neuron_77132_lay2_branch0_den1': 0.09256149661966233, 
             'rand_neuron_77132_lay2_branch0_den2': -0.20536055171250206, 
             'rand_neuron_77132_lay2_branch1_den0': 0.022896546042728123, 
             'rand_neuron_77132_lay2_branch1_den1': -0.18109626627015565, 
             'rand_neuron_77132_lay2_branch1_den2': 0.022896546042728123, 
             'rand_neuron_77132_lay2_branch2_den0': 0.26382593975165725, 
             'rand_neuron_77132_lay2_branch2_den1': -0.2533917538265178, 
             'rand_neuron_77132_lay2_branch2_den2': 0.09256149661966233}

names = ['z','v','n']
expect = [0,0,5]
spike_trajectories = [[] for i in range(len(letters))]
for run in range(300):
    if run%10==0:print(f"Run {run}:")
    total_error=0
    for i,let_idx in enumerate(let_idxs):
        defined_spikes=[let_idx,spike_times]
        input = SuperInput(type='defined',defined_spikes=defined_spikes)
        node_z.one_to_one(input)

        net = network(sim=True,dt=.1,tf=100,nodes=[node_z])
        out_spikes = net.spikes[1]

        error = expect[i] - len(out_spikes)
        total_error+=np.abs(error)
        spike_trajectories[i].append(len(out_spikes))
        node_z.neuron.spike_times=[]

        total_change = 0
        offsets = {}
        count = 0
        for dend in node_z.dendrite_list:
            if 'ref' not in dend.name:
        #         step = error*np.mean(dend.s)*.01
        #         total_change+=step
        #         dend.offset_flux += step
        #         offsets[dend.name] = dend.offset_flux

                dend.offset_flux = correct_n[list(correct_n.keys())[count]]
                count+=1
                # dend.offset_flux = correct_n[dend.name]

        # if run%10==0:
        #     print("  ",names[i],"spikes = ",len(out_spikes)," error = ",error,out_spikes,total_change)

        # print(names[i],"spikes = ",len(out_spikes)," error = ",error,out_spikes,total_change)
        # if i == 0:
        # node_z.plot_arbor_activity(net,phir=True)

    if total_error==0:
        print(f"Converged! (in {run} runs)")
        print(" ",names[i],"spikes = ",len(out_spikes)," error = ",error,out_spikes,total_change)
        print(" ",offsets)
        node_z.plot_arbor_activity(net)
        break

for i,spk_trj in enumerate(spike_trajectories):
    plt.plot(spk_trj,label=names[i])

plt.legend()
plt.show()