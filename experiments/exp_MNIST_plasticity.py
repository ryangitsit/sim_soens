import numpy as np
import matplotlib.pyplot as plt

# Import writer class from csv module
from csv import writer

import sys
sys.path.append('../sim_soens')
sys.path.append('../')

from sim_soens.soen_plotting import raster_plot, activity_plot
from sim_soens.super_input import SuperInput
from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *
from sim_soens.soen_sim import network, dendrite, HardwareInTheLoop, synapse

import time
np.random.seed(10)
print(np.random.randint(0, 100, 10))


def make_dataset(digits,samples,slowdown,duration):
    dataset = [[] for _ in range(digits)]
    fig, axs = plt.subplots(digits,samples,figsize=(36,12))
    for i in range(digits):
        for j in range(samples):
            input_MNIST = SuperInput(
                type='MNIST',
                index=i,
                sample=j,
                slow_down=slowdown,
                duration=duration
                )
            spikes = input_MNIST.spike_arrays
            dataset[i].append([spikes[0],spikes[1]])

            axs[i][j].plot(spikes[1],spikes[0],'.k',ms=.5)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
    picklit(
        dataset,
        "datasets/MNIST/",
        f"duration={duration}_slowdown={slowdown}"
        )
    plt.show()
    
#     dataset = np.array(dataset)
#     plt.plot(dataset[0][2][1],dataset[0][2][0],'.k')# ,ms=.5)
#     plt.show()



# make_dataset(10,100,100,5000)

dataset = picklin("datasets/MNIST/","duration=5000_slowdown=100")
# plt.plot(dataset[0][2][1],dataset[0][2][0],'.k')# ,ms=.5)
# plt.show()
# raster_plot(dataset[0][2])


start = time.perf_counter()
np.random.seed(10)
f_idx = 28
lay_2 = [np.random.rand(f_idx)*21/28 for _ in range(f_idx)]
weights = [
    [np.random.rand(f_idx)],
    lay_2
]

node_zero = SuperNode(name='node_zero',weights=weights,tau_di=5)
node_one  = SuperNode(name='node_one',weights=weights,tau_di=5)
node_two  = SuperNode(name='node_two',weights=weights,tau_di=5)

nodes=[node_zero,node_one,node_two]

inhibition = [-1.2,-.5,-1.2]
for i,node in enumerate(nodes):
    syn_soma = synapse(name=f'{node.name}_somatic_synapse')
    node.synapse_list.append(syn_soma)
    node.neuron.dend_soma.add_input(syn_soma,connection_strength=inhibition[i])
for i,node in enumerate(nodes):
    for other_node in nodes:
        if other_node.name != node.name:
            node.neuron.add_output(other_node.synapse_list[-1])
            print(other_node.synapse_list[-1].name)

finish = time.perf_counter()
print("Time to make neurons: ", finish-start)

desired = [
    [30,0,0],
    [0,30,0],
    [0,0,30],
]
backend = 'julia'
name = backend
print(backend)

run_times = []
init_times = []

for run in range(10000):
    # if run%10==0: 
    print("Run: ",run)
    samples_passed=0
    for sample in range(10):
        
        total_errors = [[] for i in range(3)]
        outputs = [[] for i in range(3)]
        for digit in range(3):
            
            start = time.perf_counter()
            input_ = SuperInput(type="defined",channels=784,defined_spikes=dataset[digit][sample])
            
            for node in nodes:
                for i,channel in enumerate(input_.signals):
                    node.synapse_list[i].add_input(channel)
                # node.one_to_one(input)
            f0 = time.perf_counter()
            # print("Input time: ", f0-start)

            net = network(sim=True,dt=.1,tf=500,nodes=nodes,backend=backend,print_times=True)
            run_times.append(net.run_time)
            init_times.append(net.init_time)
            spikes = array_to_rows(net.spikes,3)

            error_zero = desired[0][digit] - len(spikes[0])
            error_one  = desired[1][digit] - len(spikes[1])
            error_two  = desired[2][digit] - len(spikes[2])
            
            errors = [error_zero,error_one,error_two]
            

            total_errors[0] += np.abs(error_zero)
            total_errors[1] += np.abs(error_one)
            total_errors[2] += np.abs(error_two)

            output = [len(spikes[0]),len(spikes[1]),len(spikes[2])]
            outputs[digit].append(output)

            # spike_trajectories[i].append(len(out_spikes))
            nodes[0].neuron.spike_times=[]
            nodes[1].neuron.spike_times=[]
            nodes[2].neuron.spike_times=[]

            nodes[0].neuron.spike_indices=[]
            nodes[1].neuron.spike_indices=[]
            nodes[2].neuron.spike_indices=[]

            nodes[0].neuron.electroluminescence_cumulative_vec=[]
            nodes[1].neuron.electroluminescence_cumulative_vec=[]
            nodes[2].neuron.electroluminescence_cumulative_vec=[]

            nodes[0].neuron.time_params=[]
            nodes[1].neuron.time_params=[]
            nodes[2].neuron.time_params=[]

            s = time.perf_counter()
            # offsets = {}
            if run%10 != 0 and run!=0:
                for n,node in enumerate(nodes):

                    # spike refractory dendrite
                    # lst = node.dendrite_list[2:]
                    # lst.insert(0,node.dendrite_list[0])

                    # for dend in lst:
                        # step = errors[n]*np.mean(dend.s)*.0001
                        # dend.offset_flux += step

                    for l,layer in enumerate(node.dendrites):
                        for g,group in enumerate(layer):
                            for d,dend in enumerate(group):
                                step = errors[n]*np.mean(dend.s)*.0001 #+(2-l)*.001
                                flux = np.mean(dend.phi_r) + step #dend.offset_flux
                                if flux > 0.5 or flux < -0.5:
                                    step = -step
                                dend.offset_flux += step
                                dend.s = []
                                dend.phi_r = []

                                # if g==0 and d ==0: print("learning rate =", .0001+(2-l)*.001)
                        # offsets[dend.name] = dend.offset_flux

            f = time.perf_counter()
            # print("Update time: ", f-s)
            # print("Total runtime", f-start)

            # print(f"Sample = {sample} \n Digit = {digit}\n  Spikes = {output} \n  Error = {errors} \n  prediction = {np.argmax(output)}")
            print(f"  {sample}  -  [{digit} -> {np.argmax(output)}]  --  {f-start} ")
            # List that we want to add as a new row
            List = [sample,digit,output,errors,np.argmax(output),f-start,net.init_time,net.run_time]
            
            # Open our existing CSV file in append mode
            # Create a file object for this file
            with open(f'MNIST_ongoing_{name}.csv', 'a') as f_object:
            
                writer_object = writer(f_object)
            
                writer_object.writerow(List)

                f_object.close()

            del(net)
            del(input_)
            if np.argmax(output) == digit:
                samples_passed+=1

    print(f"samples passed: {samples_passed}/30\n\n")

    picklit(
        nodes,
        f"results/MNIST_WTA_{name}/",
        f"nodes_at_{run}"
        )
    
    if run == 0:
        picklit(
            weights,
            f"results/MNIST_WTA_{name}/",
            f"init_weights"
            )
        
    if samples_passed == 30:
        print("converged!\n\n")
        break

    # if total_error<25:
    #     break
            


# MNIST_node.plot_neuron_activity(net,dend=False,phir=True,ref=True)
# MNIST_node.plot_arbor_activity(net,size=(36,14))