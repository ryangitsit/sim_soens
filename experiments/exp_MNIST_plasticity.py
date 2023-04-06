import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from sim_soens.soen_plotting import raster_plot, activity_plot
from sim_soens.super_input import SuperInput
from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *
from sim_soens.soen_sim import network, dendrite, HardwareInTheLoop, synapse


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


np.random.seed(10)
f_idx = 28
lay_2 = [np.random.rand(f_idx)*21/28 for _ in range(f_idx)]
weights = [
    [np.random.rand(f_idx)],
    lay_2
]
MNIST_node = SuperNode(weights=weights,tau_di=5)

expect = [10,20,30]
for run in range(10000):
    # if run%10==0: 
    print("Run: ",run)
    samples_passed=0
    for sample in range(100):
        
        errors = []
        outputs = []
        for digit in range(3):
            
            input = SuperInput(type="defined",channels=784,defined_spikes=dataset[digit][sample])
            # raster_plot(input.spike_arrays)


            
            MNIST_node.one_to_one(input)

            net = network(sim=True,dt=.1,tf=500,nodes=[MNIST_node],timer=False)
            out_spikes = net.spikes[1]
            output = len(out_spikes)
            outputs.append(output)
            MNIST_node.neuron.spike_times=[]


            error = expect[digit] - output
            errors.append(np.abs(error))

            offsets = {}
            for dend in MNIST_node.dendrite_list:
                if 'ref' not in dend.name:
                    step = error*np.mean(dend.s)*.001
                    dend.offset_flux += step
                    offsets[dend.name] = dend.offset_flux

        if errors[0]<5 and errors[1]<5 and errors[2]<5:
            samples_passed+=1

        if sample%10==0:
            print(" sample: ",sample)
            print("  ",outputs)


    print(f"samples passed: {samples_passed}/100")

    if samples_passed > 70:
        print("converged!")
        print("weights = ",weights)
        print("offsets = ",offsets)
        print(offsets)
        picklit(
            offsets,
            "results/MNIST/",
            f"converged_in_{run}"
            )
        picklit(
            weights,
            "results/MNIST/",
            f"init_weights"
            )
        break

    # if total_error<25:
    #     break
            


# MNIST_node.plot_neuron_activity(net,dend=False,phir=True,ref=True)
# MNIST_node.plot_arbor_activity(net,size=(36,14))