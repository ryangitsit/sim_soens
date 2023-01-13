#%%
import numpy as np
import matplotlib.pyplot as plt

from soen_sim import input_signal, network

from super_library import NeuralZoo
from super_input import SuperInput
from super_functions import array_to_rows

from soen_plotting import raster_plot

def main():
#%%
    eurekas=0
    runs = 10
    for run in range(runs):
        z = np.array([0,1,4,7,8]) # z-pixel np.array
        v = np.array([1,4,3,6,8])-1 # v
        n = np.array([2,4,6,7,9])-1 # n
        letters = [z,v,n]

        window = 100

        indices = np.concatenate(letters)
        times = np.concatenate([np.ones(len(z)),
                                np.ones(len(v))*(2+window),
                                np.ones(len(n))*(2+window)])
        def_spikes = [indices,times]
        input = SuperInput(channels=9, type='defined', defined_spikes=def_spikes, duration=window*3)
        # raster_plot(input.spike_np.arrays)

        params= {
            "N":4,
            "s_th":.4,
            "ib":1.7,
            "ib_n":1.7,
            }

        W1 = [
            [np.random.rand(3)],
            [np.random.rand(3),np.random.rand(3),np.random.rand(3)]
            ]
        W2 = [
            [np.random.rand(3)],
            [np.random.rand(3),np.random.rand(3),np.random.rand(3)]
            ]
        W3 = [
            [np.random.rand(3)],
            [np.random.rand(3),np.random.rand(3),np.random.rand(3)]
            ]

        n_1 = NeuralZoo(type="custom",weights=W1,**params) 
        n_2 = NeuralZoo(type="custom",weights=W2,**params) 
        n_3 = NeuralZoo(type="custom",weights=W3,**params) 

        n_1.synaptic_layer()
        n_2.synaptic_layer()
        n_3.synaptic_layer()

        neurons = [n_1,n_2,n_3]
        for i in range(len(input.spike_rows)):
            for n in neurons:
                n.synapse_list[i].add_input(input_signal(name = 'input_synaptic_drive', 
                input_temporal_form = 'arbitrary_spike_train', spike_times = input.spike_rows[i]))

        net = network(dt=0.1,tf=window*3,nodes=neurons)
        net.simulate()

        # raster_plot(net.spikes)
        # for i in range(3):
        #     plt.plot(net.t,net.signal[i])
        #     plt.plot(net.t,net.phi_r[i])
        # plt.show()
        # neurons[0].arbor_activity_plot()

        rows = array_to_rows(net.spikes,3)
        counts = [ [] for _ in range(len(rows))]

        for i in range(len(neurons)):
            for j in range(len(letters)):
                winset = [j*window,j*window+window]
                # print(winset)
                frame = [rows[i][idx] for  idx,val in enumerate(rows[i]) 
                        if winset[0]<val<winset[1]]
                counts[i] .append(len(frame))
        counts = np.transpose(counts)
        maxes = [np.argmax(arr) for arr in counts]
        peaks = [np.max(arr) for arr in counts]
        if len(set(maxes))==3 and np.min(peaks)>0:
            print("EUREKA!")
            # print(counts)
            print("W1 = ",W1,"\nW2 = ",W2,"\nW3 = ",W3,"\n")
            # print(net.spikes)
            # raster_plot(net.spikes)
            eurekas+=1
            print("-----------------------------------------\n\n")
        else:
            print(f"Trying again --> attempt {run}")
    print(f"Percent natural success: {eurekas}/{runs} = {eurekas/runs}")

    
if __name__=='__main__':
    main()

