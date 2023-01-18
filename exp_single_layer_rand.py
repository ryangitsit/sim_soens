import numpy as np
import matplotlib.pyplot as plt

from soen_sim import input_signal, network

from super_library import NeuralZoo
from super_input import SuperInput
from super_functions import array_to_rows

from soen_plotting import raster_plot

def main():
    tile_time = 10
    classes = [0,1,2]
    window = tile_time*36*3
    input = SuperInput(type='saccade_MNIST', channels=36, tile_time=tile_time)
    # input = SuperInput(channels=36, type='random', total_spikes=3000, duration=500)    
    print("input generated")
    raster_plot(input.spike_arrays)
    params= {
        "N":4,
        "s_th":.3,
        "ib":1.7,
        "ib_n":1.7,
        # "tau_ni":5,
        # "tau_di":5,
        "tau_ref":75,
        }
    c=.85
    runs = 10
    for run in range(runs):
        W1 = [
            [np.random.rand(3)*c],
            [np.random.rand(3)*c,np.random.rand(3)*c,np.random.rand(3)*c],
            [np.random.rand(4)*c,np.random.rand(4)*c,np.random.rand(4)*c,
            np.random.rand(4)*c,np.random.rand(4)*c,np.random.rand(4)*c,
            np.random.rand(4)*c,np.random.rand(4)*c,np.random.rand(4)*c]
            ]
        W2 = [
            [np.random.rand(3)*c],
            [np.random.rand(3)*c,np.random.rand(3)*c,np.random.rand(3)*c],
            [np.random.rand(4)*c,np.random.rand(4)*c,np.random.rand(4)*c,
            np.random.rand(4)*c,np.random.rand(4)*c,np.random.rand(4)*c,
            np.random.rand(4)*c,np.random.rand(4)*c,np.random.rand(4)*c]
            ]
        W3 = [
            [np.random.rand(3)*c],
            [np.random.rand(3)*c,np.random.rand(3)*c,np.random.rand(3)*c],
            [np.random.rand(4)*c,np.random.rand(4)*c,np.random.rand(4)*c,
            np.random.rand(4)*c,np.random.rand(4)*c,np.random.rand(4)*c,
            np.random.rand(4)*c,np.random.rand(4)*c,np.random.rand(4)*c]
            ]

        n_1 = NeuralZoo(type="custom",weights=W1,**params) 
        n_2 = NeuralZoo(type="custom",weights=W2,**params) 
        n_3 = NeuralZoo(type="custom",weights=W3,**params) 

        n_1.synaptic_layer()
        n_2.synaptic_layer()
        n_3.synaptic_layer()

        neurons = [n_1,n_2,n_3]
        # for i in range(len(input.spike_rows)):
        for i in range(len(n_1.synapse_list)):
            for n in neurons:
                n.synapse_list[i].add_input(input.signals[i])
        print('tf = ',np.max(input.spike_arrays[1])+100)
        print('total input spikes = ', len(input.spike_arrays[1]))
        net = network(dt=0.1,tf=np.max(input.spike_arrays[1])+360,nodes=neurons)
        net.simulate()

        # neurons[0].plot_custom_structure()
        # neurons[0].arbor_activity_plot()
        # neurons[1].arbor_activity_plot()
        # neurons[2].arbor_activity_plot()
        # raster_plot(net.spikes)

        rows = array_to_rows(net.spikes,3)
        counts = [ [] for _ in range(len(rows))]

        for i in range(len(neurons)):
            for j in range(len(classes)):
                if i == len(neurons)-1 and j ==len(classes)-1:
                    winset = [j*window,j*window+window+36*tile_time]
                else:
                    winset = [j*window,j*window+window]
                # print(winset)
                frame = [rows[i][idx] for  idx,val in enumerate(rows[i]) 
                        if winset[0]<val<winset[1]]
                counts[i] .append(len(frame))
        counts = np.transpose(counts)
        # print(counts)
        maxes = [np.argmax(arr) for arr in counts]
        peaks = [np.max(arr) for arr in counts]
        if len(set(maxes))==3 and np.min(peaks)>0:
            print("EUREKA!")
            print(counts)
            print("W1 = ",W1,"\nW2 = ",W2,"\nW3 = ",W3,"\n")
            # print(net.spikes)
            # neurons[0].arbor_activity_plot()
            # neurons[1].arbor_activity_plot()
            # neurons[2].arbor_activity_plot()
            # raster_plot(net.spikes)
            eurekas+=1
            print("-----------------------------------------\n")
        else:
            print(f"Attempt {run+1} --> Try again, {len(rows[0]),len(rows[1]),len(rows[2])}")

# print(f"Percent natural success: {eurekas}/{runs} = {eurekas/runs}")
if __name__=='__main__':
    main()

