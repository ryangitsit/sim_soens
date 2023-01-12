import numpy as np
import matplotlib.pyplot as plt

from soen_sim import input_signal, network

from super_library import NeuralZoo
from super_input import SuperInput

from soen_plotting import raster_plot

def main():

    z = np.array([0,1,4,7,8]) # z-pixel array
    v = np.array([1,4,3,6,8])-1 # v
    n = np.array([2,4,6,7,9])-1 # n
    letters = [z,v,n]

    indices = np.concatenate(letters)
    times = np.concatenate([np.ones(len(z))*20,
                           np.ones(len(v))*120,
                           np.ones(len(n))*220])
    def_spikes = [indices,times]
    input = SuperInput(channels=9, type='defined', defined_spikes=def_spikes, duration=300)
    raster_plot(input.spike_arrays)

    params= {
        "N":4,
        "s_th":.5,
        "ib":1.7,
        "ib_n":1.7,
        }

    # np.random.rand(3)
    W1 = [
        [[.5,.5,.5]],
        [[.5,.5,.5],[.5,.5,.5],[.5,.5,.5]]
        ]
    W2 = [
        [[.5,.5,.5]],
        [[.5,.5,.5],[.5,.5,.5],[.5,.5,.5]]
        ]
    W3 = [
        [[.5,.5,.5]],
        [[.5,.5,.5],[.5,.5,.5],[.5,.5,.5]]
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

    net = network(dt=0.1,tf=300,nodes=neurons)
    net.simulate()
    raster_plot(net.spikes)
    
if __name__=='__main__':
    main()