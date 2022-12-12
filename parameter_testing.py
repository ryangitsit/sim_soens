from neural_zoo import NeuralZoo


from params import default_neuron_params
from super_input import SuperInput
from soen_sim import input_signal, synapse, neuron, network
# from soen_plotting import raster_plot
import numpy as np
import matplotlib.pyplot as plt

times = np.arange(0,500,100)
indices = np.zeros(len(times)).astype(int)
def_spikes = [indices,times]
input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes, duration=150)


# default_neuron_params['s_th'] = 0.5
# mono_point = NeuralZoo(type='mono_point',**default_neuron_params)
# mono_point.synapses[0][0][0][0].add_input(input.signals[0])
# net = network(sim=True,dt=.01,tf=500,nodes=[mono_point])
# mono_point.plot_neuron_activity(net,input=input,title="Monosynaptic Point Neuron")

beta_factor = np.arange(1,10,1)
neuron_biases = np.arange(1,2,.2)*50

ranges = [2*np.pi*(10**beta_factor),neuron_biases]
names = ['beta_ni','tau_ni']
params = default_neuron_params
type='mono_point'

def sweeper(type,params,names,ranges,input):
    # plt.figure(figsize=(12,10))
    fig, axs = plt.subplots(len(ranges[0]),  1,figsize=(12,10))
    params['s_th'] = 1
    for i,r in enumerate(ranges[0]):
        params[names[0]] = r
        for j,rr in enumerate(ranges[1]):
            params[names[1]] = rr
            neuron = NeuralZoo(type=type,**params)
            neuron.synapses[0][0][0][0].add_input(input.signals[0])
            net = network(sim=True,dt=.1,tf=500,nodes=[neuron])
            signal = neuron.dendrites[0][0][0].s
            axs[i].plot(net.t,signal, label=f'soma signal, beta factor')
        if i != len(ranges[0]):
            axs[i].tick_params(
                axis='x',          
                which='both',      
                bottom=False,     
                top=False,        
                labelbottom=False) 
    plt.show()

sweeper(type,params,names,ranges,input)



'''
Sweep over phi_r by driving flux with varying amounts of gradualness
    - Find way to represent rollover in terms of signal behavior
'''



