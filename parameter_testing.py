from neural_zoo import NeuralZoo


from params import default_neuron_params
from soen_functions import phi_thresholds
from super_input import SuperInput
from soen_sim import input_signal, synapse, neuron, network
# from soen_plotting import raster_plot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("pastel")


from soen_utilities import dend_load_arrays_thresholds_saturations
ib__list, phi_r__array, i_di__array, r_fq__array, phi_th_plus__vec, phi_th_minus__vec, s_max_plus__vec, s_max_minus__vec, s_max_plus__array, s_max_minus__array = dend_load_arrays_thresholds_saturations('default_ri')

times = np.arange(0,500,100)
indices = np.zeros(len(times)).astype(int)
def_spikes = [indices,times]
input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes, duration=500)



# plt.figure(figsize=(14,4))
# dts = np.array([.0001,.001,.01,.1,1,5,10,15,20]) #.0001,.001,.01,
# # dts = np.array([.1,20])
# for dt in dts:
#     default_neuron_params['s_th'] = 100
#     mono_point = NeuralZoo(type='mono_point',**default_neuron_params)
#     mono_point.synapses[0][0][0][0].add_input(input.signals[0])
#     net = network(sim=True,dt=dt,tf=500,nodes=[mono_point])
#     signal = mono_point.dendrites[0][0][0].s
#     phi_r = mono_point.dendrites[0][0][0].phi_r
#     # plt.plot(net.t,signal, label=f'dt={dt}')
#     plt.plot(net.t,phi_r, label=f'phi_r, dt = {dt}')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.subplots_adjust(right=.8)
# plt.title("Recieved Flux at soma for unform spike train with variable dt")
# plt.show()


default_neuron_params['s_th'] = 10
mono_point = NeuralZoo(type='mono_point',**default_neuron_params)
mono_point.synapses[0][0][0][0].add_input(input.signals[0])
net = network(sim=True,dt=.01,tf=500,nodes=[mono_point])
# mono_point.plot_neuron_activity(net,input=input,title="Monosynaptic Point Neuron")
# print(mono_point.neuron.__dict__.keys())
# print(mono_point.neuron.dend__nr_ni.__dict__.keys())
# _ind_ib = ( np.abs( ib__list[:] - mono_point.neuron.dend__nr_ni.ib ) ).argmin()
# print([phi_th_minus__vec[_ind_ib],phi_th_plus__vec[_ind_ib]])
print(phi_thresholds(mono_point.neuron))
# beta_factor = np.arange(1,6,1)
# tau_nis = np.arange(100,1100,100)
# # tau_nis = np.arange(0,10,1)
# ranges = [2*np.pi*(10**beta_factor),tau_nis]
# names = ['beta_ni','tau_ni']


beta_factor = np.arange(1,6,1)
neuron_biases = np.arange(1,2,.2)
ranges = [2*np.pi*(10**beta_factor),neuron_biases]
names = ['beta_ni','ib_n']


# neuron_biases = np.ones(len(dts))
# ranges = [dts,neuron_biases]
# names = ['dt','ib_n']

params = default_neuron_params
type='mono_point'

def sweeper(type,params,names,ranges,input):
    # plt.figure(figsize=(12,6))
    fig, axs = plt.subplots(len(ranges[0]),  2,figsize=(18,6))
    params['s_th'] = 1
    for i,r in enumerate(ranges[0]):
        params[names[0]] = r
        for j,rr in enumerate(ranges[1]):
            params[names[1]] = rr
            neuron = NeuralZoo(type=type,**params)
            neuron.synapses[0][0][0][0].add_input(input.signals[0])
            net = network(sim=True,dt=.1,tf=500,nodes=[neuron])
            signal = neuron.dendrites[0][0][0].s
            phi_r = neuron.dendrites[0][0][0].phi_r
            axs[i,0].plot(net.t,signal, label=f'soma signal, {names[1]} = {np.round(rr,2)}')
            axs[i,1].plot(net.t,phi_r, label=f'phi_r, {names[1]} = {np.round(rr,2)}')
        if i != len(ranges[0]):
            axs[i,0].tick_params(
                axis='x',          
                which='both',      
                bottom=False,     
                top=False,        
                labelbottom=False)
        if i != len(ranges[0]):
            axs[i,1].tick_params(
                axis='x',          
                which='both',      
                bottom=False,     
                top=False,        
                labelbottom=False)
    plt.legend(loc='center left', bbox_to_anchor=(1, 3))
    plt.subplots_adjust(right=.75)
    rounded = []
    for i in ranges[0]:
        rounded.append(np.format_float_scientific(i, unique=False, precision=3))
    axs[0,0].set_title(f"{names[0]} Plots for Different Values of {names[1]}\n {names[0]} = {rounded}") 
    plt.show()

# sweeper(type,params,names,ranges,input)



'''
Sweep over phi_r by driving flux with varying amounts of gradualness
    - Find way to represent rollover in terms of signal behavior
'''



