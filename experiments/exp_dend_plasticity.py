import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.soen_plotting import raster_plot, activity_plot

from sim_soens.super_input import SuperInput
from sim_soens.neuron_library import NeuralZoo
from sim_soens.super_functions import *
from sim_soens.soen_components import network, dendrite, HardwareInTheLoop


def main():
    '''
    Dendritic Plasticity Demonstration
    '''
    # define two 2-d input patterns via rates
    rates_1 = [51,51,51,95,95,95,50,50,50]
    rates_2 = [60,60,60,120,120,120,60,60,60]
    correct = [2,2,1,1,2,1,1]

    # define spiking expectations for previoud interval
    expect = [
        [0,50],
        [None,None],
        [None,None],
        [50,0],
        [None,None],
        [None,None],
        [None,None],
        [None,None],
        [None,None]]

    r1 = rates_1
    r2 = rates_2

    # r1 = []
    # r2 = []
    # expect = []
    # for i in range(10):
    #     r1 += rates_1
    #     r2 += rates_2
    #     expect += expect_

    interval = 500
    duration = interval*(len(r1))


    # convert input object
    times_1 = []
    times_2 = []

    for i in range(len(r1)):
        times_1.append(np.arange(i*interval,i*interval+interval,r1[i]))
        times_2.append(np.arange(i*interval,i*interval+interval,r2[i]))
    times_1 = np.concatenate(times_1)
    times_2 = np.concatenate(times_2)

    indices = np.concatenate([np.zeros(len(times_1)).astype(int),np.ones(len(times_2)).astype(int)])
    times = np.concatenate([times_1,times_2])
    def_spikes = [indices,times]
    input_obj = SuperInput(channels=2, type='defined', defined_spikes=def_spikes, duration=duration)


    # raster_plot(input.spike_arrays)

    # parameters for plasticity neurons
    plast_dict = {
        "trace_factor" :.2,
        "s_th" :0.5,
        "trace_syn_factor" :1,
        "tau_di" :250,
        "soma_factor" :.1,
        "trace_tau" : 5000
    }

    # parameters for hardware in the loop
    HW_dict = {
        "baseline" : 1.2,
        "interval":500,
        "freq_factor" :10,
    }

    # weights for either neurons' forward dendrites
    WA = [[[.6,.5]]]
    WB = [[[.4,.5]]]

    # create plasticity neurons
    nA = NeuralZoo(type='plastic_neuron',weights=WA,n_count=1,**plast_dict,input_obj=input_obj)
    nB = NeuralZoo(type='plastic_neuron',weights=WB,n_count=2,**plast_dict,input_obj=input_obj)

    nodes=[nA,nB]

    plasticity=True
    # plasticity=False

    # only add hardware to network if plasticity is true
    if plasticity==True:
        title="Error Module Engaging Plasticity at t=500ns (Neuron 2 Correct Output)"
        HW = HardwareInTheLoop(expect=expect,**HW_dict) #freq_factor=freq_factor,interval=interval,expect=expect,baseline=baseline)
        # print(HW.__dict__)
    else:
        title="No Plasiticity (Neuron 2 Correct Output)"
        HW = None

    # create and run netword
    net = network(sim=True,dt=.1,tf=duration,nodes=nodes,null_synapses=True,new_way=True,hardware=HW)

    subtitles= ["Neuron 1","Neuron 2"]
    activity_plot(nodes,net,title=title,subtitles=subtitles,input=input_obj, size=(16,6),phir=True)

    # plot change in bias for the forward dendrites of either neuron
    fig, axs = plt.subplots(2, 1,figsize=(12,6))
    for i,trace in enumerate(nA.dendrite_list):
        if "plus" not in trace.name and "minus" not in trace.name:
            if "ref" not in trace.name and "soma" not in trace.name:
                # axs[0].plot(net.t,trace.phi_r,'--',label="phi "+trace.name)
                axs[0].plot(net.t,trace.bias_dynamics, label = trace.name)
    axs[0].set_title("Neuron 1")
    for i,trace in enumerate(nB.dendrite_list):
        if "plus" not in trace.name and "minus" not in trace.name:
            if "ref" not in trace.name and "soma" not in trace.name:
                # axs[1].plot(net.t,trace.phi_r,'--',label="phi "+trace.name)
                axs[1].plot(net.t,trace.bias_dynamics, label = trace.name)
    axs[0].set_title("Neuron 1")
    axs[1].set_title("Neuron 2")
    fig.suptitle("Change in Biases of Forward Dendrites",fontsize=18)
    plt.legend()
    plt.show(block=False)

    # plot trace dendrite signals of either neuron
    fig, axs = plt.subplots(2, 1,figsize=(12,6))
    for i,trace in enumerate(nA.trace_dendrites):
        # axs[0].plot(net.t,trace.phi_r,'--',label="phi "+trace.name)
        axs[0].plot(net.t,trace.s, label = trace.name)
    axs[0].set_title("Neuron 1")

    for i,trace in enumerate(nB.trace_dendrites):
        # axs[1].plot(net.t,trace.phi_r,'--',label="phi "+trace.name)
        axs[1].plot(net.t,trace.s, label = trace.name)
    axs[1].set_title("Neuron 2")
    fig.suptitle("Trace Dendrite Signals",fontsize=18)
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()

'''
Backend notes:
 - neuronA.source_type
    - qd, ec, delay_delta
    - line 383 -> argmin in forloop?
 - just couple 

 - Consequent of inactive trace dendrites on feed-forward dendrites?
'''
