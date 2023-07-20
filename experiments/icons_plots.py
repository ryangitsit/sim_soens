import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../sim_soens')
sys.path.append('../')

from sim_soens.super_library import NeuralZoo
from sim_soens.super_input import SuperInput
# from sim_soens.params import default_neuron_params
from sim_soens.soen_sim import network
from sim_soens.soen_plotting import activity_plot
from sim_soens.super_node import SuperNode

import seaborn as sns
# colors = sns.color_palette('hls', 13)
# colors = sns.color_palette('seaborn-v0_8-muted', 13)

plt.style.use('seaborn-v0_8-muted')

# times = np.concatenate([np.arange(0,5000,250),np.arange(5000,10000,51)])
# indices = np.zeros(len(times)).astype(int)
# def_spikes = [indices,times]
# input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)
# default_ib = default_neuron_params['ib_n']

# # Using custom()
# #** put in line for synaptic refractory period
# # synaptic structure [synapse][layer][branch][dendrite]
# # one synapse, weight=1, to the 0th layer, branch, and dendrite -> implies soma
# syn_struct = [[[[1]]]] 


# # call a custom neuron with this property
# # kwargs and **kwarg dictionaries also welcome for any neuron parameters
# # here, spiking threshold s_th is set to 1
# # mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100,beta_ni=2*np.pi*1e4,tau_ni=1000)
# mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100,ib_ni=1.7,beta_ni=2*np.pi*1e3,tau_ni=1000000)
# # add input signal to somatic dendrite
# mono_point.synapses[0][0][0][0].add_input(input.signals[0])

# # sim = True will run and record network of nodes for tf duration with dt time steps
# net = network(sim=True,dt=.1,tf=10000,nodes=[mono_point])

# # many plotting options with this funcion
# mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        ### SPD ###
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def spd():
    times = np.arange(0,300,25)
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)

    syn_struct = [[[[1]]]] 
    mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100)
    mono_point.synapses[0][0][0][0].add_input(input.signals[0])
    net = network(sim=True,dt=.1,tf=300,nodes=[mono_point])
    # print(mono_point.neuron.__dict__.keys())
    plt.figure(figsize=(8,4))
    plt.plot(net.t,mono_point.neuron.dend_soma.phi_r,label="SPD Flux")

    plt.plot(input.spike_arrays[1],np.zeros(len(input.spike_arrays[1])),'xr', markersize=5, label='input event')
    # print(mono_point.neuron.__dict__.keys()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.subplots_adjust(right=.8)
    # plt.subplots_adjust(bottom=.25)
    plt.legend()
    plt.xlabel("Simulation Time (ns)",fontsize=16)
    plt.ylabel("Flux ($\phi_0$)",fontsize=16)
    plt.title("SPD Flux Response to Incoming Synaptic Events",fontsize=18)
    plt.subplots_adjust(bottom=.15)
    plt.show()

# spd()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        ### Whole Neuron ###
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def whole_neuron():
    times = np.array([25])
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)

    times = np.array([50])
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input_2 = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)

    syn_struct = [
            [
                [[0]],
                [[1,0]]
            ],
            [
                [[0]],
                [[0,1]]
            ],
    ]
    weights = [[[.42,.48]]]
    mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,weights=weights,ib_n=1.8,s_th=100)
    mono_point.synapses[0][1][0][0].add_input(input.signals[0])
    mono_point.synapses[1][1][0][1].add_input(input_2.signals[0])
    net = network(sim=True,dt=.1,tf=300,nodes=[mono_point])
    # print(mono_point.neuron.__dict__.keys())
    # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=True,SPD=False,dend=True,ref=False,size=(12,6))
    plt.figure(figsize=(6,2))
    # plt.plot(net.t,mono_point.dendrites[1][0][0].phi_r,label="SPD Flux")
    # plt.plot(net.t,mono_point.dendrites[1][0][1].phi_r,label="SPD Flux")
    # plt.plot(net.t,mono_point.dendrites[1][0][0].s,color='#ff7f0e',label="Dendrite 1 Signal")
    # plt.plot(net.t,mono_point.dendrites[1][0][1].s,color='#2ca02c',label="Dendrite 2 Signal")
    plt.plot(net.t,mono_point.dendrites[0][0][0].s,color='#1f77b4',label="Dendrite 2 Signal")

    plt.plot(input.spike_arrays[1],np.zeros(len(input.spike_arrays[1])),'xr', markersize=5, label='input 1')
    plt.plot(input_2.spike_arrays[1],np.zeros(len(input.spike_arrays[1])),'xr', markersize=5, label='input 2')
    # print(mono_point.neuron.__dict__.keys()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.subplots_adjust(right=.8)
    # plt.subplots_adjust(bottom=.25)
    # plt.legend()
    # plt.xlabel("Simulation Time (ns)")
    # plt.ylabel("Flux ($\phi_0$)")
    plt.tick_params(left=False,right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    # plt.title("Signal - Denrite 1")
    # plt.title("Signal - Denrite 2")
    # plt.title("SPD - Input 1")
    # plt.title("SPD - Input 2")
    plt.title("Signal - Somatic Denrite")
    plt.show()

# whole_neuron()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        ### Integration ###
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def integration():
    times = np.arange(0,300,51)
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)

    syn_struct = [[[[.75]]]] 
    mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=.55,beta_ni=2*np.pi*1e3,tau_ni=150,ib_n=2.2)
    mono_point.synapses[0][0][0][0].add_input(input.signals[0])
    net = network(sim=True,dt=.1,tf=300,nodes=[mono_point],new_way=False)
    # print(mono_point.neuron.__dict__.keys())
    mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=False,SPD=False,
                                    dend=True,ref=False,size=(12,6))

# integration()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        ### Leak Rate ###
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def leak():
    times = np.arange(0,300,100)
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)
    syn_struct = [[[[1]]]]
    plt.figure(figsize=(8,6))
    signals = []
    for i in range(1,8):
        mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100,tau_ni=i*20)
        mono_point.synapses[0][0][0][0].add_input(input.signals[0])
        net = network(sim=True,dt=.1,tf=300,nodes=[mono_point])
        # signals.append(mono_point.s)
        plt.plot(net.t,mono_point.neuron.dend__nr_ni.s,label=r"$\tau_{ni}$="+str(i*20))
    # print(mono_point.neuron.__dict__.keys()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.subplots_adjust(right=.8)
    # plt.subplots_adjust(bottom=.25)
    plt.legend()
    plt.xlabel("Simulation Time (ns)")
    plt.ylabel("Signal (Ic)")
    plt.title("Somatic Signal for Different Time Constants")
    plt.show()
    # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=False,SPD=False,dend=True,ref=False)

# leak()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        ### Inductance ###
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def beta():
    times = np.arange(0,300,100)
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)
    syn_struct = [[[[1]]]]
    plt.figure(figsize=(8,6))
    signals = []
    for i in range(2,6):
        mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100,beta_ni=2*np.pi*10**(i),tau_ni=150)
        mono_point.synapses[0][0][0][0].add_input(input.signals[0])
        net = network(sim=True,dt=.1,tf=300,nodes=[mono_point])
        # signals.append(mono_point.s)
        plt.plot(net.t,mono_point.neuron.dend__nr_ni.s,label=r"$\beta_{ni}$"+fr"$=2\pi\cdot 10^{i}$")
    # print(mono_point.neuron.__dict__.keys()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.subplots_adjust(right=.8)
    # plt.subplots_adjust(bottom=.25)
    plt.legend()
    plt.xlabel("Simulation Time (ns)")
    plt.ylabel("Signal (Ic)")
    plt.title("Somatic Signal for Different Inductance Values")
    plt.show()
    # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=False,SPD=False,dend=True,ref=False)

# beta()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        ### Bias ###
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def bias():
    times = np.arange(0,300,100)
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)
    syn_struct = [[[[1]]]]
    # plt.figure(figsize=(12,4))
    plt.figure(figsize=(8,6))
    signals = []
    
    for i in range(1,6):
        mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100,ib_n=1.2+i/5)
        mono_point.synapses[0][0][0][0].add_input(input.signals[0])
        net = network(sim=True,dt=.1,tf=300,nodes=[mono_point])
        # signals.append(mono_point.s)
        plt.plot(net.t,mono_point.neuron.dend__nr_ni.s,label=f"bias current = {np.round(1.2+i/5,2)}")
    # print(mono_point.neuron.__dict__.keys()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.subplots_adjust(right=.8)
    # plt.subplots_adjust(bottom=.15)
    plt.legend()
    plt.xlabel("Simulation Time (ns)")
    plt.ylabel("Signal (Ic)")
    plt.title("Somatic Signal for Different Biasing Values")
    plt.show()
    # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=False,SPD=False,dend=True,ref=False)

# bias()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        ### Modulation ###
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def modulation ():
    spike_times = np.arange(0,500,175)
    input = SuperInput(type='defined', defined_spikes=spike_times)
    params = [1.8,400,2]
    param_lists = [
        np.arange(1.4,2.05,.1),
        np.arange(10,500,50),
        np.arange(2,5,.5)
        ]
    names = [r'Bias Current $i_b$', r'Time Constant $\tau$', r'Inductance $\beta$']
    code_names = [r'$i_b$',r'$\tau$',r'$\beta$']


    fig, axs = plt.subplots(3, 1,figsize=(8,8))
    
    for i,param_list in enumerate(param_lists):
        # plt.figure(figsize=(12,4))
        for p in param_list:
            params = [1.8,200,2]
            params[i] = p

            node = SuperNode(
                s_th    = 1,                     # spiking threshold    
                ib_n    = params[0],             # bias current         
                tau_ni  = params[1],             # time constant        
                beta_ni = 2*np.pi*10**params[2], # inductance parameter 
                ) 
            node.uniform_input(input)
            net = network(sim=True,dt=.1,tf=500,nodes=[node])
            label = f"{code_names[i]}={round(p,2)}"
            axs[i].plot(net.t,node.neuron.dend_soma.s,label=label)
            if i != 2:
                axs[i].set_xticks([])
                # axs[i].set_yticks([])
            axs[i].legend(loc='center left', bbox_to_anchor=(1, .5))
            axs[1].set_ylabel('Integrated Signal',fontsize=14)
            axs[i].set_title(names[i],fontsize=14)
    plt.subplots_adjust(right=.8)
    # plt.subplots_adjust(bottom=.15)
    fig.suptitle("Parameter-Tuning Signal Integration",fontsize=16, y=0.95)
    # axs[0].sharex(axs[1])
    # axs[0].sharey(axs[2])
    # axs[2].set_xticks([1],fontsize=18)
    # plt.tight_layout()
    # plt.title(f'Signal Integrated for Different {names[i]} Values',fontsize=18)
    plt.xlabel('Time (ns)',fontsize=14)
    

    
    plt.show()
    
# modulation()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        ### Flux ###
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def flux():
    times = np.arange(0,300,100)
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)

    syn_struct = [[[[.25]]]] 
    mono_point_low = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100)
    mono_point_low.synapses[0][0][0][0].add_input(input.signals[0])

    syn_struct = [[[[.42]]]] 
    mono_point_above = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100)
    mono_point_above.synapses[0][0][0][0].add_input(input.signals[0])

    syn_struct = [[[[-.42]]]] 
    mono_point_neg = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100)
    mono_point_neg.synapses[0][0][0][0].add_input(input.signals[0])
    nodes=[mono_point_low,mono_point_above,mono_point_neg]
    net = network(sim=True,dt=.1,tf=300,nodes=nodes)

    activity_plot(
        nodes,net,title="Flux Threshold Regimes",input=input,spikes=False,
        phir=True,SPD=False,dend=True,ref=False,size=(8,5),y_range=[-.2,.2]
        )
    # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=True,SPD=False,dend=True,ref=False)
            #     axs[0].set_ylim([-.01,.18])
            #     axs[1].set_ylim([-.01,.22])
            #     axs[2].set_ylim(-.22,.1)
    # mono_point.synapses[0][0][0][0].add_input(input.signals[0])
    # net = network(sim=True,dt=.1,tf=300,nodes=[mono_point])
    # # print(mono_point.neuron.__dict__.keys())
    # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=True,SPD=False,dend=True,ref=False)
    # syn_struct = [[[[-.42]]]] 
    # mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100)
    # mono_point.synapses[0][0][0][0].add_input(input.signals[0])
    # net = network(sim=True,dt=.1,tf=300,nodes=[mono_point])
    # # print(mono_point.neuron.__dict__.keys())
    # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=True,SPD=False,dend=True,ref=False)

# flux()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        ### Rollover ###
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def rollover():
    times = np.arange(0,300,51)
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)
    # plt.figure(figsize=(12,4))
    signals = []
    neurs = {}
    nodes = []
    
    for i in range(8):
        syn_struct = [[[[.5+i*.25]]]]
        neurs[str(i)] = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100)
        neurs[str(i)].synapses[0][0][0][0].add_input(input.signals[0])
        nodes.append(neurs[str(i)])
        # net = network(sim=True,dt=.1,tf=300,nodes=[mono_point])
        # signals.append(mono_point.s)
        # plt.plot(net.t,mono_point.neuron.dend__nr_ni.s,label=f"W_syn = {i}")
        # plt.plot(net.t,mono_point.neuron.dend__nr_ni.phi_r)
    # print(mono_point.neuron.__dict__.keys()

    net = network(sim=True,dt=.1,tf=300,nodes=nodes)
    activity_plot(nodes,net,title="Rollover Effects for Increasing Synaptic Coupling Strength",input=input,
                 phir=True,SPD=False,dend=True,ref=False,spikes=False,size=(8,12),legend=False)
    

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.subplots_adjust(right=.8)
    # plt.subplots_adjust(bottom=.15)
    # plt.xlabel("Simulation Time (ns)")
    # plt.ylabel("Signal (Ic)")
    # plt.title("Somatic Signal for Different Inductance Values")
    # plt.show()
    # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=False,SPD=False,dend=True,ref=False)

    # syn_struct = [[[[2.4]]]]

    # syn_struct = [[[[3.4]]]]
    # mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100)
    # mono_point.synapses[0][0][0][0].add_input(input.signals[0])
    # net = network(sim=True,dt=.01,tf=1000,nodes=[mono_point])
    # # signals.append(mono_point.s)
    # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=True,SPD=False,dend=True,ref=False)

    # # syn_struct = [[[[1]]]]
    # mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100)
    # mono_point.synapses[0][0][0][0].add_input(input.signals[0])
    # net = network(sim=True,dt=.1,tf=300,nodes=[mono_point])
    # # signals.append(mono_point.s)
    # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=True,SPD=False,dend=True,ref=False)


rollover()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        ### Saturation ###
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def saturation():
    times = np.concatenate([np.arange(0,2500,250),np.arange(2500,5000,51)])
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)

    syn_struct = [[[[.5]]]] 
    mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100,ib_ni=1.7,beta_ni=2*np.pi*1e3,tau_ni=1000000)
    print(mono_point.synapse_list[0].__dict__)
    mono_point.synapses[0][0][0][0].add_input(input.signals[0])
    # print(mono_point.synapse_list[0].__dict__)
    # syn = mono_point.synapse_list[0].__dict__
    # if "synaptic_input" in syn:
    #     print("yes")

    times = np.concatenate([np.arange(0,2500,250),np.arange(2500,5000,51)])
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input2 = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)

    syn_struct = [[[[.5]]]] 
    mono_point2 = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100,ib_ni=1.7,beta_ni=2*np.pi*1e4,tau_ni=500)
    # print(mono_point.synapse_list[0].__dict__)
    mono_point2.synapses[0][0][0][0].add_input(input2.signals[0])

    nodes=[mono_point2,mono_point]
    titles=[r"Activity Balanced with Input (Larger $\beta$, Smaller $\tau$)",r"True Saturation (Smaller $\beta$, Larger $\tau$)"]
    net = network(sim=True,dt=.1,tf=5000,nodes=nodes)
    activity_plot(nodes,net,title="Dendritic Saturation vs Balance for Monosynaptic Point Neuron",subtitles=titles,phir=True,input=input,
                                    SPD=False,dend=False,spikes=False,ref=False,size=(10,8))
    # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",phir=True,
                                    # SPD=False,dend=True,ref=False,size=(12,10))
# saturation()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        ### Firing and Refraction ###
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def integration():
    times = np.arange(0,300,51)
    indices = np.zeros(len(times)).astype(int)
    def_spikes = [indices,times]
    input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)

    syn_struct = [[[[1]]]] 
    mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=.5)
    mono_point.synapses[0][0][0][0].add_input(input.signals[0])
    # net = network(sim=True,dt=.01,tf=300,nodes=[mono_point])
    # print(mono_point.neuron.__dict__.keys())
    # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=True,SPD=False,dend=True,ref=True)

    syn_struct = [[[[.75]]]] 
    mono_point2 = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=.55,beta_ni=2*np.pi*1e3,tau_ni=200,ib_n=2.05)

    mono_point2.synapses[0][0][0][0].add_input(input.signals[0])
    nodes=[mono_point,mono_point2]
    net = network(sim=True,dt=.1,tf=300,nodes=[mono_point,mono_point2])

    titles=[r"Low $\beta$ and $\tau$",r"Greater $\beta$,$\tau$, and $i_b$"]
    activity_plot(nodes,net,title="Spiking and Refraction Behavior for Monosynaptic Neuron",subtitles=titles,phir=True,input=input,
                                    SPD=False,dend=False,ref=True,spikes=True,size=(10,8),legend_out=True)
    # print(mono_point.neuron.__dict__.keys())
    # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=True,SPD=False,dend=True,ref=True)

# integration()
