import numpy as np
import matplotlib.pyplot as plt

from soen_plotting import raster_plot, activity_plot

from super_input import SuperInput
from params import net_args 

from super_library import NeuralZoo
from super_functions import *
from soen_sim import network, dendrite, HardwareInTheLoop
from soen_component_library import common_synapse



def main():


    def bias_ramp_spike_train():
        '''
        Only for observing bias ramp.  The bias ramp itself is
        instantiated on line 551 of soen_functions.py
            - Everything here functions as introduced in tutorial_library.py
            - The only addition is adding an .ib_ramp and .time_steps attributes 
              to the dendrite in question
        '''
        duration = 10**3
        rate = 10**2
        interval = 10
        dt = 0.1
        times = np.arange(0,duration,rate)
        indices = np.zeros(len(times)).astype(int)
        def_spikes = [indices,times]
        input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)
        syn_struct = [[[[1]]]] 
        mono_point = NeuralZoo(type="custom",synaptic_structure=syn_struct,s_th=100)


        mono_point.ib_ni = 1.4
        mono_point.dendrites[0][0][0].ib_ramp=True
        mono_point.dendrites[0][0][0].time_steps = duration/dt

        mono_point.synapses[0][0][0][0].add_input(input.signals[0])
        net = network(sim=True,dt=dt,tf=duration,nodes=[mono_point],new_way=True)

        mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=False,SPD=False,
                                        dend=True,ref=False,size=(8,6))

        plt.plot(mono_point.dendrites[0][0][0].bias_dynamics[1:])
        plt.show()

    ### uncomment to run
    # bias_ramp_spike_train()


    
    def time_to_signal(duration,interval):
        '''
        Runs bias ramped simulations for monosynaptic point neuron
            - Synaptic events at increasing time values (interval increments)
            - Appends max integrated signal to s_peaks
            - Runs for all beta values in beta
            - Saves plot, slopes (all s_peaks with final array as x-axis)
            - Shows plot
        '''

        spike_times = np.arange(0,duration-150,interval)
        betas = np.arange(3,6,.25)  
        plt.figure(figsize=(12,6))     
        slopes = []
        for beta in betas:
            s_peaks = []
            for spike_time in spike_times:
                
                times = [spike_time]
                indices = np.zeros(len(times)).astype(int)
                def_spikes = [indices,times]
                input = SuperInput(channels=1, type='defined', defined_spikes=def_spikes)

                syn_struct = [[[[1]]]] 
                mono_point = NeuralZoo(
                    type="custom",
                    synaptic_structure=syn_struct,
                    s_th=100,
                    tau_ni=10**6,
                    beta_ni=2*np.pi*10**beta)

                mono_point.ib_ni = 1.4
                mono_point.dendrites[0][0][0].ib_ramp=True
                mono_point.dendrites[0][0][0].time_steps = duration/dt

                mono_point.synapses[0][0][0][0].add_input(input.signals[0])
                net = network(sim=True,dt=dt,tf=duration,nodes=[mono_point],new_way=True)
                # s_peaks.append(np.max(mono_point.dendrites[0][0][0].s))
                s_peaks.append(mono_point.dendrites[0][0][0].s[-1])

                # mono_point.plot_neuron_activity(net,title="Monosynaptic Point Neuron",input=input,phir=True,SPD=False,
                #                                 dend=True,ref=False,size=(8,6))

            slopes.append(s_peaks)
            plt.plot(spike_times/duration,s_peaks,label=r"$\beta_{ni}$"+fr"$=2\pi\cdot 10$^{np.round(beta,2)}")


        slopes.append(spike_times/duration)
        plt.legend()
        plt.title("Max Signal Integration as a Function of Synaptic Timing",fontsize=20)
        plt.ylabel(r"$s_{peak}$",fontsize=18)
        plt.xlabel(r"$t_{synapse} / \Delta t_{ramp}$",fontsize=18)

        picklit(slopes,"time_sig","slopes")
        save_fig(plt,"time_sig","plt_time_to_signal")

        plt.show()



    duration = 10**3  # How long trial is run
    interval = 100    # Interval of synaptic event differences
    dt = 0.1
    time_to_signal(duration,interval)

if __name__=='__main__':
    main()