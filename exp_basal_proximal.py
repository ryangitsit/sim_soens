from super_library import NeuralZoo


from params import default_neuron_params
from super_input import SuperInput
from soen_sim import input_signal, synapse, neuron, network
from soen_plotting import raster_plot
import numpy as np
import matplotlib.pyplot as plt

'''
Simple test case for two-neuron predictive-processing method
 - Any function that might be relevant for testing is brought to the surface
 - Adjustable parameters and run calls are at the bottom of the script
 - Synaptic weights, dendritic weights, and neuron parameters all easily adjustable
 - Plotting function gives activity gist for both neurons
 - Just run this script to test
 - For a more thorough overview of NeuralZoo objects, see library_tour.py
'''

def main():
    def gen_input(case='Predictive'):
        '''
        Generates three input cases
            - empty for for any neuron that does not receive basal input
            - full for basal input
            - late start for proximal input
        '''
        # empty input object
        times_empty = np.arange(500,500,50)
        indices_empty = np.zeros(len(times_empty)).astype(int)
        def_spikes_empty = [indices_empty,times_empty]
        input_empty = SuperInput(channels=1, type='defined', defined_spikes=def_spikes_empty, duration=500)

        # basal input activity (one spike per 50ns for 500ns)
        times_basal = np.arange(0,500,50)
        indices_basal = np.zeros(len(times_basal)).astype(int)
        def_spikes_basal = [indices_basal,times_basal]
        input_basal = SuperInput(channels=1, type='defined', defined_spikes=def_spikes_basal, duration=500)

        # proximal input activity (one spike per 50ns starting at 325ns)
        times_proximal = np.arange(325,500,50)
        indices_proximal = np.zeros(len(times_proximal)).astype(int)
        def_spikes_proximal = [indices_proximal,times_proximal]
        input_proximal = SuperInput(channels=1, type='defined', defined_spikes=def_spikes_proximal, duration=500)

        # neither in predictive state
        if case == 'no_basal':
            input_basal = input_empty

        return input_basal, input_proximal, input_empty



    def run_prox_basal_experiment(set,n1,n2,case,input_basal,input_proximal,input_empty):
        '''
        Wires neurons exactly as shown in img/neurons/basal_proximal.png
        - Run two-neuron network and returns results
        '''

        # mutual inhibition
        n1.neuron.add_output(n2.synapses[8][1][0][2])
        n2.neuron.add_output(n1.synapses[8][1][0][2])

        # adding the lateral dendritic connections for each neuron
        if set == 'staircase':
            strength = 0.5
        elif set == 'race':
            strength = 0.1
        n1.dendrites[1][0][1].add_input(n1.dendrites[1][0][0], connection_strength=strength)
        n2.dendrites[1][0][1].add_input(n2.dendrites[1][0][0], connection_strength=strength)

        ### n_1 synapses ###
        # basal synapses 
        if case == "Predictive":
            n1.synapses[0][1][0][0].add_input(input_basal.signals[0])
            n1.synapses[1][1][0][0].add_input(input_basal.signals[0])
            n1.synapses[2][1][0][0].add_input(input_basal.signals[0])
            n1.synapses[3][1][0][0].add_input(input_basal.signals[0])

        elif case == "no_basal":
            n1.synapses[0][1][0][0].add_input(input_basal.signals[0])
            n1.synapses[1][1][0][0].add_input(input_basal.signals[0])
            n1.synapses[2][1][0][0].add_input(input_basal.signals[0])
            n1.synapses[3][1][0][0].add_input(input_basal.signals[0])

        # proximal synapses
        n1.synapses[4][1][0][1].add_input(input_proximal.signals[0])
        n1.synapses[5][1][0][1].add_input(input_proximal.signals[0])
        n1.synapses[6][1][0][1].add_input(input_proximal.signals[0])
        n1.synapses[7][1][0][1].add_input(input_proximal.signals[0])

        # inhibitory synapse
        n1.synapses[8][1][0][2].add_input(input_empty.signals[0])


        ### n_2 synapses ###
        # basal synapses 
        n2.synapses[0][1][0][0].add_input(input_empty.signals[0])
        n2.synapses[1][1][0][0].add_input(input_empty.signals[0])
        n2.synapses[2][1][0][0].add_input(input_empty.signals[0])
        n2.synapses[3][1][0][0].add_input(input_empty.signals[0])

        # proximal synapses
        n2.synapses[4][1][0][1].add_input(input_proximal.signals[0])
        n2.synapses[5][1][0][1].add_input(input_proximal.signals[0])
        n2.synapses[6][1][0][1].add_input(input_proximal.signals[0])
        n2.synapses[7][1][0][1].add_input(input_proximal.signals[0])

        # inhibitory synapse
        n2.synapses[8][1][0][2].add_input(input_empty.signals[0])


        ### create and run network ###
        net = network(sim=True,dt=.1,tf=500,nodes=[n1,n2])

        return net, n1, n2



    def bas_prox_plot(nets,neurons, input_basal,input_prox,title):
        '''
        Plots both neuron activities
        - 
        '''
        fig, axs = plt.subplots(len(nets), len(neurons[0]),figsize=(18,8),
                                sharex=True, sharey=True)
        
        for ii in range(2):
            for jj in range(2):
                # plot somatic signal
                signal = neurons[ii][jj].dendrites[0][0][0].s
                axs[jj,ii].plot(nets[ii].t,signal,  label='soma signal', linewidth=2.5)

                # plot dendritic signals
                dend_names = ['basal', 'proximal', 'inhibitory']
                for i,layer in enumerate(neurons[ii][jj].dendrites):
                    for j,branch in enumerate(layer):
                        for k,dendrite in enumerate(branch):
                            if i == 0 and j == 0 and k ==0:
                                pass
                            else:
                                weight = dendrite.weights[i-1][j][k]
                                dend_s = dendrite.s*weight
                                axs[jj,ii].plot(nets[ii].t,dend_s,'--', label='w * '+dend_names[k])

                spike_times = nets[ii].neurons[neurons[ii][jj].neuron.name].spike_t
                # print(spike_times)

                axs[jj,ii].plot(spike_times,np.ones(len(spike_times))*neurons[ii][jj].neuron.s_th,
                            'xk', markersize=8, label=f'neuron fires')

                axs[jj,ii].axhline(y = neurons[ii][jj].neuron.s_th, 
                                color = 'purple', linestyle = ':',label='Firing Threshold')

                axs[jj,ii].plot(input_prox.spike_arrays[1],np.zeros(len(input_prox.spike_arrays[1])),
                            'xg', markersize=8, label='proximal input event')


        # axs[i].plot(net.t,phi_r,  label='phi_r (soma)')
        axs[0,1].plot(input_basal[1].spike_arrays[1],np.zeros(len(input_basal[1].spike_arrays[1])),
                    'x',color='orange', markersize=8, label='basal input event')

        axs[0,0].set_title(f"No Basal Input for Either Neuron")
        axs[0,1].set_title(f"Basal Input for One Neuron")

        plt.suptitle(f"Non-Predictive vs Predictive State Neuron Couples", fontsize = 14)
        axs[1,0].set_xlabel("Simulation Time (ns)", fontsize = 12)
        axs[1,0].set_ylabel("Signal (Ic)", fontsize = 12)
        # fig.supxlabel("Simulation Time (ns)")
        # fig.supylabel("Signal (Ic)")
        axs[1,0].yaxis.set_label_coords(-.1, 1.1)
        axs[1,0].xaxis.set_label_coords(1.1, -.2)
        axs[0,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=.85)
        plt.subplots_adjust(bottom=.15)
        # plt.legend()
        plt.show()

    def get_params(set):
        if set == 'staircase':
            # synaptic structure (see library tour for details)
            syn_struct = [
                        [
                            [[0]],
                            [[.5,0,0]]
                        ],
                        [
                            [[0]],
                            [[.5,0,0]]
                        ],
                        [
                            [[0]],
                            [[.5,0,0]]
                        ],
                        [
                            [[0]],
                            [[-.5,0,0]]
                        ],

                        [
                            [[0]],
                            [[0,.5,0]]
                        ],
                        [
                            [[0]],
                            [[0,.5,0]]
                        ],
                        [
                            [[0]],
                            [[0,.5,0]]
                        ],
                        [
                            [[0]],
                            [[0,-.5,0]]
                        ],

                        [
                            [[0]],
                            [[0,0,-1]]
                        ],
                    ]

            # dendritic weights
            W = [[[.35,.7,-.1]]]

            tau_ni=500
            tau_di=500
            beta_ni=2*np.pi*1e3

        elif set == 'race':
            syn_struct = [
                        [
                            [[0]],
                            [[.42,0,0]]
                        ],
                        [
                            [[0]],
                            [[.42,0,0]]
                        ],
                        [
                            [[0]],
                            [[.42,0,0]]
                        ],
                        [
                            [[0]],
                            [[-.42,0,0]]
                        ],

                        [
                            [[0]],
                            [[0,.5,0]]
                        ],
                        [
                            [[0]],
                            [[0,.5,0]]
                        ],
                        [
                            [[0]],
                            [[0,.5,0]]
                        ],
                        [
                            [[0]],
                            [[0,-.5,0]]
                        ],

                        [
                            [[0]],
                            [[0,0,-1]]
                        ],
                    ]

            W = [[[.35,.7,-.1]]]

            tau_ni=500
            tau_di=250
            beta_ni=2*np.pi*1e2
        return syn_struct,W,tau_ni,tau_di,beta_ni

    set = 'staircase'
    # set = 'race'
    syn_struct,W,tau_ni,tau_di,beta_ni = get_params(set)


    ### Predictive Case ###
    case='Predictive'

    # generate input signals
    input_basal, input_proximal, input_empty = gen_input(case)

    # generate neurons 
    # any neuron/dendritic parameter can be passed in as keyword argument
    n1 = NeuralZoo(type='custom',synaptic_structure=syn_struct,weights=W,
                s_th=.5,tau_ni=tau_ni,tau_di=tau_di,beta_ni=beta_ni)

    n2 = NeuralZoo(type='custom',synaptic_structure=syn_struct,weights=W,
                s_th=.5,tau_ni=tau_ni,tau_di=tau_di,beta_ni=beta_ni)

    net, n1, n2 = run_prox_basal_experiment(set,n1,n2,case,input_basal,input_proximal,input_empty)
    neurons = [n1,n2]

    ### Non-Predictive Case ###
    input_basal_ = input_empty

    n1_ = NeuralZoo(type='custom',synaptic_structure=syn_struct,weights=W,
                s_th=.5,tau_ni=tau_ni,tau_di=tau_di,beta_ni=beta_ni)

    n2_ = NeuralZoo(type='custom',synaptic_structure=syn_struct,weights=W,
                s_th=.5,tau_ni=tau_ni,tau_di=tau_di,beta_ni=beta_ni)

    net_, n1_, n2_ = run_prox_basal_experiment(set,n1_,n2_,case,input_basal_,input_proximal,input_empty)

    ### Plotting ###
    neurons_ = [n1_,n2_]
    nets = [net_,net]
    NEURONS = [neurons_,neurons]
    input_bas = [input_basal_,input_basal]

    # plot activity for each neuron for each case
    bas_prox_plot(nets,NEURONS,input_bas,input_proximal,case)

if __name__=='__main__':
    main()